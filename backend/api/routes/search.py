"""POST /v1/search — search by face with liveness first."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.ml_client import (
    MLServiceRejectedError,
    MLServiceUnavailableError,
    embed_image,
    verify_passive_liveness,
)
from backend.api.schemas import Match, SearchResponse
from backend.audit.logger import log
from backend.db.models.match import MatchRow
from backend.db.models.user import User
from backend.db.session import get_session
from backend.indexer.store import get_store

load_dotenv()
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
index = get_store()

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    photo: UploadFile,
    liveness_blob: UploadFile,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SearchResponse:
    await log(actor_id=user.id, actor_type=user.role, action="search.attempt", target_id=user.id)

    # AUDIT: liveness must run before embedding to prevent stalking use.
    liveness_bytes = await liveness_blob.read()
    try:
        liveness = await verify_passive_liveness(
            ml_service_url=ML_SERVICE_URL,
            blob=liveness_bytes,
            filename=liveness_blob.filename or "liveness.jpg",
            content_type=liveness_blob.content_type or "application/octet-stream",
        )
    except MLServiceRejectedError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    if not liveness.passed:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.liveness_failed",
            target_id=user.id,
            metadata={"score": liveness.score, "reason": liveness.reason},
        )
        raise HTTPException(403, "liveness_failed")

    photo_bytes = await photo.read()
    try:
        result = await embed_image(
            ml_service_url=ML_SERVICE_URL,
            image=photo_bytes,
            filename=photo.filename or "photo.jpg",
            content_type=photo.content_type or "application/octet-stream",
        )
    except MLServiceRejectedError as exc:
        action = "search.no_faces_detected" if exc.status_code == 422 else "search.image_rejected"
        await log(actor_id=user.id, actor_type=user.role, action=action, target_id=user.id)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    try:
        results = await index.search(
            embedding=result.embedding,
            top_k=10,
            filter={"embedding_model_version": result.model_version},
        )
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.index_error",
            target_id=user.id,
        )
        raise HTTPException(500, "index_error") from exc

    embedding_ids = [result_match.embedding_id for result_match in results]
    existing_matches_by_image_id = {}
    if embedding_ids:
        existing_result = await session.execute(
            select(MatchRow).where(
                MatchRow.user_id == user.id,
                MatchRow.image_id.in_(embedding_ids),
            )
        )
        existing_matches_by_image_id = {
            match.image_id: match for match in existing_result.scalars().all()
        }

    matches = []
    search_created_at = datetime.utcnow()
    has_new_matches = False
    for result_match in results:
        metadata = result_match.metadata
        crawled_at = metadata.get("crawled_at", search_created_at)

        match_row = existing_matches_by_image_id.get(result_match.embedding_id)
        if match_row is None:
            match_row = MatchRow(
                id=str(uuid.uuid4()),
                user_id=user.id,
                image_id=result_match.embedding_id,
                source_url=str(metadata.get("source_url", "")),
                source_page=str(metadata.get("source_page", "")),
                score=float(result_match.score),
                crawled_at=crawled_at,
                notified_at=None,
                created_at=search_created_at,
            )
            session.add(match_row)
            has_new_matches = True

        matches.append(
            Match(
                match_id=match_row.id,
                source_url=match_row.source_url,
                source_page=match_row.source_page,
                score=match_row.score,
                crawled_at=match_row.crawled_at,
                created_at=match_row.created_at,
                image_thumbnail_url=result_match.metadata.get("image_thumbnail_url"),
            )
        )
    if has_new_matches:
        try:
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="search.match_persist_error",
                target_id=user.id,
            )
            raise HTTPException(500, "match_persist_error") from exc

    await log(actor_id=user.id, actor_type=user.role, action="search.success", target_id=user.id)
    return SearchResponse(query_id=str(uuid.uuid4()), matches=matches)
