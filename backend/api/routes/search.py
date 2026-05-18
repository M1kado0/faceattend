"""POST /v1/search — search by face with liveness first."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.api.dependencies import get_current_user
from backend.api.ml_client import (
    MLServiceRejectedError,
    MLServiceUnavailableError,
    embed_image,
    verify_passive_liveness,
)
from backend.api.schemas import Match, SearchResponse
from backend.audit.logger import log
from backend.db.models.user import User
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

    matches = []
    for result_match in results:
        metadata = result_match.metadata
        matches.append(
            Match(
                match_id=result_match.embedding_id,
                source_url=str(metadata.get("source_url", "")),
                source_page=str(metadata.get("source_page", "")),
                score=float(result_match.score),
                crawled_at=metadata.get("crawled_at", datetime.now(UTC)),
                image_thumbnail_url=metadata.get("image_thumbnail_url"),
            )
        )

    await log(actor_id=user.id, actor_type=user.role, action="search.success", target_id=user.id)
    return SearchResponse(query_id=str(uuid.uuid4()), matches=matches)
