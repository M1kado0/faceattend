"""POST /v1/search — search by face with liveness first."""

from __future__ import annotations

import os
import uuid
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.ml_client import (
    MLServiceRejectedError,
    MLServiceUnavailableError,
    embed_image,
    verify_passive_liveness,
)
from backend.api.schemas import SearchResponse
from backend.api.services.match_scan import scan_and_persist_matches
from backend.audit.logger import log
from backend.db.models.user import User
from backend.db.session import get_session

load_dotenv()
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")

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

    matches = await scan_and_persist_matches(
        user=user,
        embedding=result.embedding,
        model_version=result.model_version,
        session=session,
        top_k=10,
    )

    return SearchResponse(query_id=str(uuid.uuid4()), matches=matches)
