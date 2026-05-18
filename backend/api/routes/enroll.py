"""POST /v1/enroll — enroll a face with liveness first."""

from __future__ import annotations

import os
import uuid
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
from backend.api.schemas import EnrollResponse
from backend.audit.logger import log
from backend.db.models.enrollment import Enrollment
from backend.db.models.user import User
from backend.db.session import get_session
from backend.indexer.store import get_store

load_dotenv()
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
index = get_store()
router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    photo: UploadFile,
    liveness_blob: UploadFile,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EnrollResponse:
    await log(actor_id=user.id, actor_type=user.role, action="enroll.attempt", target_id=user.id)

    # AUDIT: liveness must run before embedding to prevent non-consensual enrollment.
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
            action="enroll.liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    if not liveness.passed:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.liveness_failed",
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
        action = "enroll.no_faces_detected" if exc.status_code == 422 else "enroll.image_rejected"
        await log(actor_id=user.id, actor_type=user.role, action=action, target_id=user.id)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    embedding_id = str(uuid.uuid4())
    try:
        await index.add(
            embedding_id=embedding_id,
            embedding=result.embedding,
            metadata={"user_id": user.id, "embedding_model_version": result.model_version},
        )
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.index_error",
            target_id=user.id,
        )
        raise HTTPException(500, "index_error") from exc

    enrollment = Enrollment(
        id=str(uuid.uuid4()),
        user_id=user.id,
        embedding_id=embedding_id,
        embedding_model_version=result.model_version,
    )
    try:
        session.add(enrollment)
        await session.commit()
        await session.refresh(enrollment)
    except Exception as exc:
        await session.rollback()
        try:
            await index.delete(embedding_id=embedding_id)
        except Exception as cleanup_exc:
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="enroll.cleanup_failed",
                target_id=user.id,
                metadata={
                    "embedding_id": enrollment.embedding_id,
                    "reason": "db_commit_failed_after_index_add",
                    "cleanup_error": str(cleanup_exc),
                },
            )
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.db_error",
            target_id=user.id,
            metadata={"embedding_id": embedding_id},
        )
        raise HTTPException(500, "db_error") from exc

    await log(actor_id=user.id, actor_type=user.role, action="enroll.success", target_id=user.id)
    return EnrollResponse(enrollment_id=enrollment.id, embedding_model_version=result.model_version)


@router.get("/enrollments")
async def list_enrollments(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    result = await session.execute(select(Enrollment).where(Enrollment.user_id == user.id))
    enrollments = result.scalars().all()
    return [
        {
            "id": enrollment.id,
            "embedding_id": enrollment.embedding_id,
            "embedding_model_version": enrollment.embedding_model_version,
            "created_at": enrollment.created_at,
        }
        for enrollment in enrollments
    ]


@router.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(
    enrollment_id: str,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    # AUDIT: enroll.delete
    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="enroll.delete.attempt",
        target_id=user.id,
        metadata={"enrollment_id": enrollment_id},
    )
    result = await session.execute(
        select(Enrollment).where(Enrollment.id == enrollment_id, Enrollment.user_id == user.id)
    )
    enrollment = result.scalar_one_or_none()
    if not enrollment:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.delete.not_found",
            target_id=user.id,
            metadata={"enrollment_id": enrollment_id},
        )
        raise HTTPException(status_code=404, detail="enrollment_not_found")

    try:
        await index.delete(embedding_id=enrollment.embedding_id)
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.delete.index_error",
            target_id=user.id,
            metadata={
                "enrollment_id": enrollment.id,
                "embedding_id": enrollment.embedding_id,
            },
        )
        raise HTTPException(500, "index_error") from exc

    try:
        await session.delete(enrollment)
        await session.commit()
    except Exception as exc:
        await session.rollback()

        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="enroll.delete.db_error",
            target_id=user.id,
            metadata={
                "enrollment_id": enrollment_id,
                "embedding_id": enrollment.embedding_id,
                "risk": "index_deleted_but_db_delete_failed",
            },
        )
        raise HTTPException(500, "db_error") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="enroll.delete.success",
        target_id=user.id,
        metadata={
            "enrollment_id": enrollment.id,
            "embedding_id": enrollment.embedding_id,
        },
    )

    return {"status": "deleted"}
