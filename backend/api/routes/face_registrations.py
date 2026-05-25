"""Face registration routes."""

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
    verify_active_liveness,
)
from backend.api.schemas import FaceRegistrationResponse
from backend.api.services.video_liveness import analyze_video_passive_liveness
from backend.api.video_frames import VideoFrameDecodeError
from backend.audit.logger import log
from backend.db.models.face_registration import FaceRegistration
from backend.db.models.user import User
from backend.db.session import get_session
from backend.indexer.store import get_store

load_dotenv()
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
MAX_ACTIVE_FACE_REGISTRATIONS = 3
PASSIVE_FRAME_SAMPLE_COUNT = int(os.getenv("PASSIVE_FRAME_SAMPLE_COUNT", "10"))
PASSIVE_LIVENESS_PASS_RATIO = float(os.getenv("PASSIVE_LIVENESS_PASS_RATIO", "0.8"))
FACE_VISIBLE_RATIO = float(os.getenv("FACE_VISIBLE_RATIO", "0.8"))
index = get_store()
router = APIRouter()


@router.post("/", response_model=FaceRegistrationResponse)
async def create_face_registration(
    liveness_blob: UploadFile,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> FaceRegistrationResponse:
    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="face_registration.attempt",
        target_id=user.id,
    )

    existing_result = await session.execute(
        select(FaceRegistration).where(FaceRegistration.user_id == user.id)
    )
    existing_registrations = existing_result.scalars().all()
    if len(existing_registrations) >= MAX_ACTIVE_FACE_REGISTRATIONS:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.limit_reached",
            target_id=user.id,
            metadata={"max_active_face_registrations": MAX_ACTIVE_FACE_REGISTRATIONS},
        )
        raise HTTPException(status_code=409, detail="face_registration_limit_reached")

    # AUDIT: liveness must run before embedding to prevent non-consensual face_registration.
    liveness_bytes = await liveness_blob.read()
    try:
        liveness = await verify_active_liveness(
            ml_service_url=ML_SERVICE_URL,
            blob=liveness_bytes,
            challenge="blink_twice",
            filename=liveness_blob.filename or "liveness.webm",
            content_type=liveness_blob.content_type or "video/webm",
        )
    except MLServiceRejectedError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    if not liveness.passed:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.liveness_failed",
            target_id=user.id,
            metadata={"score": liveness.score, "reason": liveness.reason},
        )
        raise HTTPException(403, "liveness_failed")

    try:
        passive_summary = await analyze_video_passive_liveness(
            ml_service_url=ML_SERVICE_URL,
            video=liveness_bytes,
            max_frames=PASSIVE_FRAME_SAMPLE_COUNT,
        )
    except VideoFrameDecodeError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.video_decode_failed",
            target_id=user.id,
            metadata={"reason": str(exc)},
        )
        raise HTTPException(400, "video_decode_failed") from exc
    except MLServiceRejectedError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.passive_liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="face_registration.video_passive_liveness_summary",
        target_id=user.id,
        metadata={
            "sampled_frames": passive_summary.sampled_frames,
            "visible_faces": passive_summary.visible_faces,
            "passive_passes": passive_summary.passive_passes,
        },
    )

    if passive_summary.passive_liveness_pass_ratio < PASSIVE_LIVENESS_PASS_RATIO:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.passive_liveness_failed",
            target_id=user.id,
            metadata={"passive_liveness_pass_ratio": passive_summary.passive_liveness_pass_ratio},
        )
        raise HTTPException(403, "passive_liveness_failed")

    if passive_summary.face_visible_ratio < FACE_VISIBLE_RATIO:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.face_not_visible_enough",
            target_id=user.id,
            metadata={"face_visible_ratio": passive_summary.face_visible_ratio},
        )
        raise HTTPException(403, "face_not_visible_enough")

    if passive_summary.embedding_frame is None:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.no_live_embedding_frame",
            target_id=user.id,
        )
        raise HTTPException(403, "passive_liveness_failed")

    try:
        result = await embed_image(
            ml_service_url=ML_SERVICE_URL,
            image=passive_summary.embedding_frame,
            filename="liveness-frame.jpg",
            content_type="image/jpeg",
        )
    except MLServiceRejectedError as exc:
        action = (
            "face_registration.no_faces_detected"
            if exc.status_code == 422
            else "face_registration.image_rejected"
        )
        await log(actor_id=user.id, actor_type=user.role, action=action, target_id=user.id)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.ml_error",
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
            action="face_registration.index_error",
            target_id=user.id,
        )
        raise HTTPException(500, "index_error") from exc

    face_registration = FaceRegistration(
        id=str(uuid.uuid4()),
        user_id=user.id,
        embedding_id=embedding_id,
        embedding_model_version=result.model_version,
    )
    try:
        session.add(face_registration)
        await session.commit()
        await session.refresh(face_registration)
    except Exception as exc:
        await session.rollback()
        try:
            await index.delete(embedding_id=embedding_id)
        except Exception as cleanup_exc:
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="face_registration.cleanup_failed",
                target_id=user.id,
                metadata={
                    "embedding_id": face_registration.embedding_id,
                    "reason": "db_commit_failed_after_index_add",
                    "cleanup_error": str(cleanup_exc),
                },
            )
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.db_error",
            target_id=user.id,
            metadata={"embedding_id": embedding_id},
        )
        raise HTTPException(500, "db_error") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="face_registration.success",
        target_id=user.id,
    )

    return FaceRegistrationResponse(
        registration_id=face_registration.id, embedding_model_version=result.model_version
    )


@router.get("/")
async def list_face_registrations(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    result = await session.execute(
        select(FaceRegistration).where(FaceRegistration.user_id == user.id)
    )
    face_registrations = result.scalars().all()
    return [
        {
            "id": face_registration.id,
            "embedding_id": face_registration.embedding_id,
            "embedding_model_version": face_registration.embedding_model_version,
            "created_at": face_registration.created_at,
        }
        for face_registration in face_registrations
    ]


@router.delete("/{registration_id}")
async def delete_face_registration(
    registration_id: str,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    # AUDIT: face_registration.delete
    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="face_registration.delete.attempt",
        target_id=user.id,
        metadata={"registration_id": registration_id},
    )
    result = await session.execute(
        select(FaceRegistration).where(
            FaceRegistration.id == registration_id,
            FaceRegistration.user_id == user.id,
        )
    )
    face_registration = result.scalar_one_or_none()
    if not face_registration:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.delete.not_found",
            target_id=user.id,
            metadata={"registration_id": registration_id},
        )
        raise HTTPException(status_code=404, detail="face_registration_not_found")

    try:
        await index.delete(embedding_id=face_registration.embedding_id)
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.delete.index_error",
            target_id=user.id,
            metadata={
                "registration_id": face_registration.id,
                "embedding_id": face_registration.embedding_id,
            },
        )
        raise HTTPException(500, "index_error") from exc

    try:
        await session.delete(face_registration)
        await session.commit()
    except Exception as exc:
        await session.rollback()

        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="face_registration.delete.db_error",
            target_id=user.id,
            metadata={
                "registration_id": registration_id,
                "embedding_id": face_registration.embedding_id,
                "risk": "index_deleted_but_db_delete_failed",
            },
        )
        raise HTTPException(500, "db_error") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="face_registration.delete.success",
        target_id=user.id,
        metadata={
            "registration_id": face_registration.id,
            "embedding_id": face_registration.embedding_id,
        },
    )

    return {"status": "deleted"}
