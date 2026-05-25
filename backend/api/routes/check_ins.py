"""POST /v1/check-ins — check in by face with liveness first."""

from __future__ import annotations

import os
import uuid
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.ml_client import (
    EmbeddingResult,
    MLServiceRejectedError,
    MLServiceUnavailableError,
    embed_image,
    verify_active_liveness,
)
from backend.api.schemas import CheckInResponse
from backend.api.services.attendance_record_scan import scan_best_and_persist_attendance_record
from backend.api.services.video_liveness import analyze_video_passive_liveness
from backend.api.video_frames import VideoFrameDecodeError
from backend.audit.logger import log
from backend.db.models.attendance_session import AttendanceSessionRow
from backend.db.models.user import User
from backend.db.session import get_session

load_dotenv()
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
PASSIVE_FRAME_SAMPLE_COUNT = int(os.getenv("PASSIVE_FRAME_SAMPLE_COUNT", "10"))
PASSIVE_LIVENESS_PASS_RATIO = float(os.getenv("PASSIVE_LIVENESS_PASS_RATIO", "0.8"))
FACE_VISIBLE_RATIO = float(os.getenv("FACE_VISIBLE_RATIO", "0.8"))
CHECK_IN_EMBEDDING_FRAME_COUNT = int(os.getenv("CHECK_IN_EMBEDDING_FRAME_COUNT", "3"))

router = APIRouter()


@router.post("/check-ins", response_model=CheckInResponse)
async def check_in(
    liveness_blob: UploadFile,
    session_id: Annotated[str, Form()],
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CheckInResponse:
    await log(actor_id=user.id, actor_type=user.role, action="check_in.attempt", target_id=user.id)

    session_result = await session.execute(
        select(AttendanceSessionRow).where(
            AttendanceSessionRow.id == session_id,
            AttendanceSessionRow.user_id == user.id,
            AttendanceSessionRow.status == "open",
        )
    )
    attendance_session = session_result.scalar_one_or_none()
    if attendance_session is None:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.attendance_session_not_found",
            target_id=session_id,
        )
        raise HTTPException(status_code=404, detail="attendance_session_not_found")

    # AUDIT: liveness must run before embedding to prevent stalking use.
    liveness_bytes = await liveness_blob.read()
    try:
        liveness = await verify_active_liveness(
            ml_service_url=ML_SERVICE_URL,
            blob=liveness_bytes,
            challenge="blink_twice",
            filename=liveness_blob.filename or "liveness.webm",
            content_type=liveness_blob.content_type or "application/octet-stream",
        )
    except MLServiceRejectedError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    if not liveness.passed:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.liveness_failed",
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
            action="check_in.video_decode_failed",
            target_id=user.id,
            metadata={"reason": str(exc)},
        )
        raise HTTPException(400, "video_decode_failed") from exc
    except MLServiceRejectedError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.passive_liveness_rejected",
            target_id=user.id,
            metadata={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="check_in.video_passive_liveness_summary",
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
            action="check_in.passive_liveness_failed",
            target_id=user.id,
            metadata={"passive_liveness_pass_ratio": passive_summary.passive_liveness_pass_ratio},
        )
        raise HTTPException(403, "passive_liveness_failed")

    if passive_summary.face_visible_ratio < FACE_VISIBLE_RATIO:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.face_not_visible_enough",
            target_id=user.id,
            metadata={"face_visible_ratio": passive_summary.face_visible_ratio},
        )
        raise HTTPException(403, "face_not_visible_enough")

    embedding_frames = list(passive_summary.embedding_frames[:CHECK_IN_EMBEDDING_FRAME_COUNT])
    if not embedding_frames:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.no_live_embedding_frame",
            target_id=user.id,
        )
        raise HTTPException(403, "passive_liveness_failed")

    try:
        embedding_results: list[EmbeddingResult] = []
        for index, frame in enumerate(embedding_frames):
            embedding_results.append(
                await embed_image(
                    ml_service_url=ML_SERVICE_URL,
                    image=frame,
                    filename=f"liveness-frame-{index}.jpg",
                    content_type="image/jpeg",
                )
            )
    except MLServiceRejectedError as exc:
        action = (
            "check_in.no_faces_detected" if exc.status_code == 422 else "check_in.image_rejected"
        )
        await log(actor_id=user.id, actor_type=user.role, action=action, target_id=user.id)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except MLServiceUnavailableError as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.ml_error",
            target_id=user.id,
        )
        raise HTTPException(503, "ml_service_unavailable") from exc

    model_version = embedding_results[0].model_version
    attendance_records = await scan_best_and_persist_attendance_record(
        user=user,
        embeddings=[result.embedding for result in embedding_results],
        model_version=model_version,
        session=session,
        attendance_session_id=attendance_session.id,
        top_k=1,
    )
    if not attendance_records:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.identity_not_matched",
            target_id=user.id,
        )
        raise HTTPException(403, "identity_not_matched")

    return CheckInResponse(
        query_id=str(uuid.uuid4()),
        attendance_records=attendance_records,
    )
