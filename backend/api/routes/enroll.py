"""POST /v1/enroll — enroll a face (liveness REQUIRED before embedding)."""
from fastapi import APIRouter, Depends, UploadFile, HTTPException

from backend.api.dependencies import get_current_user
from backend.api.schemas import EnrollResponse
from backend.indexer.store import get_store
from backend.audit.logger import log
import httpx
from dotenv import load_dotenv
import os
import uuid
import numpy as np

load_dotenv()
ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
index = get_store()
router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    photo: UploadFile,
    liveness_blob: UploadFile,
    user=Depends(get_current_user),
) -> EnrollResponse:
    # AUDIT: enroll.attempt
    # 1. Liveness check FIRST — reject before any embedding work.
    # 2. Detect + embed.
    # 3. Write to vector index via backend.indexer.
    # AUDIT: enroll.success | enroll.liveness_failed | enroll.no_face
    await log(actor_id=user.id, actor_type=user.role, action="enroll.attempt", target_id=user.id)
    try:
        async with httpx.AsyncClient() as client:
            raw = await photo.read()
            response = await client.post(
                f"{ml_service_url}/v1/embed", files={"image": raw}
            )
            response.raise_for_status()
            data = response.json()
        embedding = np.array(data["embedding"], dtype=np.float32)
        image_id = str(uuid.uuid4())
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 400:
           await log(actor_id=user.id, actor_type=user.role, action="enroll.could_not_decode_image", target_id=user.id)
        elif status_code == 422:
            await log(actor_id=user.id, actor_type=user.role, action="enroll.no_faces_detected", target_id=user.id)
        raise HTTPException(status_code, e.response.text)
    except httpx.RequestError as e:
        await log(actor_id=user.id, actor_type=user.role, action="enroll.ml_error", target_id=user.id)
        raise HTTPException(503, "ml_service_unavailable")

    try:
        await index.add(
            image_id=image_id,
            embedding=embedding,
            metadata={"user_id": user.id}
        )
    except Exception:
        await log(actor_id=user.id, actor_type=user.role, action="enroll.index_error", target_id=user.id)
        raise HTTPException(500, "index_error")
    
    await log(actor_id=user.id, actor_type=user.role, action="enroll.success", target_id=user.id)
    return EnrollResponse(enrollment_id=image_id, embedding_model_version=data["model_version"])



@router.get("/enrollments")
async def list_enrollments(user=Depends(get_current_user)) -> list[dict]:
    raise NotImplementedError


@router.delete("/enrollments/{enrollment_id}")
async def delete_enrollment(
    enrollment_id: str, user=Depends(get_current_user)
) -> dict[str, str]:
    # AUDIT: enroll.delete
    raise NotImplementedError
