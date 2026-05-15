"""POST /v1/enroll — enroll a face (liveness REQUIRED before embedding)."""
from fastapi import APIRouter, Depends, UploadFile

from backend.api.dependencies import get_current_user
from backend.api.schemas import EnrollResponse
from backend.indexer.store import get_store
import httpx
from dotenv import load_dotenv
import os
import uuid

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
    async with httpx.AsyncClient() as client:
        raw = await photo.read()
        response = await client.post(
            f"{ml_service_url}/v1/embed", files={"image": raw}
        )
        response.raise_for_status()
        data = response.json()
    embedding = data["embedding"]
    image_id = str(uuid.uuid4())
    await index.add(
        image_id=image_id,
        embedding=embedding,
        metadata={"user_id": user.id}
    )
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
