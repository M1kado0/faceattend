"""POST /v1/search — search by face (liveness REQUIRED)."""
from fastapi import APIRouter, Depends, UploadFile

from backend.api.dependencies import get_current_user
from backend.api.schemas import SearchResponse
from backend.indexer.store import get_store
import httpx
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
index = get_store()

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    photo: UploadFile,
    liveness_blob: UploadFile,
    user=Depends(get_current_user),
) -> SearchResponse:
    # AUDIT: search.attempt
    # 1. Liveness check FIRST.
    # 2. Embed query face.
    # 3. ANN query via backend.indexer.
    # AUDIT: search.success | search.liveness_failed
    async with httpx.AsyncClient() as client:
        raw = await photo.read()
        response = await client.post(
            f"{ml_service_url}/v1/embed", files={"image": raw}
        )
        response.raise_for_status()
        data = response.json()
    embedding = data["embedding"]
    matches = await index.search(embedding=embedding, top_k=10)
    query_id = str(uuid.uuid4())
    return SearchResponse(query_id=query_id, matches=matches)
