"""POST /v1/search — search by face (liveness REQUIRED)."""
from fastapi import APIRouter, Depends, UploadFile

from backend.api.dependencies import get_current_user
from backend.api.schemas import SearchResponse
from backend.indexer.store import get_store
import httpx
from dotenv import load_dotenv
import os
import uuid
import numpy as np

load_dotenv()
ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
index = get_store()

router = APIRouter()


@router.post("/search")
async def search(
    photo: UploadFile,
    liveness_blob: UploadFile,
    user=Depends(get_current_user),
) -> list[dict]:
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
    embedding = np.array(data["embedding"], dtype=np.float32)
    matches = await index.search(embedding=embedding, top_k=10)
    query_id = str(uuid.uuid4())

    res = []
    for match in matches:
        res.append({
            "match_id": match.image_id,
            "score": float(match.score),
        })
    return res
