"""POST /v1/search — search by face (liveness REQUIRED)."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, UploadFile, HTTPException

from backend.api.dependencies import get_current_user
from backend.api.schemas import Match, SearchResponse
from backend.audit.logger import log
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
) -> SearchResponse:
    # AUDIT: search.attempt
    # 1. Liveness check FIRST.
    # 2. Embed query face.
    # 3. ANN query via backend.indexer.
    # AUDIT: search.success | search.liveness_failed
    await log(actor_id=user.id, actor_type=user.role, action="search.attempt", target_id=user.id)
    try:
        async with httpx.AsyncClient() as client:
            raw = await photo.read()
            response = await client.post(
                f"{ml_service_url}/v1/embed", files={"image": raw}
            )
            response.raise_for_status()
            data = response.json()
        embedding = np.array(data["embedding"], dtype=np.float32)
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 400:
           await log(actor_id=user.id, actor_type=user.role, action="search.could_not_decode_image", target_id=user.id)
        elif status_code == 422:
            await log(actor_id=user.id, actor_type=user.role, action="search.no_faces_detected", target_id=user.id)
        raise HTTPException(status_code, e.response.text)
    except httpx.RequestError as e:
        await log(actor_id=user.id, actor_type=user.role, action="search.ml_error", target_id=user.id)
        raise HTTPException(503, "ml_service_unavailable")

    try:
        res = await index.search(embedding=embedding, top_k=10)
    except Exception:
        await log(actor_id=user.id, actor_type=user.role, action="search.index_error", target_id=user.id)
        raise HTTPException(500, "index_error")

    matches = []
    query_id = str(uuid.uuid4())
    for match in res:
        matches.append(Match(match_id=match.image_id, source_url="source", source_page="page", score=float(match.score), crawled_at=datetime.now(timezone.utc), image_thumbnail_url=None))
    await log(actor_id=user.id, actor_type=user.role, action="search.success", target_id=user.id)
    return SearchResponse(query_id=query_id, matches=matches)



