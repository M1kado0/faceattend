"""Thin HTTP wrapper for the non-Triton serving path (port 8003).

Selected by INFERENCE_BACKEND=ray. For triton, requests go straight to Triton.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile

from ml.pipeline.face import (
    EmptyEmbeddingError,
    ImageDecodeError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    embed_face,
    verify_passive_liveness,
)

app = FastAPI(title="FaceGuard ML Service", version="0.1.0")



@app.post("/v1/embed")
async def embed(image: UploadFile) -> dict:
    raw = await image.read()
    try:
        result = embed_face(raw)
    except ImageDecodeError as exc:
        raise HTTPException(400, "could_not_decode_image") from exc
    except NoFaceDetectedError as exc:
        raise HTTPException(422, "no_faces_detected") from exc
    except MultipleFacesDetectedError as exc:
        raise HTTPException(422, "multiple_faces_detected") from exc
    except EmptyEmbeddingError as exc:
        raise HTTPException(400, "empty_embedding") from exc

    return {
        "embedding": result.embedding.tolist(),
        "model_version": result.model_version,
    }


@app.post("/v1/liveness/passive")
async def liveness_passive(blob: UploadFile) -> dict:
    raw = await blob.read()
    try:
        result = verify_passive_liveness(raw)
    except ImageDecodeError as exc:
        raise HTTPException(400, "could_not_decode_image") from exc
    except NoFaceDetectedError as exc:
        raise HTTPException(422, "no_faces_detected") from exc
    except MultipleFacesDetectedError as exc:
        raise HTTPException(422, "multiple_faces_detected") from exc
    return result


@app.post("/v1/liveness/active")
async def liveness_active(blob: UploadFile, challenge: str) -> dict:
    raise NotImplementedError


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
