"""Thin HTTP wrapper for the non-Triton serving path (port 8003).

Selected by INFERENCE_BACKEND=ray. For triton, requests go straight to Triton.
"""

from __future__ import annotations

from fastapi import FastAPI, Form, HTTPException, UploadFile

from ml.pipeline.face import (
    EmptyEmbeddingError,
    ImageDecodeError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    embed_face,
    verify_active_liveness,
    verify_passive_liveness,
)

app = FastAPI(title="FaceAttend ML Service", version="0.1.0")


def _raise_for_active_liveness_service_error(result: dict) -> None:
    """Map invalid requests and service problems to HTTP errors."""
    label = result.get("label")
    if label in {"video_decode_failed", "unsupported_challenge"}:
        raise HTTPException(400, result)
    if label == "model_not_found":
        raise HTTPException(503, result)


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
async def liveness_active(
    blob: UploadFile,
    challenge: str = Form(default="blink_twice"),
) -> dict:
    raw = await blob.read()
    try:
        result = verify_active_liveness(raw, challenge)
    except Exception as exc:
        raise HTTPException(500, "active_liveness_failed") from exc
    _raise_for_active_liveness_service_error(result)
    return result


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
