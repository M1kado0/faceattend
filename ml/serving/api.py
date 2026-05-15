"""Thin HTTP wrapper for the non-Triton serving path (port 8003).

Selected by INFERENCE_BACKEND=ray. For triton, requests go straight to Triton.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from insightface.app import FaceAnalysis

app = FastAPI(title="FaceGuard ML Service", version="0.1.0")
_model: FaceAnalysis | None = None

def get_model() -> FaceAnalysis:
    global _model
    if _model is None:
        _model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _model.prepare(ctx_id=-1, det_size=(640, 640))
    return _model

_model = get_model()

@app.post("/v1/embed")
async def embed(image: UploadFile) -> dict:
    raw = await image.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could_not_decode_image")
    
    faces = _model.get(img)
    if not faces:
        raise HTTPException(422, "no_faces_detected")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    vec = face.embedding.astype(np.float32)
    vec /= np.linalg.norm(vec)

    return {"embedding": vec.tolist(), "model_version": "arcface-r100-v1"}


@app.post("/v1/detect")
async def detect(image: UploadFile) -> dict:
    raise NotImplementedError





@app.post("/v1/liveness/passive")
async def liveness_passive(blob: UploadFile) -> dict:
    raise NotImplementedError


@app.post("/v1/liveness/active")
async def liveness_active(blob: UploadFile, challenge: str) -> dict:
    raise NotImplementedError


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
