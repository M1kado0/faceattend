import os
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from ml.liveness.active import ActiveLivenessChecker
from ml.liveness.challenge import ActiveLivenessChallenge
from ml.liveness.passive import PassiveLivenessChecker

load_dotenv()
liveness_model_path = os.getenv("LIVENESS_MODEL_PATH")
active_liveness_model_path = os.getenv(
    "ACTIVE_LIVENESS_MODEL_PATH",
    "./models/face_landmarker.task",
)
liveness_threshold_passive = float(os.getenv("LIVENESS_THRESHOLD_PASSIVE", 0.85))


@dataclass(frozen=True)
class FaceEmbeddingResult:
    embedding: np.ndarray
    model_version: str
    face_count: int
    bbox: list[float]
    detector_score: float | None = None

class FacePipelineError(Exception):
    """Base class for expected Face Pipeline service failures."""

class ImageDecodeError(FacePipelineError):
    """Raised when the image cannot be decoded"""

class NoFaceDetectedError(FacePipelineError):
    """Raised when the no face can be detected"""

class MultipleFacesDetectedError(FacePipelineError):
    """Raised when multiple faces are detected"""

class EmptyEmbeddingError(FacePipelineError):
    """Raised when the embedding is empty"""


_model: FaceAnalysis | None = None
_passive_liveness_checker: PassiveLivenessChecker | None = None
_active_liveness_checker: ActiveLivenessChecker | None = None


def get_face_analyzer() -> FaceAnalysis:
    global _model
    if _model is None:
        _model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _model.prepare(ctx_id=-1, det_size=(640, 640))
    return _model


def get_passive_liveness_checker() -> PassiveLivenessChecker:
    global _passive_liveness_checker
    if _passive_liveness_checker is None:
        _passive_liveness_checker = PassiveLivenessChecker(
            liveness_model_path,
            liveness_threshold_passive,
        )
    return _passive_liveness_checker


def get_active_liveness_checker() -> ActiveLivenessChecker:
    global _active_liveness_checker
    if _active_liveness_checker is None:
        _active_liveness_checker = ActiveLivenessChecker(active_liveness_model_path)
    return _active_liveness_checker


def decode_image(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(400, "could_not_decode_image")
    return img


def detect_faces(image: np.ndarray) -> list:
    _model = get_face_analyzer()
    return _model.get(image)


def get_single_face(faces: list) -> Any:
    if not faces:
        raise NoFaceDetectedError(422, "no_faces_detected")
    if len(faces) > 1:
        raise MultipleFacesDetectedError(422, "multiple_faces_detected")
    return faces[0]


def _detector_score(face) -> float | None:
    if face.score is None:
        return None
    return float(face.score)


def embed_face(raw: bytes) -> FaceEmbeddingResult:
    img = decode_image(raw)
    faces = detect_faces(img)
    face = get_single_face(faces)
    vec = face.embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise EmptyEmbeddingError(400, "empty_embedding")
    vec /= norm
    return FaceEmbeddingResult(
        embedding=vec,
        model_version="arcface-r100-v1",
        face_count=1,
        bbox=[float(v) for v in face.bbox],
        detector_score=_detector_score(face),
    )


def verify_passive_liveness(raw: bytes):
    img = decode_image(raw)
    faces = detect_faces(img)
    face = get_single_face(faces)
    bbox_xyxy = [float(v) for v in face.bbox]
    result = get_passive_liveness_checker().check(img, bbox_xyxy)
    return asdict(result)


def verify_active_liveness(raw: bytes, challenge: ActiveLivenessChallenge | str):
    result = get_active_liveness_checker().check(raw, challenge)
    return asdict(result)
