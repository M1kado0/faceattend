"""Headless five-point face alignment for InsightFace embeddings."""

from __future__ import annotations

import numpy as np
from insightface.utils import face_align  # type: ignore[import-untyped]

from faceattend.vision.types import FaceObservation, Frame, UInt8Array


class AlignmentError(ValueError):
    """Raised when a frame or landmark set cannot be aligned."""


class InsightFaceAligner:
    """Align a detected face to the ArcFace 112-by-112 input convention."""

    def __init__(self, *, image_size: int = 112) -> None:
        if image_size <= 0 or image_size % 112 != 0:
            raise ValueError("image_size must be a positive multiple of 112")
        self.image_size = image_size

    def align(self, frame: Frame, face: FaceObservation) -> UInt8Array:
        image = np.asarray(frame.pixels)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise AlignmentError("frame must be an HWC uint8 BGR image")

        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2):
            raise AlignmentError("alignment requires exactly five 2D landmarks")
        if not np.isfinite(landmarks).all():
            raise AlignmentError("landmarks must be finite")

        aligned = face_align.norm_crop(
            image,
            landmark=np.ascontiguousarray(landmarks),
            image_size=self.image_size,
        )
        result = np.asarray(aligned, dtype=np.uint8)
        if result.shape != (self.image_size, self.image_size, 3):
            raise AlignmentError("alignment produced an unexpected image shape")
        return np.ascontiguousarray(result)
