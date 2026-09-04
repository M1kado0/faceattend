"""Tests for the five-point face alignment adapter."""

import numpy as np
import pytest

from faceattend.vision.alignment import AlignmentError, InsightFaceAligner
from faceattend.vision.types import BoundingBox, FaceObservation, FaceQuality, Frame


def _face(landmarks: np.ndarray) -> FaceObservation:
    return FaceObservation(
        bbox=BoundingBox(100.0, 80.0, 300.0, 280.0),
        detector_score=0.99,
        landmarks=landmarks.astype(np.float32),
        quality=FaceQuality(True, 0.0, 0.0, 0.1),
    )


def _frame() -> Frame:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(image, captured_at_ns=1, sequence_id=1)


def test_align_returns_arcface_sized_bgr_image() -> None:
    face = _face(
        np.array(
            [[170.0, 160.0], [230.0, 160.0], [200.0, 195.0], [180.0, 230.0], [220.0, 230.0]]
        )
    )

    aligned = InsightFaceAligner().align(_frame(), face)

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert aligned.flags.c_contiguous


@pytest.mark.parametrize(
    "landmarks",
    [np.zeros((4, 2), dtype=np.float32), np.zeros((5, 3), dtype=np.float32)],
)
def test_align_rejects_wrong_landmark_shape(landmarks: np.ndarray) -> None:
    with pytest.raises(AlignmentError, match="five"):
        InsightFaceAligner().align(_frame(), _face(landmarks))


def test_align_rejects_nonfinite_landmarks() -> None:
    landmarks = np.zeros((5, 2), dtype=np.float32)
    landmarks[0, 0] = np.nan

    with pytest.raises(AlignmentError, match="finite"):
        InsightFaceAligner().align(_frame(), _face(landmarks))


def test_align_rejects_invalid_frame() -> None:
    invalid_frame = Frame(np.zeros((480, 640), dtype=np.uint8), 1, 1)  # type: ignore[arg-type]
    landmarks = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(AlignmentError, match="HWC"):
        InsightFaceAligner().align(invalid_frame, _face(landmarks))
