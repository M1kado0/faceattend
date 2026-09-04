"""Tests for the injectable InsightFace detector adapter."""

from types import SimpleNamespace

import numpy as np

from faceattend.vision.face_detector import InsightFaceDetector
from faceattend.vision.types import Frame


def _frame() -> Frame:
    return Frame(np.zeros((480, 640, 3), dtype=np.uint8), 1, 1)


def test_detector_loads_model_once_and_maps_face_fields() -> None:
    loads: list[int] = []

    def load_model() -> SimpleNamespace:
        loads.append(1)
        return SimpleNamespace(
            detect=lambda _pixels, **_kwargs: (
                np.array([[100.0, 80.0, 300.0, 280.0, 0.97]]),
                np.array([[[120.0, 140.0], [280.0, 140.0]]]),
            )
        )

    detector = InsightFaceDetector(model_loader=load_model, model_checksum="abc")

    first = detector.detect(_frame())
    second = detector.detect(_frame())

    assert loads == [1]
    assert len(first) == len(second) == 1
    assert first[0].bbox.x_min == 100.0
    assert first[0].detector_score == 0.97
    assert first[0].landmarks.shape == (2, 2)
    assert first[0].quality.face_area_ratio == 40000 / (640 * 480)
    assert detector.model_metadata.checksum == "abc"


def test_detector_accepts_missing_landmarks_for_later_quality_rejection() -> None:
    detector = InsightFaceDetector(
        model_loader=lambda: SimpleNamespace(
            detect=lambda _pixels, **_kwargs: (
                np.array([[10.0, 20.0, 100.0, 120.0, 0.8]]),
                None,
            )
        )
    )

    result = detector.detect(_frame())[0]

    assert result.landmarks.shape == (0, 2)
    assert result.detector_score == 0.8
