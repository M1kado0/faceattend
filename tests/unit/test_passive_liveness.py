"""Tests for the headless MiniFASNet passive-liveness adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from faceattend.vision.passive_liveness import (
    MiniFASNetPassiveLivenessDetector,
    PassiveLivenessError,
)
from faceattend.vision.types import BoundingBox, FaceObservation, FaceQuality, Frame


def _frame(timestamp: int) -> Frame:
    return Frame(np.zeros((120, 160, 3), dtype=np.uint8), timestamp, timestamp)


def _face() -> FaceObservation:
    return FaceObservation(
        bbox=BoundingBox(20.0, 10.0, 100.0, 110.0),
        detector_score=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        quality=FaceQuality(True, 1.0, 0.5, 0.2),
    )


def _detector(scores: list[float]) -> MiniFASNetPassiveLivenessDetector:
    iterator = iter(scores)

    def load() -> SimpleNamespace:
        return SimpleNamespace(predict=lambda _image, _bbox: {"score": next(iterator)})

    return MiniFASNetPassiveLivenessDetector(
        model_path="unused.onnx",
        model_loader=load,
        model_checksum="abc",
    )


def test_passive_liveness_passes_only_when_all_window_frames_pass() -> None:
    detector = _detector([0.91, 0.88])

    evidence = detector.evaluate([_frame(1), _frame(2)], [_face(), _face()])

    assert evidence.decision.value == "passed"
    assert evidence.score == 0.88
    assert evidence.threshold == 0.85
    assert evidence.model_version == "MiniFASNetV2"


def test_passive_liveness_fails_on_lowest_frame_score() -> None:
    detector = _detector([0.99, 0.70])

    evidence = detector.evaluate([_frame(1), _frame(2)], [_face(), _face()])

    assert evidence.decision.value == "failed"
    assert evidence.score == 0.70
    assert evidence.reason == "liveness_score_below_threshold"


def test_passive_liveness_loads_model_once() -> None:
    loads: list[int] = []

    def load() -> SimpleNamespace:
        loads.append(1)
        return SimpleNamespace(predict=lambda _image, _bbox: {"score": 0.9})

    detector = MiniFASNetPassiveLivenessDetector(model_path="unused", model_loader=load)
    detector.evaluate([_frame(1)], [_face()])
    detector.evaluate([_frame(2)], [_face()])

    assert loads == [1]


def test_passive_liveness_rejects_invalid_window() -> None:
    detector = _detector([0.9])

    with pytest.raises(PassiveLivenessError, match="required"):
        detector.evaluate([], [])
    with pytest.raises(PassiveLivenessError, match="per frame"):
        detector.evaluate([_frame(1), _frame(2)], [_face()])


def test_passive_liveness_rejects_invalid_model_score() -> None:
    detector = _detector([1.5])

    with pytest.raises(PassiveLivenessError, match="between 0 and 1"):
        detector.evaluate([_frame(1)], [_face()])
