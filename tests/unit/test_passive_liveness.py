"""Tests for the headless MiniFASNet passive-liveness adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from faceattend.vision.passive_liveness import (
    MiniFASNetPassiveLivenessDetector,
    PassiveLivenessError,
    TemporalPassiveLivenessSession,
)
from faceattend.vision.types import (
    BoundingBox,
    EvidenceDecision,
    FaceObservation,
    FaceQuality,
    Frame,
)


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


def test_temporal_passive_liveness_reports_window_statistics() -> None:
    detector = _detector([0.91, 0.70, 0.88])

    evidence = detector.evaluate_temporal(
        [_frame(1), _frame(2), _frame(3)], [_face(), _face(), _face()]
    )

    assert evidence.minimum_score == 0.70
    assert evidence.median_score == 0.88
    assert evidence.suspicious_frame_count == 1
    assert evidence.failure_to_process_count == 0
    assert evidence.decision.value == "failed"


def test_temporal_passive_liveness_counts_inference_failures() -> None:
    iterator = iter([0.91, ValueError("bad frame"), 0.90])

    def load() -> SimpleNamespace:
        def predict(_image, _bbox):
            value = next(iterator)
            if isinstance(value, Exception):
                raise value
            return {"score": value}

        return SimpleNamespace(predict=predict)

    detector = MiniFASNetPassiveLivenessDetector(
        model_path="unused", model_loader=load, model_checksum="abc"
    )
    evidence = detector.evaluate_temporal(
        [_frame(1), _frame(2), _frame(3)], [_face(), _face(), _face()]
    )

    assert evidence.failure_to_process_count == 1
    assert evidence.decision.value == "failed"


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


def test_temporal_passive_session_keeps_bounded_window_and_finalizes_once() -> None:
    detector = _detector([0.91, 0.88])
    session = TemporalPassiveLivenessSession(
        detector, max_frames=2, min_frames=2, min_duration_ms=0, sample_interval_ms=0
    )

    session.observe(_frame(1), _face())
    session.observe(_frame(2), _face())
    evidence = session.finalize()

    assert evidence.score == 0.88
    assert session.finalize() is evidence


def test_temporal_passive_session_rejects_stale_frames() -> None:
    session = TemporalPassiveLivenessSession(_detector([0.9]), max_frames=2)
    session.observe(_frame(2), _face())

    with pytest.raises(ValueError, match="strictly increasing"):
        session.observe(_frame(1), _face())


def test_temporal_session_samples_without_evicting_early_evidence() -> None:
    detector = _detector([0.91, 0.92, 0.93])
    session = TemporalPassiveLivenessSession(
        detector, min_frames=3, min_duration_ms=200, sample_interval_ms=100, max_frames=3
    )
    for timestamp in (0, 50_000_000, 100_000_000, 200_000_000, 300_000_000):
        session.observe(_frame(timestamp), _face())
    assert [frame.captured_at_ns for frame in session.frames] == [0, 100_000_000, 200_000_000]
    assert session.finalize().decision is EvidenceDecision.PASSED


def test_temporal_session_does_not_evict_an_early_suspicious_sample() -> None:
    detector = _detector([0.70, 0.92, 0.93])
    session = TemporalPassiveLivenessSession(
        detector, min_frames=3, min_duration_ms=200, sample_interval_ms=100, max_frames=3
    )
    for timestamp in (0, 100_000_000, 200_000_000, 300_000_000):
        session.observe(_frame(timestamp), _face())
    result = session.finalize()
    assert result.decision is EvidenceDecision.FAILED
    assert result.minimum_score == 0.70


@pytest.mark.parametrize("timestamps", [(0, 100_000_000), (0, 50_000_000, 100_000_000)])
def test_temporal_session_requires_count_and_duration(timestamps: tuple[int, ...]) -> None:
    detector = _detector([0.9] * len(timestamps))
    session = TemporalPassiveLivenessSession(
        detector, min_frames=3, min_duration_ms=200, sample_interval_ms=1
    )
    for timestamp in timestamps:
        session.observe(_frame(timestamp), _face())
    result = session.finalize()
    assert result.decision is EvidenceDecision.FAILED
    assert result.reason == "insufficient_passive_evidence"


def test_temporal_session_cancel_is_terminal_and_clears_frames() -> None:
    detector = _detector([0.9])
    session = TemporalPassiveLivenessSession(detector, min_frames=1, min_duration_ms=0)
    session.observe(_frame(1), _face())
    result = session.cancel()
    assert result.reason == "cancelled" and session.frames == ()
    assert session.finalize() is result
