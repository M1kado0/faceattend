"""Tests for MediaPipe active liveness helpers."""

import pytest

import ml.liveness.mediapipe_active as mediapipe_active
from ml.liveness.mediapipe_active import (
    ActiveLivenessConfig,
    BlinkCounter,
    MediaPipeActiveLivenessChecker,
    VideoDecodeError,
    VideoFrame,
    _decode_video,
    _frame_timestamp_ms,
)


def test_decode_video_empty_bytes_raises_video_decode_error() -> None:
    with pytest.raises(VideoDecodeError, match="empty_video"):
        _decode_video(b"")


def _blink_counter(*, cooldown_ms: int = 250) -> BlinkCounter:
    return BlinkCounter(
        closed_eye_threshold=0.20,
        open_eye_threshold=0.24,
        min_closed_frames=2,
        min_open_frames=2,
        blink_cooldown_ms=cooldown_ms,
    )


def _observe_sequence(counter: BlinkCounter, values: list[float]) -> None:
    for index, eye_aspect_ratio in enumerate(values):
        counter.observe(eye_aspect_ratio, timestamp_ms=index * 100)


class _FakeCapture:
    def __init__(self, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms

    def get(self, _property_id: int) -> float:
        return self.timestamp_ms


def test_frame_timestamp_ms_clamps_duplicate_metadata_timestamps() -> None:
    timestamp_ms = _frame_timestamp_ms(
        _FakeCapture(timestamp_ms=100.0),
        frame_index=4,
        fps=30.0,
        last_timestamp_ms=100,
    )

    assert timestamp_ms == 101


def test_blink_counter_counts_stable_closed_to_open_transitions() -> None:
    counter = _blink_counter()

    _observe_sequence(
        counter,
        [0.30, 0.18, 0.17, 0.25, 0.26, 0.30, 0.19, 0.18, 0.25, 0.26],
    )

    assert counter.blinks == 2


def test_blink_counter_ignores_non_consecutive_closed_noise() -> None:
    counter = _blink_counter()

    _observe_sequence(counter, [0.30, 0.18, 0.30, 0.17, 0.25, 0.26])

    assert counter.blinks == 0


def test_blink_counter_ignores_non_consecutive_open_noise() -> None:
    counter = _blink_counter()

    _observe_sequence(counter, [0.30, 0.18, 0.17, 0.25, 0.18, 0.25])

    assert counter.blinks == 0


def test_blink_counter_respects_cooldown() -> None:
    counter = _blink_counter(cooldown_ms=500)

    _observe_sequence(counter, [0.18, 0.17, 0.25, 0.26, 0.18, 0.17, 0.25, 0.26])

    assert counter.blinks == 1


def test_check_accepts_string_challenge_before_model_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=0.0, max_seconds=6.0),
    )
    monkeypatch.setattr(
        mediapipe_active,
        "_decode_video",
        lambda _video: ([VideoFrame(image_rgb=object(), timestamp_ms=0)], 1.0),
    )

    result = checker.check(b"fake-video", challenge="blink_twice")

    assert result.passed is False
    assert result.label == "model_not_found"


def test_check_rejects_unknown_string_challenge_before_decoding() -> None:
    checker = MediaPipeActiveLivenessChecker(model_path="missing.task")

    result = checker.check(b"", challenge="turn_around")

    assert result.passed is False
    assert result.label == "unsupported_challenge"


def test_validate_duration_rejects_too_short_video() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    result = checker._validate_duration(1.5)

    assert result is not None
    assert result.passed is False
    assert result.label == "video_too_short"
    assert result.challenge_completed is False


def test_validate_duration_rejects_too_long_video() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    result = checker._validate_duration(6.5)

    assert result is not None
    assert result.passed is False
    assert result.label == "video_too_long"
    assert result.challenge_completed is False


def test_validate_duration_accepts_allowed_video_duration() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    assert checker._validate_duration(3.0) is None


def test_timestamp_offset_for_next_video_advances_between_requests() -> None:
    checker = MediaPipeActiveLivenessChecker(model_path="missing.task")
    frames = [
        VideoFrame(image_rgb=object(), timestamp_ms=0),
        VideoFrame(image_rgb=object(), timestamp_ms=100),
    ]

    first_offset = checker._timestamp_offset_for_next_video(frames)
    second_offset = checker._timestamp_offset_for_next_video(frames)

    assert first_offset == 0
    assert second_offset == 101
