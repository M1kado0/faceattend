"""Tests for MediaPipe active liveness helpers."""

import pytest

from ml.liveness.mediapipe_active import (
    ActiveLivenessConfig,
    BlinkCounter,
    MediaPipeActiveLivenessChecker,
    VideoDecodeError,
    _decode_video,
)


def test_decode_video_empty_bytes_raises_video_decode_error() -> None:
    with pytest.raises(VideoDecodeError, match="empty_video"):
        _decode_video(b"")


def test_blink_counter_counts_closed_to_open_transitions() -> None:
    counter = BlinkCounter(closed_eye_threshold=0.20, open_eye_threshold=0.24)

    for eye_aspect_ratio in [0.30, 0.18, 0.17, 0.25, 0.28, 0.19, 0.26]:
        counter.observe(eye_aspect_ratio)

    assert counter.blinks == 2


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
