"""MediaPipe-backed active liveness checks."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from ml.liveness.base import LivenessResult
from ml.liveness.challenge import ActiveLivenessChallenge

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


@dataclass(frozen=True)
class ActiveLivenessConfig:
    """Tuning knobs for blink-based active liveness."""

    min_blinks: int = 2
    min_seconds: float = 2.0
    max_seconds: float = 6.0
    min_face_frame_ratio: float = 0.65
    closed_eye_threshold: float = 0.20
    open_eye_threshold: float = 0.24
    max_faces: int = 1


@dataclass(frozen=True)
class VideoFrame:
    image_rgb: np.ndarray
    timestamp_ms: int


class VideoDecodeError(ValueError):
    """Raised when a liveness video cannot be decoded."""


def _decode_video(video: bytes) -> tuple[list[VideoFrame], float]:
    if not video:
        raise VideoDecodeError("empty_video")

    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        tmp.write(video)
        tmp.flush()

        capture = cv2.VideoCapture(tmp.name)
        if not capture.isOpened():
            raise VideoDecodeError("could_not_open_video")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frames: list[VideoFrame] = []
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            timestamp_ms = int((frame_index / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frames.append(VideoFrame(image_rgb=frame_rgb, timestamp_ms=timestamp_ms))
            frame_index += 1

        capture.release()

    if not frames:
        raise VideoDecodeError("no_decodable_frames")

    return frames, len(frames) / fps


def _failed(
    label: str,
    *,
    score: float,
    challenge_completed: bool,
    reason: str | None = None,
) -> LivenessResult:
    return LivenessResult(
        passed=False,
        score=score,
        label=label,
        reason=reason or label,
        challenge_completed=challenge_completed,
    )


class BlinkCounter:
    def __init__(self, closed_eye_threshold: float, open_eye_threshold: float) -> None:
        self.closed_eye_threshold = closed_eye_threshold
        self.open_eye_threshold = open_eye_threshold
        self.blinks = 0
        self._eye_was_closed = False

    def observe(self, eye_aspect_ratio: float) -> None:
        if eye_aspect_ratio <= self.closed_eye_threshold:
            self._eye_was_closed = True
            return
        if self._eye_was_closed and eye_aspect_ratio >= self.open_eye_threshold:
            self.blinks += 1
            self._eye_was_closed = False


def _average_eye_aspect_ratio(face_landmarks: list) -> float:
    left = _eye_aspect_ratio(face_landmarks, LEFT_EYE)
    right = _eye_aspect_ratio(face_landmarks, RIGHT_EYE)
    return (left + right) / 2.0


def _eye_aspect_ratio(
    face_landmarks: list, eye_indices: tuple[int, int, int, int, int, int]
) -> float:
    p1, p2, p3, p4, p5, p6 = [_landmark_xy(face_landmarks[index]) for index in eye_indices]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


def _landmark_xy(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float32)


class MediaPipeActiveLivenessChecker:
    def __init__(
        self,
        model_path: str,
        config: ActiveLivenessConfig | None = None,
    ):
        self.model_path = Path(model_path)
        self.config = config or ActiveLivenessConfig()
        self._landmarker: FaceLandmarker | None = None

    def check(
        self,
        video: bytes,
        challenge: ActiveLivenessChallenge | str = ActiveLivenessChallenge.BLINK_TWICE,
    ) -> LivenessResult:
        try:
            challenge = ActiveLivenessChallenge(challenge)
        except ValueError as exc:
            return _failed(
                "unsupported_challenge", score=0.0, challenge_completed=False, reason=str(exc)
            )

        try:
            frames, duration_seconds = _decode_video(video)
        except VideoDecodeError as exc:
            return _failed(
                "video_decode_failed", score=0.0, challenge_completed=False, reason=str(exc)
            )

        duration_result = self._validate_duration(duration_seconds)
        if duration_result is not None:
            return duration_result

        blink_result = self._check_blink_twice(frames)
        passed = blink_result.challenge_completed
        return LivenessResult(
            passed=passed,
            score=blink_result.score,
            label=ActiveLivenessChallenge.BLINK_TWICE.value,
            reason=None if passed else blink_result.reason,
            challenge_completed=blink_result.challenge_completed,
        )

    def _validate_duration(self, duration_seconds: float) -> LivenessResult | None:
        if duration_seconds < self.config.min_seconds:
            return _failed(
                "video_too_short",
                score=0.0,
                challenge_completed=False,
                reason=f"duration_seconds={duration_seconds:.2f}",
            )
        if duration_seconds > self.config.max_seconds:
            return _failed(
                "video_too_long",
                score=0.0,
                challenge_completed=False,
                reason=f"duration_seconds={duration_seconds:.2f}",
            )
        return None

    def _landmarker_instance(self) -> FaceLandmarker:
        if self._landmarker is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe face landmarker model not found: {self.model_path}"
                )

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=self.config.max_faces + 1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
        return self._landmarker

    def _check_blink_twice(self, frames: list[VideoFrame]) -> LivenessResult:
        if not frames:
            return _failed("no_video_frames", score=0.0, challenge_completed=False)
        blink_counter = BlinkCounter(
            self.config.closed_eye_threshold, self.config.open_eye_threshold
        )
        face_frames = 0
        multiple_face_frames = 0
        for frame in frames:
            result = self._landmarker_instance().detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.image_rgb), frame.timestamp_ms
            )
            face_count = len(result.face_landmarks)
            if face_count == 0:
                continue
            if face_count > self.config.max_faces:
                multiple_face_frames += 1
                continue

            face_frames += 1
            ear = _average_eye_aspect_ratio(result.face_landmarks[0])
            blink_counter.observe(ear)

        if multiple_face_frames:
            return _failed("multiple_faces_detected", score=0.0, challenge_completed=False)

        face_frame_ratio = face_frames / len(frames)
        if face_frame_ratio < self.config.min_face_frame_ratio:
            return _failed(
                "face_not_visible_enough",
                score=face_frame_ratio,
                challenge_completed=False,
            )
        blink_score = min(blink_counter.blinks / self.config.min_blinks, 1.0)
        challenge_completed = blink_counter.blinks >= self.config.min_blinks
        return LivenessResult(
            passed=challenge_completed,
            score=blink_score,
            label=ActiveLivenessChallenge.BLINK_TWICE.value,
            reason=None if challenge_completed else "challenge_not_completed",
            challenge_completed=challenge_completed,
        )
