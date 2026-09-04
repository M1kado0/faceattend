"""MediaPipe-backed active liveness checks."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Allow `uv run ml/liveness/mediapipe_active.py` from the repo root while keeping
# package imports for normal app/test execution.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.liveness.base import LivenessResult
from ml.liveness.challenge import ActiveLivenessChallenge
from ml.liveness.head_pose import estimate_head_pose

RIGHT_EYE = (33, 159, 158, 133, 153, 145)
LEFT_EYE = (362, 380, 374, 263, 386, 385)


@dataclass(frozen=True)
class ActiveLivenessConfig:
    """Tuning knobs for blink-based active liveness."""

    min_blinks: int = 2
    min_seconds: float = 1.0
    max_seconds: float = 15.0
    min_face_frame_ratio: float = 0.65
    closed_eye_threshold: float = 0.08
    open_eye_threshold: float = 0.14
    max_faces: int = 1
    min_closed_frames: int = 2
    min_open_frames: int = 2
    blink_cooldown_ms: int = 250
    max_multiple_face_frame_ratio: float = 0.05
    head_turn_yaw_threshold: float = 10.0
    min_head_turn_frames: int = 3


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
        last_timestamp_ms = -1
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            timestamp_ms = _frame_timestamp_ms(
                capture,
                frame_index=frame_index,
                fps=fps,
                last_timestamp_ms=last_timestamp_ms,
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frames.append(VideoFrame(image_rgb=frame_rgb, timestamp_ms=timestamp_ms))
            last_timestamp_ms = timestamp_ms
            frame_index += 1

        capture.release()

    if not frames:
        raise VideoDecodeError("no_decodable_frames")

    return frames, len(frames) / fps


def _frame_timestamp_ms(
    capture: cv2.VideoCapture,
    *,
    frame_index: int,
    fps: float,
    last_timestamp_ms: int,
) -> int:
    metadata_timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
    fallback_timestamp_ms = int(round((frame_index / fps) * 1000))
    timestamp_ms = metadata_timestamp_ms if metadata_timestamp_ms > 0 else fallback_timestamp_ms

    if timestamp_ms <= last_timestamp_ms:
        return last_timestamp_ms + 1
    return timestamp_ms


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
    def __init__(
        self,
        closed_eye_threshold: float,
        open_eye_threshold: float,
        min_closed_frames: int,
        min_open_frames: int,
        blink_cooldown_ms: int,
    ) -> None:
        self.closed_eye_threshold = closed_eye_threshold
        self.open_eye_threshold = open_eye_threshold
        self.min_closed_frames = min_closed_frames
        self.min_open_frames = min_open_frames
        self.blink_cooldown_ms = blink_cooldown_ms
        self.blinks = 0
        self.state = "OPEN"
        self.closed_frame_count = 0
        self.open_frame_count = 0
        self.last_blink_timestamp_ms: int | None = None

    def observe(self, eye_aspect_ratio: float, timestamp_ms: int) -> None:
        if self.state in {"OPEN", "CLOSED_CANDIDATE"}:
            if eye_aspect_ratio <= self.closed_eye_threshold:
                self.closed_frame_count += 1
                self.state = "CLOSED_CANDIDATE"

                if self.closed_frame_count >= self.min_closed_frames:
                    self.state = "CLOSED"
                    self.open_frame_count = 0
                return

            self.closed_frame_count = 0
            self.state = "OPEN"
            return

        if self.state in {"CLOSED", "OPEN_CANDIDATE"}:
            if eye_aspect_ratio >= self.open_eye_threshold:
                self.open_frame_count += 1
                self.state = "OPEN_CANDIDATE"

                if self.open_frame_count >= self.min_open_frames:
                    self._count_blink_if_allowed(timestamp_ms)
                    self._reset_after_blink()
                return

            self.open_frame_count = 0
            self.state = "CLOSED"

    def _count_blink_if_allowed(self, timestamp_ms: int) -> None:
        if (
            self.last_blink_timestamp_ms is None
            or timestamp_ms - self.last_blink_timestamp_ms >= self.blink_cooldown_ms
        ):
            self.blinks += 1
            self.last_blink_timestamp_ms = timestamp_ms

    def _reset_after_blink(self) -> None:
        self.state = "OPEN"
        self.closed_frame_count = 0
        self.open_frame_count = 0


class BlinkTurnLeftRightChallenge:
    def __init__(self, config: ActiveLivenessConfig) -> None:
        self.config = config
        self.blink_counter = BlinkCounter(
            config.closed_eye_threshold,
            config.open_eye_threshold,
            config.min_closed_frames,
            config.min_open_frames,
            config.blink_cooldown_ms,
        )
        self.stage = "BLINK"
        self.left_frame_count = 0
        self.right_frame_count = 0

    def observe(
        self,
        *,
        eye_aspect_ratio: float,
        yaw: float | None,
        timestamp_ms: int,
    ) -> None:
        if self.stage == "BLINK":
            self.blink_counter.observe(eye_aspect_ratio, timestamp_ms)
            if self.blink_counter.blinks >= self.config.min_blinks:
                self.stage = "TURN_LEFT"
            return

        if self.stage == "TURN_LEFT":
            if yaw is not None and yaw <= -self.config.head_turn_yaw_threshold:
                self.left_frame_count += 1
                if self.left_frame_count >= self.config.min_head_turn_frames:
                    self.stage = "TURN_RIGHT"
                return
            self.left_frame_count = 0
            return

        if self.stage == "TURN_RIGHT":
            if yaw is not None and yaw >= self.config.head_turn_yaw_threshold:
                self.right_frame_count += 1
                if self.right_frame_count >= self.config.min_head_turn_frames:
                    self.stage = "DONE"
                return
            self.right_frame_count = 0

    @property
    def completed(self) -> bool:
        return self.stage == "DONE"

    @property
    def reason(self) -> str:
        if self.stage == "BLINK":
            return "blink_not_completed"
        if self.stage == "TURN_LEFT":
            return "left_turn_not_completed"
        if self.stage == "TURN_RIGHT":
            return "right_turn_not_completed"
        return "challenge_not_completed"

    @property
    def score(self) -> float:
        blink_score = min(self.blink_counter.blinks / self.config.min_blinks, 1.0)
        left_score = min(self.left_frame_count / self.config.min_head_turn_frames, 1.0)
        right_score = min(self.right_frame_count / self.config.min_head_turn_frames, 1.0)
        return (blink_score + left_score + right_score) / 3


def _average_eye_aspect_ratio(face_landmarks: list) -> float:
    left = _eye_aspect_ratio(face_landmarks, LEFT_EYE)
    right = _eye_aspect_ratio(face_landmarks, RIGHT_EYE)
    return (left + right) / 2


def _eye_aspect_ratio(
    face_landmarks: list, eye_indices: tuple[int, int, int, int, int, int]
) -> float:
    p1, p2, p3, p4, p5, p6 = [_landmark_xy(face_landmarks[i]) for i in eye_indices]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


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
        self._landmarker_lock = Lock()
        self._next_video_timestamp_offset_ms = 0

    def check(
        self,
        video: bytes,
        challenge: ActiveLivenessChallenge | str = ActiveLivenessChallenge.BLINK_TWICE,
    ) -> LivenessResult:
        try:
            challenge = ActiveLivenessChallenge(challenge)
        except ValueError as exc:
            return _failed(
                "unsupported_challenge",
                score=0.0,
                reason=str(exc),
                challenge_completed=False,
            )
        try:
            frames, duration_seconds = _decode_video(video)
        except VideoDecodeError as exc:
            return _failed(
                "video_decode_failed",
                score=0.0,
                reason=str(exc),
                challenge_completed=False,
            )
        duration_result = self._validate_duration(duration_seconds)
        if duration_result is not None:
            return duration_result

        try:
            if challenge == ActiveLivenessChallenge.BLINK_TWICE:
                blink_result = self._check_blink_twice(frames)
                passed = blink_result.challenge_completed
                return LivenessResult(
                    passed=passed,
                    score=blink_result.score,
                    label=challenge.value,
                    reason=None if passed else blink_result.reason,
                    challenge_completed=blink_result.challenge_completed,
                )
            if challenge == ActiveLivenessChallenge.BLINK_TURN_LEFT_RIGHT:
                return self._check_blink_turn_left_right(frames)
        except FileNotFoundError as exc:
            return _failed(
                "model_not_found",
                score=0.0,
                reason=str(exc),
                challenge_completed=False,
            )
        except ValueError as exc:
            return _failed(
                "landmarker_failed",
                score=0.0,
                reason=str(exc),
                challenge_completed=False,
            )

        return _failed(
            "unsupported_challenge",
            score=0.0,
            challenge_completed=False,
            reason=f"challenge={challenge.value}",
        )

    def _validate_duration(self, duration_seconds: float) -> LivenessResult | None:
        if duration_seconds < self.config.min_seconds:
            return _failed(
                "video_too_short",
                score=0.0,
                reason=f"duration_seconds={duration_seconds:.2f}",
                challenge_completed=False,
            )
        if duration_seconds > self.config.max_seconds:
            return _failed(
                "video_too_long",
                score=0.0,
                reason=f"duration_seconds={duration_seconds:.2f}",
                challenge_completed=False,
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
            return _failed(
                "no_video_frames",
                score=0.0,
                challenge_completed=False,
            )

        with self._landmarker_lock:
            timestamp_offset_ms = self._timestamp_offset_for_next_video(frames)
            return self._check_blink_twice_with_offset(frames, timestamp_offset_ms)

    def _check_blink_turn_left_right(self, frames: list[VideoFrame]) -> LivenessResult:
        if not frames:
            return _failed(
                "no_video_frames",
                score=0.0,
                challenge_completed=False,
            )

        with self._landmarker_lock:
            timestamp_offset_ms = self._timestamp_offset_for_next_video(frames)
            return self._check_blink_turn_left_right_with_offset(frames, timestamp_offset_ms)

    def _timestamp_offset_for_next_video(self, frames: list[VideoFrame]) -> int:
        timestamp_offset_ms = self._next_video_timestamp_offset_ms
        self._next_video_timestamp_offset_ms += frames[-1].timestamp_ms + 1
        return timestamp_offset_ms

    def _check_blink_twice_with_offset(
        self,
        frames: list[VideoFrame],
        timestamp_offset_ms: int,
    ) -> LivenessResult:
        blink_counter = BlinkCounter(
            self.config.closed_eye_threshold,
            self.config.open_eye_threshold,
            self.config.min_closed_frames,
            self.config.min_open_frames,
            self.config.blink_cooldown_ms,
        )

        face_frames = 0
        multiple_face_frames = 0
        ear_values: list[float] = []
        for frame in frames:
            timestamp_ms = frame.timestamp_ms + timestamp_offset_ms
            result = self._landmarker_instance().detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.image_rgb),
                timestamp_ms,
            )
            face_count = len(result.face_landmarks)
            if face_count == 0:
                continue
            if face_count > 1:
                multiple_face_frames += 1
                continue

            face_frames += 1
            ear = _average_eye_aspect_ratio(result.face_landmarks[0])
            ear_values.append(ear)
            blink_counter.observe(ear, timestamp_ms)

        multiple_face_ratio = multiple_face_frames / len(frames)
        if multiple_face_ratio > self.config.max_multiple_face_frame_ratio:
            return _failed(
                "multiple_faces_detected",
                score=0.0,
                challenge_completed=False,
            )

        face_frame_ratio = face_frames / len(frames)
        if face_frame_ratio < self.config.min_face_frame_ratio:
            return _failed(
                "face_not_visible_enough",
                score=0.0,
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

    def _check_blink_turn_left_right_with_offset(
        self,
        frames: list[VideoFrame],
        timestamp_offset_ms: int,
    ) -> LivenessResult:
        challenge = BlinkTurnLeftRightChallenge(self.config)
        face_frames = 0
        multiple_face_frames = 0

        for frame in frames:
            timestamp_ms = frame.timestamp_ms + timestamp_offset_ms
            result = self._landmarker_instance().detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.image_rgb),
                timestamp_ms,
            )
            face_count = len(result.face_landmarks)
            if face_count == 0:
                continue
            if face_count > 1:
                multiple_face_frames += 1
                continue

            face_frames += 1
            landmarks = result.face_landmarks[0]
            eye_aspect_ratio = _average_eye_aspect_ratio(landmarks)
            image_height, image_width = frame.image_rgb.shape[:2]
            head_pose = estimate_head_pose(
                landmarks,
                image_width=image_width,
                image_height=image_height,
            )
            if head_pose is None:
                print(
                    "[head-pose-debug]",
                    f"timestamp_ms={timestamp_ms}",
                    f"stage={challenge.stage}",
                    f"ear={eye_aspect_ratio:.3f}",
                    "pose=None",
                )
            else:
                print(
                    "[head-pose-debug]",
                    f"timestamp_ms={timestamp_ms}",
                    f"stage={challenge.stage}",
                    f"ear={eye_aspect_ratio:.3f}",
                    f"yaw={head_pose.yaw:.2f}",
                    f"pitch={head_pose.pitch:.2f}",
                    f"roll={head_pose.roll:.2f}",
                )
            challenge.observe(
                eye_aspect_ratio=eye_aspect_ratio,
                yaw=head_pose.yaw if head_pose else None,
                timestamp_ms=timestamp_ms,
            )

        multiple_face_ratio = multiple_face_frames / len(frames)
        if multiple_face_ratio > self.config.max_multiple_face_frame_ratio:
            return _failed(
                "multiple_faces_detected",
                score=0.0,
                challenge_completed=False,
            )

        face_frame_ratio = face_frames / len(frames)
        if face_frame_ratio < self.config.min_face_frame_ratio:
            return _failed(
                "face_not_visible_enough",
                score=0.0,
                challenge_completed=False,
            )

        return LivenessResult(
            passed=challenge.completed,
            score=challenge.score,
            label=ActiveLivenessChallenge.BLINK_TURN_LEFT_RIGHT.value,
            reason=None if challenge.completed else challenge.reason,
            challenge_completed=challenge.completed,
        )
