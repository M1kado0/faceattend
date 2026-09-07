"""MediaPipe-backed active liveness checks."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Allow `uv run ml/liveness/mediapipe_active.py` from the repo root while keeping
# package imports for normal app/test execution.
if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repository_root))
    sys.path.append(str(repository_root / "src"))

from faceattend.vision.active_liveness import ActiveLivenessChallengeEvaluator, ChallengeStatus
from faceattend.vision.challenge_session import ChallengeAction
from faceattend.vision.evidence import DuplicateFrameDetector, EvidenceStreamGuard
from faceattend.vision.head_pose import (
    HeadPoseError,
    MediaPipeMatrixHeadPoseEstimator,
    compare_pose_estimators,
)
from faceattend.vision.passive_liveness import TemporalPassiveLivenessSession
from faceattend.vision.quality import measure_quality
from faceattend.vision.types import (
    EvidenceDecision,
    FaceObservation,
    Frame,
    FrameEvidence,
    HeadPose,
    LivenessEvidence,
)
from ml.liveness.base import LivenessResult
from ml.liveness.challenge import ActiveLivenessChallenge

RIGHT_EYE = (33, 159, 158, 133, 153, 145)
LEFT_EYE = (362, 380, 374, 263, 386, 385)
_MATRIX_POSE_ESTIMATOR = MediaPipeMatrixHeadPoseEstimator()


def _challenge_yaw(raw_yaw_degrees: float, *, invert: bool) -> float:
    """Map raw camera yaw to the physical direction used by challenge text."""
    return -raw_yaw_degrees if invert else raw_yaw_degrees


def _matrix_pose_from_result(result) -> HeadPose:
    """Return one valid MediaPipe pose or raise a fail-closed pose error."""
    matrices = getattr(result, "facial_transformation_matrixes", ())
    if len(matrices) != 1:
        raise HeadPoseError("facial_transformation_matrix_missing")
    return _MATRIX_POSE_ESTIMATOR.estimate(matrices[0])


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
    smile_score_threshold: float = 0.5
    min_smile_frames: int = 2
    max_multiple_face_frame_ratio: float = 0.05
    head_turn_yaw_threshold: float = 10.0
    min_head_turn_frames: int = 3
    # Camera orientation convention: measured physical left was positive raw
    # yaw, so invert at the user-facing challenge boundary by default.
    invert_yaw_for_challenge: bool = True


@dataclass(frozen=True)
class VideoFrame:
    image_rgb: np.ndarray
    timestamp_ms: int


class VideoDecodeError(ValueError):
    """Raised when a liveness video cannot be decoded."""


def mediapipe_smile_score(result) -> float:
    """Return the mean MediaPipe smile blendshape score for one face."""
    blendshape_faces = getattr(result, "face_blendshapes", ())
    if len(blendshape_faces) != 1:
        return 0.0
    scores = {
        getattr(category, "category_name", ""): float(getattr(category, "score", 0.0))
        for category in blendshape_faces[0]
    }
    available = [scores[name] for name in ("mouthSmileLeft", "mouthSmileRight") if name in scores]
    return sum(available) / len(available) if available else 0.0


class LivenessRuntimePhase(StrEnum):
    """User-visible stages of the combined active and passive session."""

    ACTIVE_CHALLENGE = "active_challenge"
    FACE_CAMERA = "face_camera"
    HOLD_STILL = "hold_still"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class NeutralPadConfig:
    """Requirements for the dedicated post-active neutral PAD window."""

    neutral_dwell_ms: int = 1_000
    timeout_ms: int = 8_000
    max_smile_score: float = 0.3

    def __post_init__(self) -> None:
        if self.neutral_dwell_ms < 0:
            raise ValueError("neutral_dwell_ms must be non-negative")
        if self.timeout_ms <= self.neutral_dwell_ms:
            raise ValueError("timeout_ms must be greater than neutral_dwell_ms")
        if not 0.0 <= self.max_smile_score <= 1.0:
            raise ValueError("max_smile_score must be between 0 and 1")


class MediaPipeActionEvidence:
    """Convert real Face Landmarker landmarks/blendshapes into actions."""

    def __init__(self, config: ActiveLivenessConfig | None = None) -> None:
        self.config = config or ActiveLivenessConfig()
        self.reset()

    def reset(self) -> None:
        """Discard action history at attempt/challenge boundaries."""
        self.blink_counter = BlinkCounter(
            self.config.closed_eye_threshold,
            self.config.open_eye_threshold,
            self.config.min_closed_frames,
            self.config.min_open_frames,
            self.config.blink_cooldown_ms,
        )
        self._smile_frames = 0
        self._smile_armed = False

    def observe(self, result, timestamp_ms: int) -> ChallengeAction | None:
        """Return at most one action event for the current MediaPipe result."""
        landmarks = getattr(result, "face_landmarks", ())
        if len(landmarks) != 1:
            self._smile_frames = 0
            return None

        before_blinks = self.blink_counter.blinks
        ear = _average_eye_aspect_ratio(landmarks[0])
        if not np.isfinite(ear):
            raise ValueError("invalid_eye_evidence")
        for face_scores in getattr(result, "face_blendshapes", ()):
            if any(not 0 <= float(category.score) <= 1 for category in face_scores):
                raise ValueError("invalid_blendshape_evidence")
        self.blink_counter.observe(ear, timestamp_ms)
        if self.blink_counter.blinks > before_blinks:
            self._smile_frames = 0
            return ChallengeAction.BLINK

        smile = mediapipe_smile_score(result)
        if smile < self.config.smile_score_threshold:
            self._smile_armed = True
            self._smile_frames = 0
        elif self._smile_armed:
            self._smile_frames += 1
        else:
            self._smile_frames = 0
        if self._smile_frames >= self.config.min_smile_frames:
            self._smile_armed = False
            self._smile_frames = 0
            return ChallengeAction.SMILE
        return None


class MediaPipeChallengeSessionAdapter:
    """Bridge MediaPipe action evidence into the headless challenge evaluator."""

    def __init__(
        self,
        evaluator: ActiveLivenessChallengeEvaluator,
        config: ActiveLivenessConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.action_evidence = MediaPipeActionEvidence(config)
        self._invert_yaw = (config or ActiveLivenessConfig()).invert_yaw_for_challenge
        self.duplicate_frames = DuplicateFrameDetector()

    def observe(
        self,
        result,
        pose: HeadPose,
        timestamp_ms: int,
        *,
        face_count: int = 1,
        track_id: str | None = None,
    ):
        action = self.action_evidence.observe(result, timestamp_ms)
        return self.evaluator.observe(
            pose,
            timestamp_ms,
            face_count=face_count,
            track_id=track_id,
            action=action,
        )

    def build_evidence(
        self,
        result,
        pose: HeadPose | None,
        frame: Frame,
        face: FaceObservation | None,
        timestamp_ms: int,
        *,
        face_count: int = 1,
        track_id: str | None = None,
        motion_score: float | None = None,
        embedding: np.ndarray | None = None,
        passive: LivenessEvidence | None = None,
    ) -> FrameEvidence:
        # Capture owns time. A legacy caller must not supply a second clock/count.
        reason = None
        if timestamp_ms != frame.captured_at_ns // 1_000_000:
            reason = "timestamp_source_mismatch"
        actual_count = len(getattr(result, "face_landmarks", ()))
        if actual_count != face_count:
            reason = "face_count_source_mismatch"
        if face_count != 1:
            face = None
        if face_count == 1 and face is None:
            face_count = 0
            reason = "face_observation_missing"
        if face is not None:
            face = replace(face, quality=measure_quality(frame, face.bbox, face.landmarks))
        if reason is None and face_count == 1:
            try:
                matrix_pose = _matrix_pose_from_result(result)
                if pose is not None and pose != matrix_pose:
                    reason = "pose_source_mismatch"
                pose = matrix_pose
            except (HeadPoseError, ValueError, TypeError):
                reason = "head_pose_unavailable"
        face_quality = face.quality if face is not None else None
        duplicate = self.duplicate_frames.observe(frame)
        return FrameEvidence(
            frame=frame,
            face_count=face_count,
            face=face,
            track_id=track_id,
            pose=pose,
            quality=face_quality,
            lighting_score=(face_quality.brightness if face_quality else None),
            motion_score=motion_score,
            embedding=embedding,
            frame_fingerprint=DuplicateFrameDetector.fingerprint(frame),
            action=(
                self.action_evidence.observe(result, frame.captured_at_ns // 1_000_000)
                if reason is None and face_count == 1
                else None
            ),
            smile_score=(
                mediapipe_smile_score(result) if reason is None and face_count == 1 else None
            ),
            passive=passive,
            failure_reason=reason or ("duplicate_frame" if duplicate else None),
        )

    def observe_evidence(self, evidence: FrameEvidence):
        phase_before = self.evaluator.result.phase
        if evidence.pose is not None:
            evidence = replace(
                evidence,
                pose=replace(
                    evidence.pose,
                    yaw_degrees=_challenge_yaw(evidence.pose.yaw_degrees, invert=self._invert_yaw),
                ),
            )
        result = self.evaluator.observe_evidence(evidence)
        if result.status is ChallengeStatus.IN_PROGRESS and result.phase is not phase_before:
            self.action_evidence.reset()
        return result

    def is_neutral(self, evidence: FrameEvidence) -> bool:
        if evidence.pose is None or evidence.smile_score is None:
            return False
        pose = replace(
            evidence.pose,
            yaw_degrees=_challenge_yaw(evidence.pose.yaw_degrees, invert=self._invert_yaw),
        )
        return self.evaluator.is_neutral(pose)


@dataclass(frozen=True)
class LivenessSessionResult:
    """Overall decision: an active pass alone never means the attempt passed."""

    decision: EvidenceDecision
    reason: str | None
    passive: LivenessEvidence | None


class MediaPipeLivenessSession:
    """Runtime composition of MediaPipe active actions and temporal passive PAD."""

    def __init__(
        self,
        evaluator: ActiveLivenessChallengeEvaluator,
        passive_session: TemporalPassiveLivenessSession,
        config: ActiveLivenessConfig | None = None,
        *,
        post_active_config: NeutralPadConfig | None = None,
        evidence_guard: EvidenceStreamGuard | None = None,
        processor: Callable[[Frame], FrameEvidence] | None = None,
        embedding_extractor: Callable[[FrameEvidence], np.ndarray] | None = None,
    ) -> None:
        self.active = MediaPipeChallengeSessionAdapter(evaluator, config)
        self.passive = passive_session
        self.post_active_config = post_active_config or NeutralPadConfig()
        self._passive_evidence: LivenessEvidence | None = None
        self.guard = evidence_guard or EvidenceStreamGuard()
        self.processor = processor
        self.embedding_extractor = embedding_extractor
        self.embedding: np.ndarray | None = None
        self.last_evidence: FrameEvidence | None = None
        self.failure_reason: str | None = None
        self.phase = LivenessRuntimePhase.ACTIVE_CHALLENGE
        self._neutral_started_ms: int | None = None
        self._post_active_started_ms: int | None = None
        self._best_neutral_evidence: FrameEvidence | None = None

    @property
    def instruction(self) -> str:
        if self.phase is LivenessRuntimePhase.FACE_CAMERA:
            return "Face the camera"
        if self.phase is LivenessRuntimePhase.HOLD_STILL:
            return "Hold still"
        if self.phase is LivenessRuntimePhase.ACTIVE_CHALLENGE:
            return self.active.evaluator.result.phase.value.replace("_", " ").upper()
        return self.phase.value.replace("_", " ").upper()

    @property
    def result(self) -> LivenessSessionResult:
        if self.failure_reason:
            return LivenessSessionResult(
                EvidenceDecision.FAILED, self.failure_reason, self._passive_evidence
            )
        if self._passive_evidence is not None:
            return LivenessSessionResult(
                self._passive_evidence.decision,
                self._passive_evidence.reason,
                self._passive_evidence,
            )
        return LivenessSessionResult(EvidenceDecision.INCONCLUSIVE, None, None)

    def observe(
        self,
        result,
        pose: HeadPose,
        frame: Frame,
        face: FaceObservation,
        timestamp_ms: int,
        *,
        face_count: int = 1,
        track_id: str | None = None,
    ):
        try:
            evidence = self.active.build_evidence(
                result,
                pose,
                frame,
                face,
                timestamp_ms,
                face_count=face_count,
                track_id=track_id,
            )
        except (ValueError, TypeError, IndexError, AttributeError, cv2.error):
            evidence = FrameEvidence(
                frame, 0, None, None, None, failure_reason="invalid_action_evidence"
            )
        return self.observe_evidence(evidence)

    def observe_evidence(self, evidence: FrameEvidence):
        """Run active challenges, then collect a dedicated neutral PAD window."""
        if self.failure_reason or self._passive_evidence is not None:
            return self.active.evaluator.result
        evidence = self.guard.observe(evidence)
        self.last_evidence = evidence
        if evidence.failure_reason:
            self._fail(evidence.failure_reason)
            return self.active.evaluator.result

        timestamp_ms = evidence.frame.captured_at_ns // 1_000_000
        if self.phase is LivenessRuntimePhase.ACTIVE_CHALLENGE:
            active = self.active.observe_evidence(evidence)
            if active.status not in {ChallengeStatus.IN_PROGRESS, ChallengeStatus.COMPLETED}:
                self._fail(active.reason or "active_liveness_failed")
            elif active.status is ChallengeStatus.COMPLETED:
                self.phase = LivenessRuntimePhase.FACE_CAMERA
                self._post_active_started_ms = timestamp_ms
                self._neutral_started_ms = None
                self.passive.reset()
            return active

        if self.phase not in {LivenessRuntimePhase.FACE_CAMERA, LivenessRuntimePhase.HOLD_STILL}:
            return self.active.evaluator.result
        assert self._post_active_started_ms is not None
        if timestamp_ms - self._post_active_started_ms > self.post_active_config.timeout_ms:
            self._fail("neutral_pad_timeout")
            return self.active.evaluator.result

        expression_is_neutral = (
            evidence.smile_score is not None
            and evidence.smile_score <= self.post_active_config.max_smile_score
        )
        if not self.active.is_neutral(evidence) or not expression_is_neutral:
            self.phase = LivenessRuntimePhase.FACE_CAMERA
            self._neutral_started_ms = None
            self._best_neutral_evidence = None
            self.passive.reset()
            return self.active.evaluator.result

        if self._neutral_started_ms is None:
            self._neutral_started_ms = timestamp_ms
            self.phase = LivenessRuntimePhase.HOLD_STILL
        self._retain_best_neutral(evidence)
        self.passive.observe_evidence(evidence)
        if self.passive.result is not None:
            self._fail(self.passive.result.reason or "passive_liveness_failed")
        elif (
            timestamp_ms - self._neutral_started_ms >= self.post_active_config.neutral_dwell_ms
            and self.passive.ready
        ):
            self.finalize_passive()
        return self.active.evaluator.result

    def _retain_best_neutral(self, evidence: FrameEvidence) -> None:
        """Retain one sharp neutral candidate in memory for embedding extraction."""
        current = self._best_neutral_evidence
        if current is None or (
            evidence.quality is not None
            and (current.quality is None or evidence.quality.sharpness > current.quality.sharpness)
        ):
            self._best_neutral_evidence = evidence

    def take_embedding_candidate(self) -> FrameEvidence:
        """Transfer the best neutral frame once, after both liveness gates pass."""
        if self.result.decision is not EvidenceDecision.PASSED:
            raise RuntimeError("embedding candidate requires completed liveness")
        if self._best_neutral_evidence is None:
            raise RuntimeError("no neutral embedding candidate is available")
        evidence = self._best_neutral_evidence
        self._best_neutral_evidence = None
        return evidence

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.phase = LivenessRuntimePhase.FAILED
        self._neutral_started_ms = None
        self._best_neutral_evidence = None
        self._passive_evidence = self.passive.abort(reason)

    def __call__(self, frame: Frame) -> FrameEvidence:
        """Camera-worker callable. The injected headless processor owns inference."""
        if self.processor is None:
            raise RuntimeError("a headless evidence processor is required")
        if self.failure_reason or self._passive_evidence is not None:
            return FrameEvidence(
                frame, 0, None, None, None, failure_reason=self.failure_reason or "session_finished"
            )
        try:
            evidence = self.processor(frame)
        except (ValueError, TypeError, RuntimeError, OSError, cv2.error):
            evidence = FrameEvidence(
                frame, 0, None, None, None, failure_reason="evidence_processing_failed"
            )
        if evidence.frame is not frame:
            evidence = replace(evidence, failure_reason="frame_source_mismatch")
        self.observe_evidence(evidence)
        assert self.last_evidence is not None
        return self.last_evidence

    def finalize_passive(self) -> LivenessEvidence:
        """Finalize PAD only after active liveness has completed."""
        if self._passive_evidence is not None:
            return self._passive_evidence
        if self.active.evaluator.result.status.value != "completed":
            self._fail("active_liveness_incomplete")
            assert self._passive_evidence is not None
            return self._passive_evidence
        if self._passive_evidence is None:
            self._passive_evidence = self.passive.finalize()
        if self._passive_evidence.decision is not EvidenceDecision.PASSED:
            self.failure_reason = self._passive_evidence.reason or "passive_liveness_failed"
            self.phase = LivenessRuntimePhase.FAILED
            self._best_neutral_evidence = None
        else:
            if self.embedding_extractor is not None:
                try:
                    candidate = self._best_neutral_evidence
                    if candidate is None:
                        raise RuntimeError("neutral embedding candidate unavailable")
                    embedding = np.asarray(self.embedding_extractor(candidate), dtype=np.float32)
                    if (
                        embedding.ndim != 1
                        or not embedding.size
                        or not np.isfinite(embedding).all()
                    ):
                        raise ValueError("invalid embedding")
                    self.embedding = embedding
                except (ValueError, TypeError, RuntimeError, OSError, cv2.error):
                    self._best_neutral_evidence = None
                    self.failure_reason = "embedding_extraction_failed"
                    self.phase = LivenessRuntimePhase.FAILED
                    return self._passive_evidence
                self._best_neutral_evidence = None
            self.phase = LivenessRuntimePhase.COMPLETED
        return self._passive_evidence

    def cancel(self) -> LivenessSessionResult:
        """Terminate this one-use attempt and discard transient PAD frames."""
        if self._passive_evidence is None:
            self.active.action_evidence.reset()
            self.active.evaluator.cancel()
            self.guard.cancel()
            self.failure_reason = "cancelled"
            self._passive_evidence = self.passive.cancel()
            self._best_neutral_evidence = None
            self.embedding = None
            self._neutral_started_ms = None
            self.phase = LivenessRuntimePhase.CANCELLED
        return self.result


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
        self.state = "UNARMED"
        self.closed_frame_count = 0
        self.open_frame_count = 0
        self.last_blink_timestamp_ms: int | None = None

    def observe(self, eye_aspect_ratio: float, timestamp_ms: int) -> None:
        if self.state == "UNARMED":
            if eye_aspect_ratio >= self.open_eye_threshold:
                self.open_frame_count += 1
                if self.open_frame_count >= self.min_open_frames:
                    self.state = "OPEN"
                    self.open_frame_count = 0
            else:
                self.open_frame_count = 0
            return
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
                output_face_blendshapes=True,
                # Keep MediaPipe's facial transform available for validation
                # against the calibrated solvePnP estimate. The matrix is
                # diagnostic evidence; solvePnP remains the challenge input
                # until camera-based comparison establishes agreement.
                output_facial_transformation_matrixes=True,
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
            try:
                head_pose = _matrix_pose_from_result(result)
            except (HeadPoseError, IndexError, TypeError, ValueError) as exc:
                return _failed(
                    "head_pose_unavailable",
                    score=0.0,
                    challenge_completed=False,
                    reason=f"facial_transformation_matrix_invalid:{exc}",
                )

            challenge_yaw = _challenge_yaw(
                head_pose.yaw_degrees,
                invert=self.config.invert_yaw_for_challenge,
            )
            print(
                "[head-pose-debug]",
                f"timestamp_ms={timestamp_ms}",
                f"stage={challenge.stage}",
                f"ear={eye_aspect_ratio:.3f}",
                f"yaw={head_pose.yaw_degrees:.2f}",
                f"pitch={head_pose.pitch_degrees:.2f}",
                f"roll={head_pose.roll_degrees:.2f}",
                f"challenge_yaw={challenge_yaw:.2f}",
            )
            try:
                comparison = compare_pose_estimators(
                    result.facial_transformation_matrixes[0],
                    landmarks,
                    image_width=image_width,
                    image_height=image_height,
                )
            except (HeadPoseError, IndexError, TypeError, ValueError) as exc:
                print("[head-pose-compare] solvepnp=unavailable", f"reason={exc}")
            else:
                error = comparison.absolute_error_degrees
                print(
                    "[head-pose-compare]",
                    f"solvepnp_yaw={comparison.solvepnp_pose.yaw_degrees:.2f}",
                    f"solvepnp_pitch={comparison.solvepnp_pose.pitch_degrees:.2f}",
                    f"solvepnp_roll={comparison.solvepnp_pose.roll_degrees:.2f}",
                    f"error_yaw={error.yaw_degrees:.2f}",
                    f"error_pitch={error.pitch_degrees:.2f}",
                    f"error_roll={error.roll_degrees:.2f}",
                )
            challenge.observe(
                eye_aspect_ratio=eye_aspect_ratio,
                yaw=(
                    _challenge_yaw(
                        head_pose.yaw_degrees,
                        invert=self.config.invert_yaw_for_challenge,
                    )
                    if head_pose
                    else None
                ),
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
