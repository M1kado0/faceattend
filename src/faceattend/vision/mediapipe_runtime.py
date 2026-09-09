"""MediaPipe action evidence and active-then-passive liveness runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from faceattend.vision.active_liveness import ActiveLivenessChallengeEvaluator, ChallengeStatus
from faceattend.vision.challenge_session import ChallengeAction
from faceattend.vision.evidence import EvidenceStreamGuard
from faceattend.vision.passive_liveness import TemporalPassiveLivenessSession
from faceattend.vision.types import EvidenceDecision, Frame, FrameEvidence, LivenessEvidence

_RIGHT_EYE = (33, 159, 158, 133, 153, 145)
_LEFT_EYE = (362, 380, 374, 263, 386, 385)


@dataclass(frozen=True, slots=True)
class ActiveLivenessConfig:
    closed_eye_threshold: float = 0.08
    open_eye_threshold: float = 0.14
    min_closed_frames: int = 2
    min_open_frames: int = 2
    blink_cooldown_ms: int = 250
    smile_score_threshold: float = 0.5
    min_smile_frames: int = 2
    invert_yaw_for_challenge: bool = True


@dataclass(frozen=True, slots=True)
class NeutralPadConfig:
    neutral_dwell_ms: int = 1_000
    timeout_ms: int = 8_000
    max_smile_score: float = 0.3


class LivenessRuntimePhase(StrEnum):
    ACTIVE_CHALLENGE = "active_challenge"
    FACE_CAMERA = "face_camera"
    HOLD_STILL = "hold_still"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class LivenessSessionResult:
    decision: EvidenceDecision
    reason: str | None
    passive: LivenessEvidence | None


class _BlinkCounter:
    def __init__(self, config: ActiveLivenessConfig) -> None:
        self.config = config
        self.blinks = 0
        self.state = "unarmed"
        self.closed_frames = 0
        self.open_frames = 0
        self.last_blink_ms: int | None = None

    def observe(self, ear: float, timestamp_ms: int) -> None:
        if self.state == "unarmed":
            self.open_frames = self.open_frames + 1 if ear >= self.config.open_eye_threshold else 0
            if self.open_frames >= self.config.min_open_frames:
                self.state, self.open_frames = "open", 0
            return
        if self.state in {"open", "closed_candidate"}:
            if ear <= self.config.closed_eye_threshold:
                self.closed_frames += 1
                self.state = "closed_candidate"
                if self.closed_frames >= self.config.min_closed_frames:
                    self.state, self.open_frames = "closed", 0
            else:
                self.state, self.closed_frames = "open", 0
            return
        if ear >= self.config.open_eye_threshold:
            self.open_frames += 1
            if self.open_frames >= self.config.min_open_frames:
                if (
                    self.last_blink_ms is None
                    or timestamp_ms - self.last_blink_ms >= self.config.blink_cooldown_ms
                ):
                    self.blinks += 1
                    self.last_blink_ms = timestamp_ms
                self.state, self.closed_frames, self.open_frames = "open", 0, 0
        else:
            self.open_frames = 0


def mediapipe_smile_score(result: Any) -> float:
    faces: list[Any] = list(getattr(result, "face_blendshapes", ()))
    if len(faces) != 1:
        return 0.0
    values = {
        getattr(item, "category_name", ""): float(getattr(item, "score", 0.0)) for item in faces[0]
    }
    scores = [values[name] for name in ("mouthSmileLeft", "mouthSmileRight") if name in values]
    return sum(scores) / len(scores) if scores else 0.0


class MediaPipeActionEvidence:
    """Produce blink/smile events from Face Landmarker results."""

    def __init__(self, config: ActiveLivenessConfig | None = None) -> None:
        self.config = config or ActiveLivenessConfig()
        self.reset()

    def reset(self) -> None:
        self.blinks = _BlinkCounter(self.config)
        self._smile_armed = False
        self._smile_frames = 0

    def observe(self, result: Any, timestamp_ms: int) -> ChallengeAction | None:
        faces: list[Any] = list(getattr(result, "face_landmarks", ()))
        if len(faces) != 1:
            self._smile_frames = 0
            return None
        before = self.blinks.blinks
        ear = _mean_ear(faces[0])
        if not np.isfinite(ear):
            raise ValueError("invalid_eye_evidence")
        self.blinks.observe(ear, timestamp_ms)
        if self.blinks.blinks > before:
            self._smile_frames = 0
            return ChallengeAction.BLINK
        smile = mediapipe_smile_score(result)
        if smile < self.config.smile_score_threshold:
            self._smile_armed, self._smile_frames = True, 0
        elif self._smile_armed:
            self._smile_frames += 1
            if self._smile_frames >= self.config.min_smile_frames:
                self._smile_armed, self._smile_frames = False, 0
                return ChallengeAction.SMILE
        return None


class MediaPipeChallengeSessionAdapter:
    """Compatibility boundary between MediaPipe action evidence and the evaluator."""

    def __init__(
        self,
        evaluator: ActiveLivenessChallengeEvaluator,
        config: ActiveLivenessConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.action_evidence = MediaPipeActionEvidence(config)

    def reset(self) -> None:
        self.action_evidence.reset()


class MediaPipeLivenessSession:
    """Run randomized active liveness, then a neutral temporal PAD window."""

    def __init__(
        self,
        evaluator: ActiveLivenessChallengeEvaluator,
        passive_session: TemporalPassiveLivenessSession,
        *,
        processor: Callable[[Frame], FrameEvidence],
        config: ActiveLivenessConfig | None = None,
        post_active_config: NeutralPadConfig | None = None,
    ) -> None:
        self.active = MediaPipeChallengeSessionAdapter(evaluator, config)
        self.passive = passive_session
        self.config = config or ActiveLivenessConfig()
        self.post_active = post_active_config or NeutralPadConfig()
        self.guard = EvidenceStreamGuard()
        self.processor = processor
        self.phase = LivenessRuntimePhase.ACTIVE_CHALLENGE
        self.failure_reason: str | None = None
        self._passive_evidence: LivenessEvidence | None = None
        self._post_active_started_ms: int | None = None
        self._neutral_started_ms: int | None = None
        self._best: FrameEvidence | None = None
        self._candidates: list[FrameEvidence] = []
        self.last_evidence: FrameEvidence | None = None

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

    def __call__(self, frame: Frame) -> FrameEvidence:
        if self.failure_reason or self._passive_evidence is not None:
            return FrameEvidence(
                frame, 0, None, None, None, failure_reason=self.failure_reason or "session_finished"
            )
        try:
            evidence = self.processor(frame)
        except (ValueError, TypeError, RuntimeError, OSError):
            evidence = FrameEvidence(
                frame, 0, None, None, None, failure_reason="evidence_processing_failed"
            )
        self._observe(self.guard.observe(evidence))
        assert self.last_evidence is not None
        return self.last_evidence

    def _observe(self, evidence: FrameEvidence) -> None:
        self.last_evidence = evidence
        if evidence.failure_reason:
            self._fail(evidence.failure_reason)
            return
        timestamp_ms = evidence.frame.captured_at_ns // 1_000_000
        if self.phase is LivenessRuntimePhase.ACTIVE_CHALLENGE:
            action = evidence.action
            if evidence.pose is not None and self.config.invert_yaw_for_challenge:
                evidence = replace(
                    evidence, pose=replace(evidence.pose, yaw_degrees=-evidence.pose.yaw_degrees)
                )
            result = self.active.evaluator.observe_evidence(replace(evidence, action=action))
            if result.status not in {ChallengeStatus.IN_PROGRESS, ChallengeStatus.COMPLETED}:
                self._fail(result.reason or "active_liveness_failed")
            elif result.status is ChallengeStatus.COMPLETED:
                self.phase, self._post_active_started_ms = (
                    LivenessRuntimePhase.FACE_CAMERA,
                    timestamp_ms,
                )
                self.passive.reset()
            return
        if (
            self._post_active_started_ms is None
            or timestamp_ms - self._post_active_started_ms > self.post_active.timeout_ms
        ):
            self._fail("neutral_pad_timeout")
            return
        neutral = evidence.pose is not None and self.active.evaluator.is_neutral(
            replace(
                evidence.pose,
                yaw_degrees=-evidence.pose.yaw_degrees
                if self.config.invert_yaw_for_challenge
                else evidence.pose.yaw_degrees,
            )
        )
        if (
            not neutral
            or evidence.smile_score is None
            or evidence.smile_score > self.post_active.max_smile_score
        ):
            self.phase, self._neutral_started_ms, self._best = (
                LivenessRuntimePhase.FACE_CAMERA,
                None,
                None,
            )
            self._candidates.clear()
            self.passive.reset()
            return
        if self._neutral_started_ms is None:
            self._neutral_started_ms, self.phase = timestamp_ms, LivenessRuntimePhase.HOLD_STILL
        if len(self._candidates) < self.passive.max_frames:
            self._candidates.append(evidence)
        if self._best is None or (
            evidence.quality is not None
            and self._best.quality is not None
            and evidence.quality.sharpness > self._best.quality.sharpness
        ):
            self._best = evidence
        self.passive.observe_evidence(evidence)
        if self.passive.result is not None:
            self._fail(self.passive.result.reason or "passive_liveness_failed")
        elif (
            timestamp_ms - self._neutral_started_ms >= self.post_active.neutral_dwell_ms
            and self.passive.ready
        ):
            self._passive_evidence = self.passive.finalize()
            if self._passive_evidence.decision is EvidenceDecision.PASSED:
                self.phase = LivenessRuntimePhase.COMPLETED
            else:
                self._fail(self._passive_evidence.reason or "passive_liveness_failed")

    def take_embedding_candidates(
        self, *, min_candidates: int = 3, max_candidates: int = 5
    ) -> tuple[FrameEvidence, ...]:
        if (
            self.result.decision is not EvidenceDecision.PASSED
            or len(self._candidates) < min_candidates
        ):
            raise RuntimeError("completed liveness and sufficient candidates are required")
        candidates = tuple(self._candidates[:max_candidates])
        self._clear_candidates()
        return candidates

    def take_embedding_candidate(self) -> FrameEvidence:
        if self.result.decision is not EvidenceDecision.PASSED or self._best is None:
            raise RuntimeError("completed liveness and a neutral candidate are required")
        result = self._best
        self._clear_candidates()
        return result

    def _clear_candidates(self) -> None:
        self._candidates.clear()
        self._best = None

    def _fail(self, reason: str) -> None:
        self.failure_reason, self.phase = reason, LivenessRuntimePhase.FAILED
        self._passive_evidence = self.passive.abort(reason)
        self._clear_candidates()

    def cancel(self) -> LivenessSessionResult:
        if self._passive_evidence is None:
            self.active.evaluator.cancel()
            self.guard.cancel()
            self.failure_reason = "cancelled"
            self._passive_evidence = self.passive.cancel()
            self.phase = LivenessRuntimePhase.CANCELLED
        self._clear_candidates()
        return self.result


def _mean_ear(landmarks: Any) -> float:
    return (_ear(landmarks, _LEFT_EYE) + _ear(landmarks, _RIGHT_EYE)) / 2


def _ear(landmarks: Any, indices: tuple[int, int, int, int, int, int]) -> float:
    points = [
        np.array([landmarks[index].x, landmarks[index].y], dtype=np.float32) for index in indices
    ]
    horizontal = 2 * np.linalg.norm(points[0] - points[3])
    return (
        0.0
        if horizontal == 0
        else float(
            (np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4]))
            / horizontal
        )
    )
