"""Calibration-aware, headless active-liveness challenge evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from faceattend.vision.challenge_session import (
    ChallengeAction,
    ChallengeSession,
    ChallengeSessionStatus,
)
from faceattend.vision.evidence import evidence_failure
from faceattend.vision.types import FrameEvidence, HeadPose


class ChallengePhase(StrEnum):
    BLINK = "blink"
    SMILE = "smile"
    NEUTRAL = "neutral"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ROLL_LEFT = "roll_left"
    ROLL_RIGHT = "roll_right"


class ChallengeStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    WRONG_DIRECTION = "wrong_direction"
    INSUFFICIENT_DWELL = "insufficient_dwell"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    FACE_SUBSTITUTION = "face_substitution"
    INVALID_EVIDENCE = "invalid_evidence"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ActiveLivenessCalibration:
    """Per-camera pose baseline and configurable enter/exit thresholds."""

    neutral_yaw: float = 0.0
    neutral_pitch: float = 0.0
    neutral_roll: float = 0.0
    neutral_yaw_tolerance: float = 12.0
    neutral_pitch_tolerance: float = 10.0
    neutral_roll_tolerance: float = 10.0
    yaw_enter: float = 35.0
    yaw_exit: float = 25.0
    pitch_enter: float = 12.0
    pitch_exit: float = 8.0
    roll_enter: float = 15.0
    roll_exit: float = 10.0
    dwell_ms: int = 300
    timeout_ms: int = 10_000
    min_valid_frames: int = 2

    def __post_init__(self) -> None:
        for name in (
            "neutral_yaw_tolerance",
            "neutral_pitch_tolerance",
            "neutral_roll_tolerance",
            "yaw_enter",
            "yaw_exit",
            "pitch_enter",
            "pitch_exit",
            "roll_enter",
            "roll_exit",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for axis in ("yaw", "pitch", "roll"):
            if getattr(self, f"{axis}_enter") < getattr(self, f"{axis}_exit"):
                raise ValueError(f"{axis}_enter must be >= {axis}_exit")
        if self.dwell_ms <= 0 or self.timeout_ms <= 0 or self.min_valid_frames <= 0:
            raise ValueError("timings and min_valid_frames must be positive")


@dataclass(frozen=True, slots=True)
class ChallengeResult:
    status: ChallengeStatus
    phase: ChallengePhase
    completed_phases: tuple[ChallengePhase, ...]
    reason: str | None = None


class ActiveLivenessChallengeEvaluator:
    """Evaluate ordered pose phases with dwell, hysteresis, and fail-closed states."""

    def __init__(
        self,
        phases: tuple[ChallengePhase, ...] | None = None,
        calibration: ActiveLivenessCalibration | None = None,
        *,
        session: ChallengeSession | None = None,
    ) -> None:
        if session is not None:
            snapshot = session.snapshot
            if snapshot.status is not ChallengeSessionStatus.IN_PROGRESS:
                raise ValueError("session must be started before creating an evaluator")
            phases = tuple(_phase_for_action(action) for action in snapshot.sequence)
        if phases is None:
            raise ValueError("phases or an active session is required")
        if not phases:
            raise ValueError("at least one challenge phase is required")
        self.phases = phases
        self.calibration = calibration or ActiveLivenessCalibration()
        self.session = session
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._started_at: int | None = None
        self._last_timestamp: int | None = None
        self._dwell_started_at: int | None = None
        self._valid_frame_count = 0
        self._neutral_ready = False
        self._track_id: str | None = None
        self._status = ChallengeStatus.IN_PROGRESS
        self._reason: str | None = None

    def _observe_session_clock(self, timestamp_ms: int) -> bool:
        """Observe the shared session before any evidence branch.

        Returns False when the session has already timed out. Keeping this at
        the top of ``observe`` prevents neutral waits, missing faces, and dwell
        resets from bypassing session deadlines.
        """
        if self.session is None:
            return True
        try:
            state = self.session.observe(timestamp_ms)
        except ValueError:
            self._fail(ChallengeStatus.TIMEOUT, "timestamp_not_monotonic")
            return False
        if state.status is ChallengeSessionStatus.SESSION_TIMEOUT:
            self._fail(ChallengeStatus.TIMEOUT, "session_timeout")
            return False
        if state.status is ChallengeSessionStatus.CHALLENGE_TIMEOUT:
            self._fail(ChallengeStatus.TIMEOUT, "challenge_timeout")
            return False
        return True

    @property
    def result(self) -> ChallengeResult:
        return ChallengeResult(
            status=self._status,
            phase=self.phases[min(self._index, len(self.phases) - 1)],
            completed_phases=self.phases[: self._index],
            reason=self._reason,
        )

    def observe(
        self,
        pose: HeadPose | None,
        timestamp_ms: int,
        *,
        face_count: int = 1,
        track_id: str | None = None,
        action: ChallengeAction | None = None,
    ) -> ChallengeResult:
        if self._status is not ChallengeStatus.IN_PROGRESS:
            return self.result
        if self._started_at is None:
            self._started_at = (
                self.session.snapshot.started_at_ms if self.session is not None else timestamp_ms
            )
        if self._last_timestamp is not None and timestamp_ms <= self._last_timestamp:
            return self._fail(ChallengeStatus.TIMEOUT, "timestamp_not_monotonic")
        self._last_timestamp = timestamp_ms
        if not self._observe_session_clock(timestamp_ms):
            return self.result
        assert self._started_at is not None
        if timestamp_ms - self._started_at > self.calibration.timeout_ms:
            return self._fail(
                ChallengeStatus.INSUFFICIENT_DWELL
                if self._dwell_started_at is not None
                else ChallengeStatus.TIMEOUT,
                "challenge_timeout",
            )
        if face_count == 0:
            return self._fail(ChallengeStatus.NO_FACE, "no_face")
        if face_count > 1:
            return self._fail(ChallengeStatus.MULTIPLE_FACES, "multiple_faces")
        if self._track_id is None:
            self._track_id = track_id
        elif track_id != self._track_id:
            return self._fail(ChallengeStatus.FACE_SUBSTITUTION, "track_changed")
        if pose is None:
            return self._fail(ChallengeStatus.NO_FACE, "pose_unavailable")

        phase = self.phases[self._index]
        if (
            phase
            in {
                ChallengePhase.LEFT,
                ChallengePhase.RIGHT,
                ChallengePhase.UP,
                ChallengePhase.DOWN,
                ChallengePhase.ROLL_LEFT,
                ChallengePhase.ROLL_RIGHT,
            }
            and not self._neutral_ready
        ):
            if self._matches(ChallengePhase.NEUTRAL, pose, entering=True):
                self._neutral_ready = True
            else:
                opposite = self._opposite(phase)
                if opposite is not None and self._matches(opposite, pose, entering=True):
                    return self._fail(ChallengeStatus.WRONG_DIRECTION, f"expected_{phase.value}")
                return self.result
        entering = self._dwell_started_at is None
        matched = self._matches_action(phase, action) or (
            phase not in {ChallengePhase.BLINK, ChallengePhase.SMILE}
            and self._matches(phase, pose, entering=entering)
        )
        if matched:
            if phase in {ChallengePhase.BLINK, ChallengePhase.SMILE}:
                self._index += 1
                if self.session is not None:
                    session_state = self.session.complete_current(timestamp_ms)
                    if session_state.status is not ChallengeSessionStatus.IN_PROGRESS:
                        if session_state.status is ChallengeSessionStatus.COMPLETED:
                            self._status = ChallengeStatus.COMPLETED
                        else:
                            return self._fail(
                                ChallengeStatus.TIMEOUT,
                                session_state.reason or "session_failed",
                            )
                if self._index == len(self.phases):
                    self._status = ChallengeStatus.COMPLETED
                return self.result
            if self._dwell_started_at is None:
                self._dwell_started_at = timestamp_ms
                self._valid_frame_count = 1
            else:
                self._valid_frame_count += 1
            if (
                self._dwell_started_at is not None
                and self._valid_frame_count >= self.calibration.min_valid_frames
                and timestamp_ms - self._dwell_started_at >= self.calibration.dwell_ms
            ):
                completed_neutral = phase is ChallengePhase.NEUTRAL
                self._index += 1
                self._dwell_started_at = None
                self._valid_frame_count = 0
                self._neutral_ready = completed_neutral
                if self.session is not None:
                    session_state = self.session.complete_current(timestamp_ms)
                    if session_state.status is not ChallengeSessionStatus.IN_PROGRESS:
                        if session_state.status is ChallengeSessionStatus.COMPLETED:
                            self._status = ChallengeStatus.COMPLETED
                        else:
                            return self._fail(
                                ChallengeStatus.TIMEOUT,
                                session_state.reason or "session_failed",
                            )
                if self._index == len(self.phases):
                    self._status = ChallengeStatus.COMPLETED
                    return self.result
            return self.result

        if self._dwell_started_at is not None and not self._matches(phase, pose, entering=False):
            self._dwell_started_at = None
            self._valid_frame_count = 0

        opposite = self._opposite(phase)
        if opposite is not None and self._matches(opposite, pose, entering=True):
            return self._fail(ChallengeStatus.WRONG_DIRECTION, f"expected_{phase.value}")
        return self.result

    def observe_evidence(self, evidence: FrameEvidence) -> ChallengeResult:
        """Consume the active-liveness fields from one frame evidence record."""
        reason = evidence_failure(evidence)
        if reason:
            return self.reject(reason)
        return self.observe(
            evidence.pose,
            evidence.frame.captured_at_ns // 1_000_000,
            face_count=evidence.face_count,
            track_id=evidence.track_id,
            action=evidence.action,
        )

    def reject(self, reason: str) -> ChallengeResult:
        """Latch an upstream evidence failure; it cannot later become a pass."""
        if self._status is not ChallengeStatus.IN_PROGRESS:
            return self.result
        status = {
            "no_face": ChallengeStatus.NO_FACE,
            "multiple_faces": ChallengeStatus.MULTIPLE_FACES,
            "track_changed": ChallengeStatus.FACE_SUBSTITUTION,
        }.get(reason, ChallengeStatus.INVALID_EVIDENCE)
        return self._fail(status, reason)

    def cancel(self) -> ChallengeResult:
        return self._fail(ChallengeStatus.CANCELLED, "cancelled")

    def is_neutral(self, pose: HeadPose) -> bool:
        """Apply the configured neutral pose contract without advancing state."""
        return self._matches(ChallengePhase.NEUTRAL, pose, entering=True)

    def _fail(self, status: ChallengeStatus, reason: str) -> ChallengeResult:
        self._status = status
        self._reason = reason
        if self.session is not None:
            self.session.fail(reason)
        return self.result

    def _matches(self, phase: ChallengePhase, pose: HeadPose, *, entering: bool) -> bool:
        c = self.calibration
        yaw = pose.yaw_degrees - c.neutral_yaw
        pitch = pose.pitch_degrees - c.neutral_pitch
        roll = pose.roll_degrees - c.neutral_roll
        if phase is ChallengePhase.NEUTRAL:
            return (
                abs(yaw) <= c.neutral_yaw_tolerance
                and abs(pitch) <= c.neutral_pitch_tolerance
                and abs(roll) <= c.neutral_roll_tolerance
            )
        threshold = {
            "yaw": c.yaw_enter if entering else c.yaw_exit,
            "pitch": c.pitch_enter if entering else c.pitch_exit,
            "roll": c.roll_enter if entering else c.roll_exit,
        }
        if phase is ChallengePhase.LEFT:
            return yaw <= -threshold["yaw"]
        if phase is ChallengePhase.RIGHT:
            return yaw >= threshold["yaw"]
        if phase is ChallengePhase.UP:
            return pitch <= -threshold["pitch"]
        if phase is ChallengePhase.DOWN:
            return pitch >= threshold["pitch"]
        if phase is ChallengePhase.ROLL_LEFT:
            return roll <= -threshold["roll"]
        return roll >= threshold["roll"]

    @staticmethod
    def _matches_action(phase: ChallengePhase, action: ChallengeAction | None) -> bool:
        return action is not None and _phase_for_action(action) is phase

    @staticmethod
    def _opposite(phase: ChallengePhase) -> ChallengePhase | None:
        return {
            ChallengePhase.LEFT: ChallengePhase.RIGHT,
            ChallengePhase.RIGHT: ChallengePhase.LEFT,
            ChallengePhase.UP: ChallengePhase.DOWN,
            ChallengePhase.DOWN: ChallengePhase.UP,
            ChallengePhase.ROLL_LEFT: ChallengePhase.ROLL_RIGHT,
            ChallengePhase.ROLL_RIGHT: ChallengePhase.ROLL_LEFT,
        }.get(phase)


def _phase_for_action(action: ChallengeAction) -> ChallengePhase:
    """Map randomized session actions to evaluator phases."""

    return {
        ChallengeAction.BLINK: ChallengePhase.BLINK,
        ChallengeAction.SMILE: ChallengePhase.SMILE,
        ChallengeAction.TURN_LEFT: ChallengePhase.LEFT,
        ChallengeAction.TURN_RIGHT: ChallengePhase.RIGHT,
        ChallengeAction.LOOK_UP: ChallengePhase.UP,
    }[action]
