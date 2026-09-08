"""Desktop presentation adapter for the existing MediaPipe liveness runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.vision.types import Frame, FrameEvidence


class RuntimePhase(Protocol):
    @property
    def value(self) -> str: ...


class ChallengeProgress(Protocol):
    @property
    def completed_phases(self) -> tuple[object, ...]: ...


class ChallengeEvaluatorView(Protocol):
    @property
    def phases(self) -> tuple[object, ...]: ...

    @property
    def result(self) -> ChallengeProgress: ...


class ActiveSessionView(Protocol):
    @property
    def evaluator(self) -> ChallengeEvaluatorView: ...


class MediaPipeSessionView(Protocol):
    @property
    def phase(self) -> RuntimePhase: ...

    @property
    def instruction(self) -> str: ...

    @property
    def failure_reason(self) -> str | None: ...

    @property
    def active(self) -> ActiveSessionView: ...

    def __call__(self, frame: Frame) -> FrameEvidence: ...

    def cancel(self) -> object: ...


class MediaPipeDesktopSessionProcessor:
    """Expose Phase 2 liveness as the inference worker's session contract."""

    def __init__(
        self,
        mode: DesktopMode,
        session: MediaPipeSessionView,
        *,
        close: Callable[[], object] | None = None,
    ) -> None:
        if mode is DesktopMode.HOME:
            raise ValueError("home is not a liveness mode")
        self.mode = mode
        self.session = session
        self._close = close
        self._closed = False

    def __call__(self, frame: Frame) -> FrameEvidence:
        return self.session(frame)

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        phase = self.session.phase.value
        status = {
            "completed": RuntimeStatus.COMPLETED,
            "failed": RuntimeStatus.FAILED,
            "cancelled": RuntimeStatus.CANCELLED,
        }.get(phase, RuntimeStatus.RUNNING)
        evaluator = self.session.active.evaluator
        completed = len(evaluator.result.completed_phases)
        total = len(evaluator.phases)
        if phase in {
            "face_camera",
            "hold_still",
            "completed",
        }:
            completed = total
        failure_reason = self.session.failure_reason
        if status is RuntimeStatus.FAILED and failure_reason is None:
            failure_reason = evidence.failure_reason or "liveness failed"
        return SessionPresentation(
            mode=self.mode,
            status=status,
            instruction=self.session.instruction,
            completed_challenges=completed,
            total_challenges=total,
            evidence=evidence,
            failure_reason=failure_reason,
        )

    def cancel(self) -> None:
        self.session.cancel()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._close is not None:
                self._close()
