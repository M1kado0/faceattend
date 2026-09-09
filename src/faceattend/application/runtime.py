"""Headless lifecycle and presentation contracts for the desktop runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from faceattend.vision.types import Frame, FrameEvidence


class DesktopMode(StrEnum):
    HOME = "home"
    REGISTRATION = "registration"
    ATTENDANCE = "attendance"


class RuntimeStatus(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NO_CAMERA = "no_camera"
    NO_MODEL = "no_model"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class SessionPresentation:
    """Immutable facts rendered by Qt; widgets never award progress."""

    mode: DesktopMode
    status: RuntimeStatus
    instruction: str
    completed_challenges: int
    total_challenges: int
    evidence: FrameEvidence | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.completed_challenges < 0 or self.total_challenges < 0:
            raise ValueError("challenge counts must be non-negative")
        if self.completed_challenges > self.total_challenges:
            raise ValueError("completed challenges cannot exceed total challenges")

    @property
    def progress_percent(self) -> int:
        if self.total_challenges == 0:
            return 0
        return int((100 * self.completed_challenges) / self.total_challenges)


class SessionProcessor(Protocol):
    """One inference-thread-owned liveness/application attempt."""

    def __call__(self, frame: Frame) -> FrameEvidence: ...

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class CallbackSessionProcessor:
    """Adapt the existing Phase 2 processor/session without importing Qt."""

    def __init__(
        self,
        process: Callable[[Frame], FrameEvidence],
        present: Callable[[FrameEvidence], SessionPresentation],
        *,
        cancel: Callable[[], object] | None = None,
        close: Callable[[], object] | None = None,
    ) -> None:
        self._process = process
        self._present = present
        self._cancel = cancel
        self._close = close

    def __call__(self, frame: Frame) -> FrameEvidence:
        return self._process(frame)

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        return self._present(evidence)

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel()

    def close(self) -> None:
        if self._close is not None:
            self._close()


class DesktopLifecycle:
    """Explicit restartable runtime state independent of Qt."""

    def __init__(self) -> None:
        self.mode = DesktopMode.HOME
        self.status = RuntimeStatus.IDLE
        self.reason: str | None = None

    def start(self, mode: DesktopMode) -> None:
        if mode is DesktopMode.HOME:
            raise ValueError("home is not a capture mode")
        if self.status in {RuntimeStatus.STARTING, RuntimeStatus.RUNNING}:
            raise RuntimeError("desktop runtime is already active")
        if self.status is RuntimeStatus.SHUTDOWN:
            raise RuntimeError("desktop runtime has shut down")
        self.mode = mode
        self.status = RuntimeStatus.STARTING
        self.reason = None

    def mark_running(self) -> None:
        if self.status is not RuntimeStatus.STARTING:
            raise RuntimeError("runtime must be starting before it can run")
        self.status = RuntimeStatus.RUNNING

    def complete(self) -> None:
        if self.status is not RuntimeStatus.RUNNING:
            raise RuntimeError("only a running session can complete")
        self.status = RuntimeStatus.COMPLETED

    def cancel(self) -> None:
        if self.status in {RuntimeStatus.STARTING, RuntimeStatus.RUNNING}:
            self.status = RuntimeStatus.CANCELLED
            self.reason = "cancelled"

    def fail(self, reason: str) -> None:
        self.status = RuntimeStatus.FAILED
        self.reason = reason

    def fail_no_camera(self, reason: str) -> None:
        self.status = RuntimeStatus.NO_CAMERA
        self.reason = reason

    def fail_no_model(self, reason: str) -> None:
        self.status = RuntimeStatus.NO_MODEL
        self.reason = reason

    def shutdown(self) -> None:
        self.status = RuntimeStatus.SHUTDOWN
        self.reason = None
