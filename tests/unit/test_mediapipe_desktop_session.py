"""Presentation bridge tests for the existing Phase 2 liveness runtime."""

from types import SimpleNamespace

import numpy as np

from faceattend.application.mediapipe_session import MediaPipeDesktopSessionProcessor
from faceattend.application.runtime import DesktopMode, RuntimeStatus
from faceattend.vision.types import Frame, FrameEvidence
from ml.liveness.mediapipe_active import LivenessRuntimePhase


def _evidence() -> FrameEvidence:
    frame = Frame(np.zeros((2, 2, 3), dtype=np.uint8), 100, 1)
    return FrameEvidence(frame, 0, None, None, None)


class _Session:
    def __init__(self) -> None:
        self.phase = LivenessRuntimePhase.ACTIVE_CHALLENGE
        self.instruction = "TURN RIGHT"
        self.failure_reason = None
        result = SimpleNamespace(completed_phases=("blink",))
        evaluator = SimpleNamespace(phases=("blink", "right"), result=result)
        self.active = SimpleNamespace(evaluator=evaluator)
        self.cancelled = False

    def __call__(self, _frame: Frame) -> FrameEvidence:
        return _evidence()

    def cancel(self) -> object:
        self.cancelled = True
        return None


def test_adapter_uses_session_instruction_progress_and_neutral_guidance() -> None:
    session = _Session()
    adapter = MediaPipeDesktopSessionProcessor(DesktopMode.ATTENDANCE, session)

    active = adapter.presentation(_evidence())
    assert active.instruction == "TURN RIGHT"
    assert active.progress_percent == 50

    session.phase = LivenessRuntimePhase.HOLD_STILL
    session.instruction = "Hold still"
    neutral = adapter.presentation(_evidence())
    assert neutral.instruction == "Hold still"
    assert neutral.progress_percent == 100
    assert neutral.status is RuntimeStatus.RUNNING

    session.phase = LivenessRuntimePhase.FAILED
    session.failure_reason = "multiple_faces"
    failed = adapter.presentation(_evidence())
    assert failed.status is RuntimeStatus.FAILED
    assert failed.failure_reason == "multiple_faces"
