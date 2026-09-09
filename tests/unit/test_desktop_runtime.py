"""Headless desktop-runtime contracts shared with Qt workers."""

import numpy as np

from faceattend.application.runtime import (
    CallbackSessionProcessor,
    DesktopLifecycle,
    DesktopMode,
    RuntimeStatus,
    SessionPresentation,
)
from faceattend.vision.types import Frame, FrameEvidence


def _evidence() -> FrameEvidence:
    frame = Frame(np.zeros((2, 2, 3), dtype=np.uint8), 100, 1)
    return FrameEvidence(frame, 0, None, None, None, failure_reason="no_face")


def test_desktop_lifecycle_has_explicit_start_cancel_error_and_shutdown_states() -> None:
    lifecycle = DesktopLifecycle()

    lifecycle.start(DesktopMode.REGISTRATION)
    lifecycle.mark_running()
    lifecycle.cancel()
    assert lifecycle.status is RuntimeStatus.CANCELLED

    lifecycle.start(DesktopMode.ATTENDANCE)
    lifecycle.fail_no_camera("camera unavailable")
    assert lifecycle.status is RuntimeStatus.NO_CAMERA
    assert lifecycle.reason == "camera unavailable"

    lifecycle.start(DesktopMode.ATTENDANCE)
    lifecycle.fail_no_model("model unavailable")
    lifecycle.shutdown()
    assert lifecycle.status is RuntimeStatus.SHUTDOWN


def test_presentation_progress_is_derived_from_session_completion() -> None:
    presentation = SessionPresentation(
        mode=DesktopMode.REGISTRATION,
        status=RuntimeStatus.RUNNING,
        instruction="TURN LEFT",
        completed_challenges=1,
        total_challenges=3,
        evidence=_evidence(),
    )

    assert presentation.progress_percent == 33


def test_callback_adapter_reuses_headless_processor_and_presentation_evidence() -> None:
    evidence = _evidence()
    cancelled: list[bool] = []
    closed: list[bool] = []
    adapter = CallbackSessionProcessor(
        lambda _frame: evidence,
        lambda item: SessionPresentation(
            DesktopMode.ATTENDANCE,
            RuntimeStatus.RUNNING,
            "BLINK",
            1,
            2,
            item,
        ),
        cancel=lambda: cancelled.append(True),
        close=lambda: closed.append(True),
    )

    assert adapter(evidence.frame) is evidence
    assert adapter.presentation(evidence).instruction == "BLINK"
    adapter.cancel()
    adapter.close()
    assert cancelled == [True]
    assert closed == [True]
