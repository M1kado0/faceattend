"""Offscreen tests for evidence-driven Qt Widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.gui.views import RegistrationView

_APP = QApplication.instance() or QApplication([])


def test_session_view_only_renders_supplied_instruction_progress_and_failure() -> None:
    view = RegistrationView()

    view.apply_presentation(
        SessionPresentation(
            DesktopMode.REGISTRATION,
            RuntimeStatus.RUNNING,
            "TURN LEFT",
            1,
            3,
        )
    )
    assert view.instruction_label.text() == "TURN LEFT"
    assert view.progress_bar.value() == 33

    view.apply_presentation(
        SessionPresentation(
            DesktopMode.REGISTRATION,
            RuntimeStatus.FAILED,
            "FAILED",
            1,
            3,
            failure_reason="multiple_faces",
        )
    )
    assert view.status_label.text() == "multiple faces"
    assert view.progress_bar.value() == 33
