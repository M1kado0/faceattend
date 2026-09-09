"""Offscreen tests for evidence-driven Qt Widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.gui.views import AttendanceView, RegistrationView

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


def test_registration_and_attendance_views_collect_only_operator_inputs() -> None:
    registration = RegistrationView()
    attendance = AttendanceView()
    registrations: list[tuple[str, bool]] = []
    selected_sessions: list[str] = []
    registration.start_requested.connect(
        lambda name, consent: registrations.append((name, consent))
    )
    attendance.start_requested.connect(selected_sessions.append)

    registration.name_input.setText("Ada")
    registration.consent_checkbox.setChecked(True)
    registration.start_button.click()
    attendance.set_sessions((("session-1", "Morning"),))
    attendance.start_button.click()

    assert registrations == [("Ada", True)]
    assert selected_sessions == ["session-1"]
