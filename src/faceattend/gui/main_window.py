"""Primary Qt window for mode selection and session presentation."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from faceattend.application.runtime import DesktopMode, RuntimeStatus
from faceattend.gui.runtime import DesktopRuntime
from faceattend.gui.views import AttendanceView, HomeView, RegistrationView, SessionView


class MainWindow(QMainWindow):
    def __init__(self, runtime: DesktopRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.setWindowTitle("FaceAttend")
        self.resize(900, 700)
        self.stack = QStackedWidget()
        self.home = HomeView()
        self.registration = RegistrationView()
        self.attendance = AttendanceView()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.registration)
        self.stack.addWidget(self.attendance)
        self.setCentralWidget(self.stack)

        self.home.mode_requested.connect(self.start_mode)
        self.registration.cancel_requested.connect(self.cancel_session)
        self.attendance.cancel_requested.connect(self.cancel_session)
        self.runtime.preview_ready.connect(self._show_preview)
        self.runtime.presentation_ready.connect(self._show_presentation)
        self.runtime.status_changed.connect(self._show_runtime_status)
        self.runtime.error.connect(self._show_error)

    @Slot(object)
    def start_mode(self, mode: object) -> None:
        if not isinstance(mode, DesktopMode) or mode is DesktopMode.HOME:
            raise ValueError("invalid desktop mode")
        self.stack.setCurrentWidget(self._view_for(mode))
        self.runtime.start(mode)

    @Slot()
    def cancel_session(self) -> None:
        self.runtime.cancel()
        self.stack.setCurrentWidget(self.home)

    @Slot(object)
    def _show_preview(self, frame: object) -> None:
        self._current_session_view().show_frame(frame)

    @Slot(object)
    def _show_presentation(self, presentation: object) -> None:
        self._current_session_view().apply_presentation(presentation)

    @Slot(object)
    def _show_runtime_status(self, status: object) -> None:
        if status in {
            RuntimeStatus.CANCELLED,
            RuntimeStatus.COMPLETED,
            RuntimeStatus.NO_CAMERA,
            RuntimeStatus.NO_MODEL,
            RuntimeStatus.FAILED,
        }:
            view = self._current_session_view()
            view.status_label.setText(str(getattr(status, "value", status)).replace("_", " "))

    @Slot(str)
    def _show_error(self, reason: str) -> None:
        self._current_session_view().status_label.setText(reason)

    def _view_for(self, mode: DesktopMode) -> SessionView:
        return self.registration if mode is DesktopMode.REGISTRATION else self.attendance

    def _current_session_view(self) -> SessionView:
        current = self.stack.currentWidget()
        if isinstance(current, SessionView):
            return current
        return self._view_for(self.runtime.lifecycle.mode)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.runtime.shutdown()
        event.accept()
