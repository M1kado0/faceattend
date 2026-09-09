"""Dumb Qt Widgets that render desktop runtime state."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from faceattend.application.runtime import DesktopMode, SessionPresentation
from faceattend.vision.types import Frame


class HomeView(QWidget):
    mode_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("FaceAttend")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        registration = QPushButton("Register a person")
        attendance = QPushButton("Check in")
        registration.clicked.connect(lambda: self.mode_requested.emit(DesktopMode.REGISTRATION))
        attendance.clicked.connect(lambda: self.mode_requested.emit(DesktopMode.ATTENDANCE))
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(registration)
        layout.addWidget(attendance)
        layout.addStretch()


class SessionView(QWidget):
    """Present session facts; contains no CV decisions or model inference."""

    cancel_requested = Signal()

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_label = QLabel(title)
        self.preview_label = QLabel("Camera preview unavailable")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.instruction_label = QLabel("Ready")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.content_layout = QVBoxLayout(self)
        self.content_layout.addWidget(self.title_label)
        self.content_layout.addWidget(self.preview_label, stretch=1)
        self.content_layout.addWidget(self.instruction_label)
        self.content_layout.addWidget(self.progress_bar)
        self.content_layout.addWidget(self.status_label)
        self.content_layout.addWidget(self.cancel_button)

    @Slot(object)
    def apply_presentation(self, presentation: object) -> None:
        if not isinstance(presentation, SessionPresentation):
            raise TypeError("presentation must be SessionPresentation")
        self.instruction_label.setText(presentation.instruction)
        self.progress_bar.setValue(presentation.progress_percent)
        status = presentation.failure_reason or presentation.status.value
        self.status_label.setText(status.replace("_", " "))

    @Slot(object)
    def show_frame(self, frame: object) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("preview must be a Frame")
        pixels = np.ascontiguousarray(frame.pixels)
        height, width, channels = pixels.shape
        if channels != 3:
            raise ValueError("preview frame must have three BGR channels")
        image = QImage(
            pixels.data,
            width,
            height,
            int(pixels.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        self.preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class RegistrationView(SessionView):
    start_requested = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__("Face registration")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Person name")
        self.consent_checkbox = QCheckBox("I have explicit consent to register this person")
        self.start_button = QPushButton("Start registration")
        self.start_button.clicked.connect(
            lambda: self.start_requested.emit(
                self.name_input.text(), self.consent_checkbox.isChecked()
            )
        )
        form = QFormLayout()
        form.addRow("Name", self.name_input)
        form.addRow(self.consent_checkbox)
        form.addRow(self.start_button)
        self.content_layout.insertLayout(1, form)


class AttendanceView(SessionView):
    start_requested = Signal(str)
    create_session_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__("Attendance check-in")
        self.session_choices = QComboBox()
        self.session_name_input = QLineEdit()
        self.session_name_input.setPlaceholderText("New attendance session name")
        self.create_button = QPushButton("Create session")
        self.start_button = QPushButton("Start check-in")
        self.create_button.clicked.connect(
            lambda: self.create_session_requested.emit(self.session_name_input.text())
        )
        self.start_button.clicked.connect(
            lambda: self.start_requested.emit(self.session_choices.currentData() or "")
        )
        form = QFormLayout()
        form.addRow("Attendance session", self.session_choices)
        row = QHBoxLayout()
        row.addWidget(self.session_name_input)
        row.addWidget(self.create_button)
        form.addRow("New session", row)
        form.addRow(self.start_button)
        self.content_layout.insertLayout(1, form)

    def set_sessions(self, sessions: tuple[tuple[str, str], ...]) -> None:
        self.session_choices.clear()
        for session_id, name in sessions:
            self.session_choices.addItem(name, session_id)
