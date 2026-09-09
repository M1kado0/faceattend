"""Offscreen integration tests for Qt camera and inference ownership."""

from __future__ import annotations

import os
import threading
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.gui.runtime import DesktopRuntime
from faceattend.vision.types import FrameEvidence

_APP = QApplication.instance() or QApplication([])


class _Capture:
    def __init__(self) -> None:
        self.owner = threading.get_ident()
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self):
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    def release(self) -> None:
        assert threading.get_ident() == self.owner
        self.released = True


class _UnavailableCapture(_Capture):
    def isOpened(self) -> bool:  # noqa: N802
        return False


class _Processor:
    def __init__(self, mode: DesktopMode) -> None:
        self.mode = mode
        self.owner = threading.get_ident()
        self.closed = False
        self.cancelled = False

    def __call__(self, frame):
        assert threading.get_ident() == self.owner
        return FrameEvidence(frame, 0, None, None, None, failure_reason="no_face")

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        return SessionPresentation(
            self.mode,
            RuntimeStatus.RUNNING,
            "TURN LEFT",
            0,
            2,
            evidence,
            evidence.failure_reason,
        )

    def cancel(self) -> None:
        assert threading.get_ident() == self.owner
        self.cancelled = True

    def close(self) -> None:
        assert threading.get_ident() == self.owner
        self.closed = True


class _CompletingProcessor(_Processor):
    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        return SessionPresentation(
            self.mode,
            RuntimeStatus.COMPLETED,
            "COMPLETED",
            2,
            2,
            evidence,
        )


class _FailingProcessor(_Processor):
    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        return SessionPresentation(
            self.mode,
            RuntimeStatus.FAILED,
            "RETRY",
            0,
            2,
            evidence,
            "liveness_score_below_threshold",
        )


class _SlowProcessor(_Processor):
    def __call__(self, frame):
        time.sleep(0.1)
        return super().__call__(frame)


def _wait_until(app: QApplication, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


def test_runtime_separates_owners_drops_backlog_and_restarts_cleanly() -> None:
    app = _APP
    captures: list[_Capture] = []
    processors: list[_Processor] = []

    def capture_factory(_index: int) -> _Capture:
        capture = _Capture()
        captures.append(capture)
        return capture

    def processor_factory(mode: DesktopMode) -> _Processor:
        processor = _Processor(mode)
        processors.append(processor)
        return processor

    runtime = DesktopRuntime(
        processor_factory,
        capture_factory=capture_factory,
        capture_interval_ms=2,
        preview_interval_ms=10,
        inference_interval_ms=25,
    )
    evidence: list[FrameEvidence] = []
    previews = []
    runtime.evidence_ready.connect(evidence.append)
    runtime.preview_ready.connect(previews.append)

    main_owner = threading.get_ident()
    for mode in (DesktopMode.REGISTRATION, DesktopMode.ATTENDANCE):
        previous_count = len(evidence)
        runtime.start(mode)
        _wait_until(
            app,
            lambda previous_count=previous_count: (
                runtime.lifecycle.status is RuntimeStatus.RUNNING and len(evidence) > previous_count
            ),
        )
        assert runtime.camera_owner_thread_id not in {None, main_owner}
        assert runtime.inference_owner_thread_id not in {None, main_owner}
        assert runtime.camera_owner_thread_id != runtime.inference_owner_thread_id
        assert len(previews) > 0
        assert runtime.frame_buffer.overwritten_count > 0
        runtime.cancel()
        assert runtime.lifecycle.status is RuntimeStatus.CANCELLED
        assert captures[-1].released
        assert processors[-1].cancelled and processors[-1].closed

    runtime.shutdown()
    assert runtime.lifecycle.status is RuntimeStatus.SHUTDOWN


def test_runtime_reports_camera_and_model_startup_failures() -> None:
    app = _APP

    no_camera = DesktopRuntime(
        lambda mode: _Processor(mode),
        capture_factory=lambda _index: _UnavailableCapture(),
    )
    no_camera.start(DesktopMode.ATTENDANCE)
    _wait_until(app, lambda: no_camera.lifecycle.status is RuntimeStatus.NO_CAMERA)
    no_camera.stop()

    def missing_model(_mode: DesktopMode) -> _Processor:
        raise RuntimeError("configured model is missing")

    no_model = DesktopRuntime(missing_model, capture_factory=lambda _index: _Capture())
    no_model.start(DesktopMode.REGISTRATION)
    _wait_until(app, lambda: no_model.lifecycle.status is RuntimeStatus.NO_MODEL)
    no_model.stop()


def test_terminal_session_stops_camera_and_inference_workers() -> None:
    app = _APP
    captures: list[_Capture] = []
    processors: list[_CompletingProcessor] = []

    def capture_factory(_index: int) -> _Capture:
        capture = _Capture()
        captures.append(capture)
        return capture

    def processor_factory(mode: DesktopMode) -> _CompletingProcessor:
        processor = _CompletingProcessor(mode)
        processors.append(processor)
        return processor

    runtime = DesktopRuntime(
        processor_factory,
        capture_factory=capture_factory,
        capture_interval_ms=2,
        inference_interval_ms=10,
    )
    runtime.start(DesktopMode.REGISTRATION)

    _wait_until(app, lambda: runtime.lifecycle.status is RuntimeStatus.COMPLETED)
    runtime.stop()
    assert captures[0].released
    assert processors[0].closed


def test_failed_attempt_releases_resources_before_a_fresh_retry() -> None:
    app = _APP
    captures: list[_Capture] = []
    processors: list[_Processor] = []

    def capture_factory(_index: int) -> _Capture:
        capture = _Capture()
        captures.append(capture)
        return capture

    def processor_factory(mode: DesktopMode) -> _Processor:
        processor: _Processor = (
            _FailingProcessor(mode) if not processors else _CompletingProcessor(mode)
        )
        processors.append(processor)
        return processor

    runtime = DesktopRuntime(
        processor_factory,
        capture_factory=capture_factory,
        capture_interval_ms=2,
        inference_interval_ms=10,
    )

    runtime.start(DesktopMode.REGISTRATION)
    _wait_until(app, lambda: runtime.lifecycle.status is RuntimeStatus.FAILED)
    _wait_until(app, lambda: captures[0].released and processors[0].closed)

    runtime.start(DesktopMode.REGISTRATION)
    _wait_until(app, lambda: runtime.lifecycle.status is RuntimeStatus.COMPLETED)
    _wait_until(app, lambda: captures[1].released and processors[1].closed)

    assert runtime.frame_buffer.take(now_ns=time.monotonic_ns(), max_age_ns=1) is None


def test_slow_inference_does_not_block_the_gui_event_loop() -> None:
    app = _APP
    runtime = DesktopRuntime(
        lambda mode: _SlowProcessor(mode),
        capture_factory=lambda _index: _Capture(),
        capture_interval_ms=2,
        inference_interval_ms=5,
    )
    evidence: list[FrameEvidence] = []
    heartbeat_count = 0

    def heartbeat() -> None:
        nonlocal heartbeat_count
        heartbeat_count += 1

    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(heartbeat)
    runtime.evidence_ready.connect(evidence.append)
    timer.start()
    runtime.start(DesktopMode.ATTENDANCE)
    _wait_until(app, lambda: len(evidence) == 1)
    timer.stop()
    runtime.cancel()

    assert heartbeat_count >= 5
