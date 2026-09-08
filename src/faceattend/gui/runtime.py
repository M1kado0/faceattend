"""Thread orchestration for the local PySide6 runtime."""

from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from faceattend.application.runtime import (
    DesktopLifecycle,
    DesktopMode,
    RuntimeStatus,
    SessionPresentation,
    SessionProcessor,
)
from faceattend.camera.latest_frame import LatestFrameBuffer
from faceattend.gui.workers import CameraWorker, InferenceWorker, VideoCapture


class DesktopRuntime(QObject):
    """Own two QThreads and expose queued signals to the main-thread GUI."""

    preview_ready = Signal(object)
    evidence_ready = Signal(object)
    presentation_ready = Signal(object)
    status_changed = Signal(object)
    error = Signal(str)
    _stop_workers = Signal()
    _cancel_workers = Signal()

    def __init__(
        self,
        processor_factory: Callable[[DesktopMode], SessionProcessor],
        *,
        camera_index: int = 0,
        capture_factory: Callable[[int], VideoCapture] = cv2.VideoCapture,
        capture_interval_ms: int = 15,
        preview_interval_ms: int = 33,
        inference_interval_ms: int = 100,
        max_frame_age_ms: int = 500,
    ) -> None:
        super().__init__()
        self.processor_factory = processor_factory
        self.camera_index = camera_index
        self.capture_factory = capture_factory
        self.capture_interval_ms = capture_interval_ms
        self.preview_interval_ms = preview_interval_ms
        self.inference_interval_ms = inference_interval_ms
        self.max_frame_age_ms = max_frame_age_ms
        self.lifecycle = DesktopLifecycle()
        self.frame_buffer = LatestFrameBuffer()
        self.camera_owner_thread_id: int | None = None
        self.inference_owner_thread_id: int | None = None
        self._camera_thread: QThread | None = None
        self._inference_thread: QThread | None = None
        self._camera_worker: CameraWorker | None = None
        self._inference_worker: InferenceWorker | None = None
        self._started_workers: set[str] = set()

    def start(self, mode: DesktopMode) -> None:
        self._discard_finished_threads()
        self.lifecycle.start(mode)
        self.status_changed.emit(self.lifecycle.status)
        self.frame_buffer = LatestFrameBuffer()
        self._started_workers.clear()
        self.camera_owner_thread_id = None
        self.inference_owner_thread_id = None

        camera_thread = QThread(self)
        inference_thread = QThread(self)
        camera_worker = CameraWorker(
            self.frame_buffer,
            camera_index=self.camera_index,
            capture_factory=self.capture_factory,
            capture_interval_ms=self.capture_interval_ms,
            preview_interval_ms=self.preview_interval_ms,
        )
        inference_worker = InferenceWorker(
            self.frame_buffer,
            lambda: self.processor_factory(mode),
            inference_interval_ms=self.inference_interval_ms,
            max_frame_age_ms=self.max_frame_age_ms,
        )
        camera_worker.moveToThread(camera_thread)
        inference_worker.moveToThread(inference_thread)

        camera_thread.started.connect(camera_worker.start)
        inference_thread.started.connect(inference_worker.start)
        self._stop_workers.connect(camera_worker.stop)
        self._stop_workers.connect(inference_worker.stop)
        self._cancel_workers.connect(camera_worker.stop)
        self._cancel_workers.connect(inference_worker.cancel)
        camera_worker.started.connect(self._camera_started)
        inference_worker.started.connect(self._inference_started)
        camera_worker.preview_ready.connect(self.preview_ready)
        inference_worker.evidence_ready.connect(self.evidence_ready)
        inference_worker.presentation_ready.connect(self._presentation_received)
        camera_worker.failed.connect(self._camera_failed)
        inference_worker.failed.connect(self._model_failed)
        # QThread itself belongs to the creating (GUI) thread. A queued quit
        # would deadlock when shutdown waits for the worker, so call the
        # documented thread-safe quit method directly from the worker signal.
        camera_worker.stopped.connect(camera_thread.quit, Qt.ConnectionType.DirectConnection)
        inference_worker.stopped.connect(inference_thread.quit, Qt.ConnectionType.DirectConnection)
        camera_thread.finished.connect(camera_worker.deleteLater)
        inference_thread.finished.connect(inference_worker.deleteLater)

        self._camera_thread = camera_thread
        self._inference_thread = inference_thread
        self._camera_worker = camera_worker
        self._inference_worker = inference_worker
        camera_thread.start()
        inference_thread.start()

    @Slot(object)
    def _camera_started(self, owner_id: object) -> None:
        if not isinstance(owner_id, int):
            raise TypeError("camera owner identifier must be an integer")
        self.camera_owner_thread_id = owner_id
        self._worker_started("camera")

    @Slot(object)
    def _inference_started(self, owner_id: object) -> None:
        if not isinstance(owner_id, int):
            raise TypeError("inference owner identifier must be an integer")
        self.inference_owner_thread_id = owner_id
        self._worker_started("inference")

    def _worker_started(self, name: str) -> None:
        self._started_workers.add(name)
        if (
            self._started_workers == {"camera", "inference"}
            and self.lifecycle.status is RuntimeStatus.STARTING
        ):
            self.lifecycle.mark_running()
            self.status_changed.emit(self.lifecycle.status)

    @Slot(object)
    def _presentation_received(self, presentation: object) -> None:
        if not isinstance(presentation, SessionPresentation):
            self.lifecycle.fail("invalid session presentation")
            self.status_changed.emit(self.lifecycle.status)
            self._stop_workers.emit()
            return
        self.presentation_ready.emit(presentation)
        if presentation.status is RuntimeStatus.COMPLETED:
            if self.lifecycle.status is RuntimeStatus.STARTING:
                self.lifecycle.mark_running()
            if self.lifecycle.status is RuntimeStatus.RUNNING:
                self.lifecycle.complete()
        elif presentation.status is RuntimeStatus.FAILED:
            self.lifecycle.fail(presentation.failure_reason or "session failed")
        elif presentation.status is RuntimeStatus.CANCELLED:
            self.lifecycle.cancel()
        else:
            return
        self.status_changed.emit(self.lifecycle.status)
        self._stop_workers.emit()

    @Slot(str)
    def _camera_failed(self, reason: str) -> None:
        self.lifecycle.fail_no_camera(reason)
        self.error.emit(reason)
        self.status_changed.emit(self.lifecycle.status)
        self._stop_workers.emit()

    @Slot(str)
    def _model_failed(self, reason: str) -> None:
        self.lifecycle.fail_no_model(reason)
        self.error.emit(reason)
        self.status_changed.emit(self.lifecycle.status)
        self._stop_workers.emit()

    def stop(self, *, timeout_ms: int = 3_000) -> None:
        self._stop_workers.emit()
        self._wait_for_threads(timeout_ms)

    def cancel(self, *, timeout_ms: int = 3_000) -> None:
        self.lifecycle.cancel()
        self.status_changed.emit(self.lifecycle.status)
        self._cancel_workers.emit()
        self._wait_for_threads(timeout_ms)

    def shutdown(self, *, timeout_ms: int = 3_000) -> None:
        self._stop_workers.emit()
        self._wait_for_threads(timeout_ms)
        self.lifecycle.shutdown()
        self.status_changed.emit(self.lifecycle.status)

    def _wait_for_threads(self, timeout_ms: int) -> None:
        for thread in (self._camera_thread, self._inference_thread):
            if thread is not None and thread.isRunning() and not thread.wait(timeout_ms):
                raise RuntimeError("desktop worker did not stop cleanly")
        self._discard_finished_threads()

    def _discard_finished_threads(self) -> None:
        for thread in (self._camera_thread, self._inference_thread):
            if thread is not None and thread.isRunning():
                raise RuntimeError("previous desktop workers are still active")
        self._camera_thread = None
        self._inference_thread = None
        self._camera_worker = None
        self._inference_worker = None
