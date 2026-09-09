"""Qt worker objects that exclusively own capture and inference resources."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from faceattend.application.runtime import RuntimeStatus, SessionProcessor
from faceattend.camera.latest_frame import LatestFrameBuffer
from faceattend.vision.types import Frame


class VideoCapture(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...


class CameraWorker(QObject):
    """Create, read, and release one OpenCV capture in one Qt thread."""

    preview_ready = Signal(object)
    started = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        frame_buffer: LatestFrameBuffer,
        *,
        camera_index: int,
        capture_factory: Callable[[int], VideoCapture],
        capture_interval_ms: int,
        preview_interval_ms: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        if capture_interval_ms <= 0 or preview_interval_ms <= 0:
            raise ValueError("camera intervals must be positive")
        self.frame_buffer = frame_buffer
        self.camera_index = camera_index
        self.capture_factory = capture_factory
        self.capture_interval_ms = capture_interval_ms
        self.preview_interval_ns = preview_interval_ms * 1_000_000
        self.clock_ns = clock_ns
        self._capture: VideoCapture | None = None
        self._timer: QTimer | None = None
        self._sequence_id = 0
        self._last_preview_ns: int | None = None
        self._finished = False

    @Slot()
    def start(self) -> None:
        try:
            capture = self.capture_factory(self.camera_index)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"camera {self.camera_index} is unavailable")
            self._capture = capture
            self._timer = QTimer(self)
            self._timer.setInterval(self.capture_interval_ms)
            self._timer.timeout.connect(self.capture_once)
            self._timer.start()
            self.started.emit(threading.get_ident())
        except (OSError, RuntimeError, cv2.error) as exc:
            self.failed.emit(str(exc))
            self._finish()

    @Slot()
    def capture_once(self) -> None:
        if self._capture is None or self._finished:
            return
        try:
            ok, pixels = self._capture.read()
            if not ok or pixels is None:
                raise RuntimeError("camera stopped returning frames")
            now_ns = self.clock_ns()
            self._sequence_id += 1
            frame = Frame(
                np.ascontiguousarray(pixels, dtype=np.uint8),
                now_ns,
                self._sequence_id,
            )
            self.frame_buffer.offer(frame)
            if (
                self._last_preview_ns is None
                or now_ns - self._last_preview_ns >= self.preview_interval_ns
            ):
                self._last_preview_ns = now_ns
                self.preview_ready.emit(frame)
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            self.failed.emit(str(exc))
            self._finish()

    @Slot()
    def stop(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.stopped.emit()


class InferenceWorker(QObject):
    """Construct and reuse one synchronous headless processor in one Qt thread."""

    evidence_ready = Signal(object)
    presentation_ready = Signal(object)
    started = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        frame_buffer: LatestFrameBuffer,
        processor_factory: Callable[[], SessionProcessor],
        *,
        inference_interval_ms: int,
        max_frame_age_ms: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        if inference_interval_ms <= 0 or max_frame_age_ms <= 0:
            raise ValueError("inference interval and frame age must be positive")
        self.frame_buffer = frame_buffer
        self.processor_factory = processor_factory
        self.inference_interval_ms = inference_interval_ms
        self.max_frame_age_ns = max_frame_age_ms * 1_000_000
        self.clock_ns = clock_ns
        self._processor: SessionProcessor | None = None
        self._timer: QTimer | None = None
        self._finished = False

    @Slot()
    def start(self) -> None:
        try:
            self._processor = self.processor_factory()
            self._timer = QTimer(self)
            self._timer.setInterval(self.inference_interval_ms)
            self._timer.timeout.connect(self.infer_once)
            self._timer.start()
            self.started.emit(threading.get_ident())
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            self.failed.emit(str(exc))
            self._finish()

    @Slot()
    def infer_once(self) -> None:
        if self._processor is None or self._finished:
            return
        frame = self.frame_buffer.take(now_ns=self.clock_ns(), max_age_ns=self.max_frame_age_ns)
        if frame is None:
            return
        try:
            evidence = self._processor(frame)
            presentation = self._processor.presentation(evidence)
            self.evidence_ready.emit(evidence)
            self.presentation_ready.emit(presentation)
            if presentation.status in {
                RuntimeStatus.COMPLETED,
                RuntimeStatus.FAILED,
                RuntimeStatus.CANCELLED,
            }:
                self._finish()
        except (OSError, RuntimeError, ValueError, TypeError, cv2.error) as exc:
            self.failed.emit(str(exc))
            self._finish()

    @Slot()
    def cancel(self) -> None:
        if self._processor is not None and not self._finished:
            self._processor.cancel()
        self._finish()

    @Slot()
    def stop(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._processor is not None:
            self._processor.close()
            self._processor = None
        self.frame_buffer.clear()
        self.stopped.emit()
