"""Small headless camera worker that forwards frames to an evidence builder."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from faceattend.vision.types import Frame, FrameEvidence


class FrameEvidenceProcessor(Protocol):
    def __call__(self, frame: Frame) -> FrameEvidence: ...


class CameraCaptureError(RuntimeError):
    """Raised when a camera cannot be opened or stops producing frames."""


class CameraEvidenceWorker:
    """Own OpenCV capture and emit one shared evidence record per frame.

    The processor performs CV inference outside this class. This boundary is
    deliberately usable by a future Qt worker without putting Qt in the CV
    package.
    """

    def __init__(
        self,
        camera_index: int = 0,
        *,
        capture_factory: Callable[[int], cv2.VideoCapture] = cv2.VideoCapture,
    ) -> None:
        self.camera_index = camera_index
        self._capture_factory = capture_factory
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def run(
        self,
        processor: FrameEvidenceProcessor,
        *,
        max_frames: int | None = None,
        result_capacity: int = 1,
        on_evidence: Callable[[FrameEvidence], None] | None = None,
    ) -> list[FrameEvidence]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if result_capacity < 0:
            raise ValueError("result_capacity must be non-negative")
        capture = self._capture_factory(self.camera_index)
        if not capture.isOpened():
            raise CameraCaptureError(f"could not open camera {self.camera_index}")

        self._stopped = False
        evidence: list[FrameEvidence] = []
        try:
            sequence_id = 0
            while not self._stopped and (max_frames is None or sequence_id < max_frames):
                ok, pixels = capture.read()
                if not ok:
                    raise CameraCaptureError("camera stopped returning frames")
                sequence_id += 1
                frame = Frame(
                    pixels=np.ascontiguousarray(pixels, dtype=np.uint8),
                    captured_at_ns=time.monotonic_ns(),
                    sequence_id=sequence_id,
                )
                item = processor(frame)
                if on_evidence is not None:
                    on_evidence(item)
                if result_capacity:
                    evidence.append(item)
                    if len(evidence) > result_capacity:
                        evidence.pop(0)
        finally:
            capture.release()
        return evidence
