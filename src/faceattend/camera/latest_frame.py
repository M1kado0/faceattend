"""Thread-safe capacity-one handoff between capture and inference."""

from __future__ import annotations

from threading import Lock

from faceattend.vision.types import Frame


class LatestFrameBuffer:
    """Keep only the newest valid frame so inference latency cannot accumulate."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._frame: Frame | None = None
        self._last_timestamp_ns: int | None = None
        self._last_sequence_id: int | None = None
        self.overwritten_count = 0
        self.rejected_count = 0
        self.expired_count = 0

    def offer(self, frame: Frame) -> bool:
        with self._lock:
            if (
                frame.captured_at_ns < 0
                or frame.sequence_id <= 0
                or (
                    self._last_timestamp_ns is not None
                    and frame.captured_at_ns <= self._last_timestamp_ns
                )
                or (
                    self._last_sequence_id is not None
                    and frame.sequence_id <= self._last_sequence_id
                )
            ):
                self.rejected_count += 1
                return False
            if self._frame is not None:
                self.overwritten_count += 1
            self._frame = frame
            self._last_timestamp_ns = frame.captured_at_ns
            self._last_sequence_id = frame.sequence_id
            return True

    def take(self, *, now_ns: int, max_age_ns: int) -> Frame | None:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        with self._lock:
            frame, self._frame = self._frame, None
            if frame is None:
                return None
            if now_ns < frame.captured_at_ns or now_ns - frame.captured_at_ns > max_age_ns:
                self.expired_count += 1
                return None
            return frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None
