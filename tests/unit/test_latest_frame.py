"""Behavior tests for capacity-one camera-to-inference backpressure."""

import numpy as np

from faceattend.camera.latest_frame import LatestFrameBuffer
from faceattend.vision.types import Frame


def _frame(sequence_id: int, timestamp_ns: int) -> Frame:
    return Frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp_ns, sequence_id)


def test_latest_frame_buffer_replaces_stale_work_instead_of_queueing() -> None:
    buffer = LatestFrameBuffer()

    assert buffer.offer(_frame(1, 100)) is True
    assert buffer.offer(_frame(2, 200)) is True

    assert buffer.take(now_ns=250, max_age_ns=100).sequence_id == 2
    assert buffer.take(now_ns=251, max_age_ns=100) is None
    assert buffer.overwritten_count == 1


def test_latest_frame_buffer_rejects_non_monotonic_and_expired_frames() -> None:
    buffer = LatestFrameBuffer()

    assert buffer.offer(_frame(1, 100)) is True
    assert buffer.offer(_frame(2, 100)) is False
    assert buffer.take(now_ns=250, max_age_ns=100) is None
    assert buffer.rejected_count == 1
    assert buffer.expired_count == 1
