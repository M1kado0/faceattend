import numpy as np
import pytest

from faceattend.vision.evidence import DuplicateFrameDetector
from faceattend.vision.types import Frame, FrameEvidence


def _frame(value: int, timestamp: int = 1) -> Frame:
    return Frame(np.full((2, 2, 3), value, dtype=np.uint8), timestamp, value)


def test_duplicate_detector_flags_repeated_pixels_only_within_history() -> None:
    detector = DuplicateFrameDetector(history_size=2)

    assert detector.observe(_frame(1)) is False
    assert detector.observe(_frame(1, 2)) is True
    assert detector.observe(_frame(2, 3)) is False
    assert detector.observe(_frame(1, 4)) is True


def test_frame_evidence_rejects_inconsistent_face_count() -> None:
    with pytest.raises(ValueError, match="requires a face observation"):
        FrameEvidence(_frame(1), 1, None, "track", None)


def test_frame_evidence_rejects_invalid_signal_ranges() -> None:
    with pytest.raises(ValueError, match="lighting_score"):
        FrameEvidence(_frame(1), 0, None, None, None, lighting_score=2.0)
    with pytest.raises(ValueError, match="motion_score"):
        FrameEvidence(_frame(1), 0, None, None, None, motion_score=-1.0)
