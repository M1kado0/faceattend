import numpy as np
import pytest

from faceattend.camera.worker import CameraCaptureError, CameraEvidenceWorker
from faceattend.vision.types import FrameEvidence


class _Capture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV API
        return self.opened

    def read(self):
        self.reads += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_camera_worker_forwards_frames_as_evidence_and_releases_capture() -> None:
    capture = _Capture()
    worker = CameraEvidenceWorker(capture_factory=lambda _index: capture)

    emitted = []
    records = worker.run(
        lambda frame: FrameEvidence(
            frame=frame,
            face_count=0,
            face=None,
            track_id=None,
            pose=None,
        ),
        max_frames=2,
        result_capacity=1,
        on_evidence=emitted.append,
    )

    assert [record.frame.sequence_id for record in records] == [2]
    assert [record.frame.sequence_id for record in emitted] == [1, 2]
    assert capture.released is True


def test_camera_worker_rejects_unavailable_camera() -> None:
    with pytest.raises(CameraCaptureError, match="could not open"):
        CameraEvidenceWorker(capture_factory=lambda _index: _Capture(False)).run(
            lambda _frame: pytest.fail("processor must not run"), max_frames=1
        )


def test_camera_worker_releases_on_read_and_processor_failure() -> None:
    class BrokenRead(_Capture):
        def read(self):
            return False, None

    for capture, processor, error in (
        (BrokenRead(), lambda frame: pytest.fail(str(frame)), CameraCaptureError),
        (_Capture(), lambda _frame: (_ for _ in ()).throw(ValueError("processor")), ValueError),
    ):
        with pytest.raises(error):
            CameraEvidenceWorker(capture_factory=lambda _index, c=capture: c).run(
                processor, max_frames=1
            )
        assert capture.released


def test_camera_worker_stop_from_callback_and_zero_retention() -> None:
    capture = _Capture()
    worker = CameraEvidenceWorker(capture_factory=lambda _index: capture)
    emitted = []

    def receive(item: FrameEvidence) -> None:
        emitted.append(item)
        worker.stop()

    records = worker.run(
        lambda frame: FrameEvidence(frame, 0, None, None, None),
        on_evidence=receive,
        result_capacity=0,
    )
    assert records == [] and len(emitted) == 1 and capture.reads == 1 and capture.released


def test_camera_worker_rejects_negative_retention_without_opening_camera() -> None:
    opened = False

    def capture_factory(_index: int) -> _Capture:
        nonlocal opened
        opened = True
        return _Capture()

    with pytest.raises(ValueError, match="non-negative"):
        CameraEvidenceWorker(capture_factory=capture_factory).run(
            lambda _frame: pytest.fail("processor must not run"), result_capacity=-1
        )
    assert opened is False
