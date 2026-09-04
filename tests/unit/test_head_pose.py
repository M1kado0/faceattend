"""Deterministic validation for the new headless pose boundary."""

import cv2
import numpy as np
import pytest

from faceattend.vision.continuity import (
    ContinuityStatus,
    FaceContinuityConfig,
    FaceContinuityTracker,
)
from faceattend.vision.head_pose import (
    CANONICAL_FACE_POINTS,
    CANONICAL_LANDMARK_INDICES,
    CanonicalSolvePnPHeadPoseEstimator,
    ExponentialPoseSmoother,
    HeadPoseError,
    PoseHysteresis,
    estimate_matrix_head_pose,
    estimate_solvepnp_head_pose,
)
from faceattend.vision.model_lifecycle import LazyModel
from faceattend.vision.types import BoundingBox, FaceObservation, FaceQuality, Frame, HeadPose


def _projected_landmarks(
    angles: tuple[float, float, float],
    *,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    camera = np.array(
        [[640.0, 0.0, width / 2], [0.0, 640.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_vector = np.deg2rad(np.array(angles, dtype=np.float64)).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        CANONICAL_FACE_POINTS,
        rotation_vector,
        np.array([[0.0], [0.0], [600.0]], dtype=np.float64),
        camera,
        None,
    )
    landmarks = np.zeros((468, 2), dtype=np.float32)
    for index, point in zip(CANONICAL_LANDMARK_INDICES, projected.reshape(-1, 2), strict=True):
        landmarks[index] = (point[0] / width, point[1] / height)
    return landmarks


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, -20.0, 0.0), (15.0, 0.0, 0.0), (0.0, 0.0, 15.0)],
)
def test_solvepnp_recovers_known_pose(angles: tuple[float, float, float]) -> None:
    estimated = estimate_solvepnp_head_pose(
        _projected_landmarks(angles), image_width=640, image_height=480
    )

    assert estimated.yaw_degrees == pytest.approx(angles[1], abs=0.2)
    assert estimated.pitch_degrees == pytest.approx(angles[0], abs=0.2)
    assert estimated.roll_degrees == pytest.approx(angles[2], abs=0.2)


def test_matrix_estimator_uses_degrees_without_arbitrary_scaling() -> None:
    rotation_vector = np.deg2rad(np.array([0.0, -18.0, 0.0])).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation

    pose = estimate_matrix_head_pose(transform)

    assert pose.pitch_degrees == pytest.approx(0.0, abs=0.01)
    assert pose.yaw_degrees == pytest.approx(-18.0, abs=0.01)
    assert pose.roll_degrees == pytest.approx(0.0, abs=0.01)


def test_pose_estimator_protocol_adapter_uses_frame_dimensions() -> None:
    frame = Frame(np.zeros((480, 640, 3), dtype=np.uint8), 1, 1)
    face = FaceObservation(
        bbox=BoundingBox(250.0, 150.0, 390.0, 330.0),
        detector_score=0.99,
        landmarks=_projected_landmarks((0.0, 10.0, 0.0)),
        quality=FaceQuality(True, 100.0, 0.5, 0.1),
    )

    pose = CanonicalSolvePnPHeadPoseEstimator().estimate(frame, face)

    assert pose.yaw_degrees == pytest.approx(10.0, abs=0.2)


def test_invalid_pose_inputs_raise_head_pose_error() -> None:
    with pytest.raises(HeadPoseError, match="incomplete"):
        estimate_solvepnp_head_pose(np.zeros((2, 2)), image_width=640, image_height=480)

    with pytest.raises(HeadPoseError, match="dimensions"):
        estimate_solvepnp_head_pose(np.zeros((468, 2)), image_width=0, image_height=480)

    invalid = _projected_landmarks((0.0, 0.0, 0.0))
    invalid[1, 0] = np.nan
    with pytest.raises(HeadPoseError, match="non-finite"):
        estimate_solvepnp_head_pose(invalid, image_width=640, image_height=480)


def test_pose_smoother_changes_rendered_value_but_not_raw_pose() -> None:
    smoother = ExponentialPoseSmoother(alpha=0.5)
    first = HeadPose(0.0, 0.0, 0.0)
    second = HeadPose(20.0, 10.0, -4.0)

    assert smoother.update(first) == first
    rendered = smoother.update(second)

    assert rendered == HeadPose(10.0, 5.0, -2.0)
    assert second == HeadPose(20.0, 10.0, -4.0)


def test_pose_hysteresis_has_separate_enter_and_exit_thresholds() -> None:
    hysteresis = PoseHysteresis(enter_degrees=15.0, exit_degrees=10.0)

    assert hysteresis.update(14.9) is False
    assert hysteresis.update(15.0) is True
    assert hysteresis.update(12.0) is True
    assert hysteresis.update(9.9) is False


def test_lazy_model_factory_runs_once() -> None:
    calls: list[int] = []
    loader = LazyModel(lambda: calls.append(1) or object())

    first = loader.get()
    second = loader.get()

    assert first is second
    assert calls == [1]


def _face(*, x: float = 40.0, track_id: str | None = "person") -> FaceObservation:
    return FaceObservation(
        bbox=BoundingBox(x, 40.0, x + 120.0, 200.0),
        detector_score=0.99,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        quality=FaceQuality(True, 100.0, 0.5, 0.1),
        track_id=track_id,
    )


def _frame(timestamp_ns: int, sequence_id: int = 1) -> Frame:
    return Frame(np.zeros((480, 640, 3), dtype=np.uint8), timestamp_ns, sequence_id)


def test_continuity_accepts_same_face_and_short_gap() -> None:
    tracker = FaceContinuityTracker(FaceContinuityConfig(max_gap_ms=400))

    assert tracker.observe(_frame(0), [_face()]).status is ContinuityStatus.ACCEPTED
    assert tracker.observe(_frame(100_000_000), []).status is ContinuityStatus.WAITING
    assert tracker.observe(_frame(200_000_000), [_face(x=43.0)]).status is ContinuityStatus.ACCEPTED


@pytest.mark.parametrize(
    ("faces", "reason"),
    [([_face(track_id="other")], "track_changed"), ([_face(), _face(x=250.0)], "multiple_faces")],
)
def test_continuity_fails_face_substitution_or_multiple_faces(
    faces: list[FaceObservation], reason: str
) -> None:
    tracker = FaceContinuityTracker()
    tracker.observe(_frame(0), [_face()])

    result = tracker.observe(_frame(100_000_000), faces)

    assert result.status is ContinuityStatus.FAILED
    assert result.reason == reason


def test_continuity_fails_long_gap_jump_and_non_monotonic_time() -> None:
    tracker = FaceContinuityTracker(FaceContinuityConfig(max_gap_ms=100))
    tracker.observe(_frame(0), [_face()])

    assert tracker.observe(_frame(200_000_000), []).reason == "face_missing_too_long"
    assert tracker.observe(_frame(300_000_000), [_face(x=500.0)]).reason == "implausible_face_jump"
    assert tracker.observe(_frame(300_000_000), [_face()]).reason == "timestamp_not_monotonic"
