"""Validated, framework-independent head-pose estimators.

The legacy ``ml.liveness.head_pose`` module remains in place for compatibility.
New workflows should use these estimators and their explicit conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from faceattend.vision.types import FaceObservation, Frame, HeadPose

Float64Array = NDArray[np.float64]

# MediaPipe landmark indices: nose tip, image-left eye, image-right eye,
# image-left mouth, image-right mouth, chin. The canonical model uses the
# image-coordinate convention: image x increases right and image y increases
# down. The canonical object model uses the conventional +y-up face frame;
# its projection therefore places eyes above the mouth and chin.
CANONICAL_LANDMARK_INDICES = (1, 33, 263, 61, 291, 199)
CANONICAL_FACE_POINTS: Float64Array = np.array(
    [
        (0.0, 0.0, 0.0),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
        (0.0, -63.6, -12.5),
    ],
    dtype=np.float64,
)


class HeadPoseError(ValueError):
    """Raised when landmarks or a transformation matrix cannot yield a pose."""


@dataclass(frozen=True, slots=True)
class PoseComparison:
    """Side-by-side pose estimates used for validation diagnostics."""

    matrix_pose: HeadPose
    solvepnp_pose: HeadPose

    @property
    def absolute_error_degrees(self) -> HeadPose:
        return HeadPose(
            yaw_degrees=abs(self.matrix_pose.yaw_degrees - self.solvepnp_pose.yaw_degrees),
            pitch_degrees=abs(self.matrix_pose.pitch_degrees - self.solvepnp_pose.pitch_degrees),
            roll_degrees=abs(self.matrix_pose.roll_degrees - self.solvepnp_pose.roll_degrees),
        )


def _landmark_xy(landmark: Any, width: int, height: int) -> tuple[float, float]:
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x) * width, float(landmark.y) * height
    values = np.asarray(landmark, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise HeadPoseError("landmark must contain x and y")
    # Array landmarks are accepted in either normalized or pixel coordinates.
    x, y = float(values[0]), float(values[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def _rotation_to_pose(rotation_matrix: Float64Array) -> HeadPose:
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    angles, *_ = cv2.RQDecomp3x3(matrix)
    # RQDecomp3x3 already returns degrees. No arbitrary scaling is valid here.
    return HeadPose(
        yaw_degrees=float(angles[1]),
        pitch_degrees=float(angles[0]),
        roll_degrees=float(angles[2]),
    )


def _camera_matrix(width: int, height: int, focal_length: float | None) -> Float64Array:
    focal = float(focal_length if focal_length is not None else width)
    if focal <= 0:
        raise HeadPoseError("focal length must be positive")
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_solvepnp_head_pose(
    landmarks: Any,
    *,
    image_width: int,
    image_height: int,
    camera_matrix: Float64Array | None = None,
    distortion: Float64Array | None = None,
) -> HeadPose:
    """Estimate pose from six landmarks and a canonical 3D face model.

    Angles are returned in degrees as ``(yaw, pitch, roll)``. Positive yaw is
    rotation toward image-right in this camera convention. Camera intrinsics
    should be calibrated for production; image-size/focal-length approximation
    is provided only as a documented fallback for a monocular webcam.
    """
    if image_width <= 0 or image_height <= 0:
        raise HeadPoseError("image dimensions must be positive")
    try:
        points_2d = np.array(
            [
                _landmark_xy(landmarks[index], image_width, image_height)
                for index in CANONICAL_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise HeadPoseError("incomplete or invalid landmarks") from exc
    if not np.isfinite(points_2d).all():
        raise HeadPoseError("landmarks contain non-finite values")

    intrinsics = (
        camera_matrix
        if camera_matrix is not None
        else _camera_matrix(image_width, image_height, None)
    )
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise HeadPoseError("camera matrix must be a finite 3x3 matrix")
    distortion_coefficients = (
        np.zeros((4, 1), dtype=np.float64) if distortion is None else distortion
    )
    initial_rotation = np.zeros((3, 1), dtype=np.float64)
    initial_translation = np.array(
        [[0.0], [0.0], [max(float(intrinsics[0, 0]), float(intrinsics[1, 1]))]],
        dtype=np.float64,
    )
    success, rotation_vector, _translation = cv2.solvePnP(
        CANONICAL_FACE_POINTS,
        points_2d,
        intrinsics,
        distortion_coefficients,
        rvec=initial_rotation,
        tvec=initial_translation,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise HeadPoseError("solvePnP failed")
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return _rotation_to_pose(np.asarray(rotation_matrix, dtype=np.float64))


def estimate_matrix_head_pose(matrix: Float64Array) -> HeadPose:
    """Extract the same degree/sign convention from a 4x4 face transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape == (4, 4):
        rotation = transform[:3, :3]
    elif transform.shape == (3, 3):
        rotation = transform
    else:
        raise HeadPoseError("facial transformation matrix must be 3x3 or 4x4")
    if not np.isfinite(rotation).all():
        raise HeadPoseError("transformation matrix contains non-finite values")
    return _rotation_to_pose(rotation)


def compare_pose_estimators(
    matrix: Float64Array,
    landmarks: Any,
    *,
    image_width: int,
    image_height: int,
    camera_matrix: Float64Array | None = None,
    distortion: Float64Array | None = None,
) -> PoseComparison:
    """Compare MediaPipe's matrix pose with calibrated landmark ``solvePnP``."""
    return PoseComparison(
        matrix_pose=estimate_matrix_head_pose(matrix),
        solvepnp_pose=estimate_solvepnp_head_pose(
            landmarks,
            image_width=image_width,
            image_height=image_height,
            camera_matrix=camera_matrix,
            distortion=distortion,
        ),
    )


class MediaPipeMatrixHeadPoseEstimator:
    """Headless adapter for MediaPipe facial transformation matrices."""

    def estimate(self, matrix: Float64Array) -> HeadPose:
        return estimate_matrix_head_pose(matrix)


class CanonicalSolvePnPHeadPoseEstimator:
    """Headless adapter implementing the project's pose-estimator protocol."""

    def __init__(
        self,
        *,
        focal_length: float | None = None,
        camera_matrix: Float64Array | None = None,
        distortion: Float64Array | None = None,
    ) -> None:
        self._focal_length = focal_length
        self._camera_matrix = camera_matrix
        self._distortion = distortion

    def estimate(self, frame: Frame, face: FaceObservation) -> HeadPose:
        height, width = frame.pixels.shape[:2]
        matrix = self._camera_matrix
        if matrix is None:
            matrix = _camera_matrix(width, height, self._focal_length)
        return estimate_solvepnp_head_pose(
            face.landmarks,
            image_width=width,
            image_height=height,
            camera_matrix=matrix,
            distortion=self._distortion,
        )


@dataclass(slots=True)
class ExponentialPoseSmoother:
    """Smooth rendered/decision pose values without changing raw evidence."""

    alpha: float = 0.35
    _previous: HeadPose | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def update(self, pose: HeadPose) -> HeadPose:
        if self._previous is None:
            self._previous = pose
            return pose
        previous = self._previous
        smoothed = HeadPose(
            yaw_degrees=self.alpha * pose.yaw_degrees + (1 - self.alpha) * previous.yaw_degrees,
            pitch_degrees=self.alpha * pose.pitch_degrees
            + (1 - self.alpha) * previous.pitch_degrees,
            roll_degrees=self.alpha * pose.roll_degrees + (1 - self.alpha) * previous.roll_degrees,
        )
        self._previous = smoothed
        return smoothed


@dataclass(slots=True)
class PoseHysteresis:
    """Prevent pose-bin flicker with distinct enter and exit thresholds."""

    enter_degrees: float
    exit_degrees: float
    active: bool = False

    def __post_init__(self) -> None:
        if self.enter_degrees < self.exit_degrees:
            raise ValueError("enter threshold must be >= exit threshold")

    def update(self, absolute_angle_degrees: float) -> bool:
        magnitude = abs(float(absolute_angle_degrees))
        if self.active:
            if magnitude < self.exit_degrees:
                self.active = False
        elif magnitude >= self.enter_degrees:
            self.active = True
        return self.active
