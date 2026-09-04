"""Head-pose helpers for MediaPipe face landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HEAD_POSE_LANDMARKS = (1, 33, 61, 199, 263, 291)


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float


def estimate_head_pose(
    face_landmarks: list,
    *,
    image_width: int,
    image_height: int,
) -> HeadPose | None:
    """Estimate pitch/yaw/roll from MediaPipe landmarks with solvePnP."""
    face_2d: list[list[float]] = []
    face_3d: list[list[float]] = []

    for index in HEAD_POSE_LANDMARKS:
        landmark = face_landmarks[index]
        x = landmark.x * image_width
        y = landmark.y * image_height
        z = landmark.z * 3000 if index == 1 else landmark.z
        face_2d.append([x, y])
        face_3d.append([x, y, z])

    focal_length = float(image_width)
    camera_matrix = np.array(
        [
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        np.array(face_3d, dtype=np.float64),
        np.array(face_2d, dtype=np.float64),
        camera_matrix,
        distortion,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    return HeadPose(
        pitch=float(angles[0] * 360),
        yaw=float(angles[1] * 360),
        roll=float(angles[2] * 360),
    )
