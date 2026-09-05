"""Print MediaPipe-matrix versus calibrated solvePnP pose per webcam frame.

Run from the repository root:

    uv run python scripts/compare_head_pose_webcam.py

No frames or biometric images are written to disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from faceattend.vision.head_pose import (  # noqa: E402
    HeadPoseError,
    PoseComparison,
    compare_pose_estimators,
)


def _print_comparison(frame_number: int, comparison: PoseComparison) -> None:
    matrix = comparison.matrix_pose
    solvepnp = comparison.solvepnp_pose
    error = comparison.absolute_error_degrees
    print(
        f"frame={frame_number:03d} "
        f"matrix(yaw={matrix.yaw_degrees:7.2f}, "
        f"pitch={matrix.pitch_degrees:7.2f}, roll={matrix.roll_degrees:7.2f}) "
        f"solvepnp(yaw={solvepnp.yaw_degrees:7.2f}, "
        f"pitch={solvepnp.pitch_degrees:7.2f}, roll={solvepnp.roll_degrees:7.2f}) "
        f"abs_error(yaw={error.yaw_degrees:6.2f}, "
        f"pitch={error.pitch_degrees:6.2f}, roll={error.roll_degrees:6.2f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/face_landmarker_v2_with_blendshapes.task"),
        help="MediaPipe Face Landmarker task model.",
    )
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be positive")
    if not args.model.is_file():
        print(f"ERROR: MediaPipe model not found: {args.model}")
        return 1

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        print(f"ERROR: could not open camera {args.camera_index}")
        return 1

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=RunningMode.VIDEO,
        num_faces=2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    comparisons: list[PoseComparison] = []
    failures: dict[str, int] = {}

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print(f"Capturing {args.frames} frames from camera {args.camera_index}.")
            print("Keep one face visible; move slowly through neutral, turns, pitch, and roll.")
            for frame_number in range(1, args.frames + 1):
                ok, frame_bgr = camera.read()
                if not ok:
                    print(f"frame={frame_number:03d} status=camera_read_failed")
                    failures["camera_read_failed"] = failures.get("camera_read_failed", 0) + 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.monotonic_ns() // 1_000_000)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb),
                    timestamp_ms,
                )
                if len(result.face_landmarks) != 1:
                    reason = "no_face" if not result.face_landmarks else "multiple_faces"
                    print(f"frame={frame_number:03d} status={reason}")
                    failures[reason] = failures.get(reason, 0) + 1
                    continue

                matrices = getattr(result, "facial_transformation_matrixes", ())
                if not matrices:
                    print(f"frame={frame_number:03d} status=matrix_unavailable")
                    failures["matrix_unavailable"] = failures.get("matrix_unavailable", 0) + 1
                    continue

                try:
                    height, width = frame_rgb.shape[:2]
                    comparison = compare_pose_estimators(
                        matrices[0],
                        result.face_landmarks[0],
                        image_width=width,
                        image_height=height,
                    )
                except (HeadPoseError, IndexError, TypeError, ValueError) as exc:
                    print(f"frame={frame_number:03d} status=pose_error reason={exc}")
                    failures["pose_error"] = failures.get("pose_error", 0) + 1
                    continue

                comparisons.append(comparison)
                _print_comparison(frame_number, comparison)
    finally:
        camera.release()

    print(f"Compared frames: {len(comparisons)}/{args.frames}")
    if comparisons:
        errors = np.array(
            [
                [
                    item.absolute_error_degrees.yaw_degrees,
                    item.absolute_error_degrees.pitch_degrees,
                    item.absolute_error_degrees.roll_degrees,
                ]
                for item in comparisons
            ],
            dtype=np.float64,
        )
        print(
            "Mean absolute error degrees: "
            f"yaw={errors[:, 0].mean():.2f}, "
            f"pitch={errors[:, 1].mean():.2f}, "
            f"roll={errors[:, 2].mean():.2f}"
        )
    if failures:
        print(f"Skipped frames: {failures}")
    return 0 if comparisons else 1


if __name__ == "__main__":
    raise SystemExit(main())
