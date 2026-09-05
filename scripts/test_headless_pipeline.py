"""Smoke-test the headless detector, liveness, alignment, and embedder.

Image mode:
    uv run python scripts/test_headless_pipeline.py IMAGE \
        --embedding-model models/w600k_r50.onnx

Webcam mode:
    uv run python scripts/test_headless_pipeline.py --camera \
        --embedding-model models/w600k_r50.onnx

Diagnostic images are saved to a temporary directory by default. Use
``--save-dir`` to choose a persistent location explicitly.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from faceattend.vision.alignment import AlignmentError, InsightFaceAligner  # noqa: E402
from faceattend.vision.embeddings import EmbeddingError, InsightFaceEmbedder  # noqa: E402
from faceattend.vision.face_detector import (  # noqa: E402
    FaceDetectionError,
    InsightFaceDetector,
)
from faceattend.vision.passive_liveness import (  # noqa: E402
    MiniFASNetPassiveLivenessDetector,
    PassiveLivenessError,
)
from faceattend.vision.types import FaceObservation, Frame  # noqa: E402


def _show_image(title: str, image: np.ndarray) -> None:
    """Display one diagnostic image and wait until a key is pressed."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def _save_image(directory: Path, filename: str, image: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / filename), image):
        raise OSError(f"could not save diagnostic image: {directory / filename}")


def _capture_camera(camera_index: int, frame_count: int) -> list[Frame]:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise OSError(f"could not open camera {camera_index}")

    print(f"Capturing {frame_count} webcam frames; keep one face in view.")
    frames: list[Frame] = []
    try:
        for sequence_id in range(1, frame_count + 1):
            ok, image = capture.read()
            if not ok:
                raise OSError("camera stopped returning frames")
            frames.append(
                Frame(
                    pixels=np.ascontiguousarray(image, dtype=np.uint8),
                    captured_at_ns=time.monotonic_ns(),
                    sequence_id=sequence_id,
                )
            )
    finally:
        capture.release()
    return frames


def _load_image(path: Path) -> list[Frame]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return [
        Frame(
            pixels=np.ascontiguousarray(image, dtype=np.uint8),
            captured_at_ns=time.monotonic_ns(),
            sequence_id=1,
        )
    ]


def _draw_detection(image: np.ndarray, face: FaceObservation) -> np.ndarray:
    detected = image.copy()
    box = face.bbox
    cv2.rectangle(
        detected,
        (round(box.x_min), round(box.y_min)),
        (round(box.x_max), round(box.y_max)),
        (0, 255, 0),
        2,
    )
    for x, y in face.landmarks:
        cv2.circle(detected, (round(float(x)), round(float(y))), 3, (0, 255, 255), -1)
    return detected


def main() -> int:
    parser = argparse.ArgumentParser(description="Test FaceAttend's headless CV pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("image", type=Path, nargs="?", help="Image containing one face.")
    source.add_argument("--camera", action="store_true", help="Capture a short webcam window.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of webcam frames used for passive liveness.",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=Path("models/det_10g.onnx"),
        help="Path to the InsightFace detector ONNX model.",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        required=True,
        help="Path to the InsightFace recognition ONNX model.",
    )
    parser.add_argument(
        "--passive-model",
        type=Path,
        default=Path("models/MiniFASNetV2.onnx"),
        help="Path to the MiniFASNetV2 ONNX model.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show original, detection, and aligned images in OpenCV windows.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Save diagnostic images here instead of a temporary directory.",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be positive")
    for model_path, label in (
        (args.detector_model, "detector"),
        (args.embedding_model, "embedding"),
        (args.passive_model, "passive-liveness"),
    ):
        if not model_path.is_file():
            print(f"ERROR: {label} model not found: {model_path}")
            return 1

    debug_dir = args.save_dir or Path(tempfile.mkdtemp(prefix="faceattend-debug-"))
    print(f"Diagnostic images: {debug_dir}")

    try:
        frames = (
            _capture_camera(args.camera_index, args.frames)
            if args.camera
            else _load_image(args.image)
        )
        detector = InsightFaceDetector(model_path=str(args.detector_model))
        faces_by_frame = [detector.detect(frame) for frame in frames]
        print(f"Detected faces per frame: {[len(faces) for faces in faces_by_frame]}")
        if any(len(faces) != 1 for faces in faces_by_frame):
            print("ERROR: expected exactly one face in every frame")
            return 1

        faces = [face_list[0] for face_list in faces_by_frame]
        last_frame = frames[-1]
        last_face = faces[-1]
        print(f"Final detector score: {last_face.detector_score:.4f}")
        print(f"Final landmarks: {last_face.landmarks.shape}")

        passive = MiniFASNetPassiveLivenessDetector(model_path=str(args.passive_model))
        evidence = passive.evaluate(frames, faces)
        print(f"Passive liveness: {evidence.decision.value}")
        print(f"Passive score: {evidence.score:.6f}")
        print(f"Passive threshold: {evidence.threshold:.6f}")
        print(f"Passive reason: {evidence.reason}")

        original = last_frame.pixels
        _save_image(debug_dir, "01_original.png", original)
        if args.show:
            _show_image("1 - Original", original)

        detected = _draw_detection(original, last_face)
        _save_image(debug_dir, "02_detected.png", detected)
        if args.show:
            _show_image("2 - Detected face and landmarks", detected)

        aligned = InsightFaceAligner().align(last_frame, last_face)
        print(f"Aligned image: shape={aligned.shape}, dtype={aligned.dtype}")
        _save_image(debug_dir, "03_aligned.png", aligned)
        if args.show:
            _show_image("3 - Aligned face", aligned)

        embedding = InsightFaceEmbedder(model_path=str(args.embedding_model)).embed(aligned)
        print(f"Embedding: dimensions={embedding.size}, dtype={embedding.dtype}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    except (
        AlignmentError,
        EmbeddingError,
        FaceDetectionError,
        OSError,
        PassiveLivenessError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("COMPLETED: camera/image → detection → passive liveness → alignment → embedding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
