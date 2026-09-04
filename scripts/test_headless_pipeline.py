"""Smoke-test the headless detector, aligner, and embedder pipeline.

Usage:
     uv run python scripts/test_headless_pipeline.py \
        images/11.jpg \
        --detector-model models/det_10g.onnx \
        --embedding-model models/w600k_r50.onnx

Diagnostic images are saved to a temporary directory by default. The detector
and recognition ONNX files must already be available locally.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from faceattend.vision.alignment import AlignmentError, InsightFaceAligner  # noqa: E402
from faceattend.vision.embeddings import EmbeddingError, InsightFaceEmbedder  # noqa: E402
from faceattend.vision.face_detector import InsightFaceDetector  # noqa: E402
from faceattend.vision.types import Frame  # noqa: E402


def _show_image(title: str, image: np.ndarray) -> None:
    """Display one diagnostic image and wait until a key is pressed."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def _save_image(directory: Path | None, filename: str, image: np.ndarray) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / filename), image):
        raise OSError(f"could not save diagnostic image: {directory / filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test FaceAttend's headless CV pipeline.")
    parser.add_argument("image", type=Path, help="Local image containing exactly one face.")
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
        help="Path to the InsightFace recognition ONNX model (for example w600k_r50.onnx).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the original, detected, and aligned images in OpenCV windows.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Save diagnostic images here instead of a temporary directory.",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: could not read image: {args.image}")
        return 1
    if not args.embedding_model.is_file():
        print(f"ERROR: embedding model not found: {args.embedding_model}")
        return 1
    if not args.detector_model.is_file():
        print(f"ERROR: detector model not found: {args.detector_model}")
        return 1

    debug_dir = args.save_dir or Path(tempfile.mkdtemp(prefix="faceattend-debug-"))
    print(f"Diagnostic images: {debug_dir}")

    frame = Frame(
        pixels=np.ascontiguousarray(image, dtype=np.uint8),
        captured_at_ns=time.monotonic_ns(),
        sequence_id=1,
    )
    _save_image(debug_dir, "01_original.png", image)
    if args.show:
        _show_image("1 - Original", image)

    try:
        faces = InsightFaceDetector(model_path=str(args.detector_model)).detect(frame)
        print(f"Detected faces: {len(faces)}")
        if len(faces) != 1:
            print("ERROR: expected exactly one face")
            return 1

        face = faces[0]
        print(f"Detector score: {face.detector_score:.4f}")
        print(f"Landmarks: {face.landmarks.shape}")

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
        _save_image(debug_dir, "02_detected.png", detected)
        if args.show:
            _show_image("2 - Detected face and landmarks", detected)

        aligned = InsightFaceAligner().align(frame, face)
        print(f"Aligned image: shape={aligned.shape}, dtype={aligned.dtype}")
        _save_image(debug_dir, "03_aligned.png", aligned)
        if args.show:
            _show_image("3 - Aligned face", aligned)

        embedder = InsightFaceEmbedder(model_path=str(args.embedding_model))
        embedding = embedder.embed(aligned)
        print(f"Embedding: dimensions={embedding.size}, dtype={embedding.dtype}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
        print(f"Model: {embedder.model_name} ({embedder.model_version})")
    except (AlignmentError, EmbeddingError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("SUCCESS: detector → alignment → embedding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
