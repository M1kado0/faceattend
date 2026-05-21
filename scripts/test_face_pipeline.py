"""Smoke test the local face pipeline without running the ML HTTP service.

Usage:
    uv run python scripts/test_face_pipeline.py images/11.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ml.pipeline.face import (  # noqa: E402
    EmptyEmbeddingError,
    FaceEmbeddingResult,
    ImageDecodeError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    embed_face,
    verify_passive_liveness,
)


def _read_image(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path.read_bytes()


def _print_embedding(result: FaceEmbeddingResult) -> None:
    print("Embedding")
    print(f"  model_version: {result.model_version}")
    print(f"  dimensions: {result.embedding.shape[0]}")
    print(f"  norm: {np.linalg.norm(result.embedding):.6f}")
    print(f"  face_count: {result.face_count}")
    print(f"  bbox: {result.bbox}")
    print(f"  detector_score: {result.detector_score}")


def _print_liveness(result: dict) -> None:
    print("Liveness")
    print(f"  passed: {result.get('passed')}")
    print(f"  score: {result.get('score')}")
    print(f"  label: {result.get('label')}")
    print(f"  reason: {result.get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test FaceAttend's local face pipeline.")
    parser.add_argument("image", type=Path, help="Path to an image file.")
    args = parser.parse_args()

    raw = _read_image(args.image)

    try:
        embedding = embed_face(raw)
        liveness = verify_passive_liveness(raw)
    except ImageDecodeError:
        print("ERROR: could_not_decode_image")
        raise SystemExit(1) from None
    except NoFaceDetectedError:
        print("ERROR: no_faces_detected")
        raise SystemExit(1) from None
    except MultipleFacesDetectedError:
        print("ERROR: multiple_faces_detected")
        raise SystemExit(1) from None
    except EmptyEmbeddingError:
        print("ERROR: empty_embedding")
        raise SystemExit(1) from None

    _print_embedding(embedding)
    print()
    _print_liveness(liveness)


if __name__ == "__main__":
    main()
