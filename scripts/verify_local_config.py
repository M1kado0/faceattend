"""Verify local FaceAttend paths and model checksums without opening a camera."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from faceattend.application.local_config import LocalAppConfig  # noqa: E402
from faceattend.vision.face_analyzer import validate_model_files  # noqa: E402


def main() -> int:
    config = LocalAppConfig.from_environment(root)
    validate_model_files(config.manifest)
    print(f"camera_index={config.camera_index}")
    print(f"database_path={config.database_path}")
    print("model_manifest=verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
