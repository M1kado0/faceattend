"""Launch the local FaceAttend PySide6 shell from the repository checkout."""

from __future__ import annotations

import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repository_root / "src"))
sys.path.insert(0, str(repository_root))

from faceattend.gui.main import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
