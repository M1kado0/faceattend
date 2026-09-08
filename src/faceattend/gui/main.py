"""Launch the Phase 4 FaceAttend desktop shell."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from faceattend.application.runtime import DesktopMode, SessionProcessor
from faceattend.gui.main_window import MainWindow
from faceattend.gui.runtime import DesktopRuntime


def _unconfigured_processor(_mode: DesktopMode) -> SessionProcessor:
    raise RuntimeError("CV session is not configured; connect the Phase 2 runtime factory")


def main() -> int:
    application = QApplication(sys.argv)
    runtime = DesktopRuntime(_unconfigured_processor)
    window = MainWindow(runtime)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
