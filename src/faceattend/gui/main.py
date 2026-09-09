"""Launch the Phase 4 FaceAttend desktop shell."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from faceattend.application.local_composition import LocalApplication
from faceattend.application.local_config import LocalAppConfig
from faceattend.gui.main_window import MainWindow
from faceattend.gui.runtime import DesktopRuntime


def main() -> int:
    application = QApplication(sys.argv)
    root = Path(__file__).resolve().parents[3]
    local_application = LocalApplication(LocalAppConfig.from_environment(root))
    runtime = DesktopRuntime(
        local_application.processor_factory,
        camera_index=local_application.config.camera_index,
        capture_interval_ms=local_application.config.capture_interval_ms,
        preview_interval_ms=local_application.config.preview_interval_ms,
        inference_interval_ms=local_application.config.inference_interval_ms,
        max_frame_age_ms=local_application.config.max_frame_age_ms,
    )
    window = MainWindow(runtime, local_application)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
