"""Tests for explicit local desktop configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from faceattend.application.local_composition import (
    LocalApplication,
    RegistrationInput,
    _start_before_first_frame,
)
from faceattend.application.local_config import LocalAppConfig
from faceattend.application.runtime import DesktopMode


def test_local_configuration_uses_explicit_camera_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACEATTEND_CAMERA_INDEX", "1")
    monkeypatch.setenv("FACEATTEND_DATABASE_PATH", str(tmp_path / "faceattend.sqlite3"))

    config = LocalAppConfig.from_environment(tmp_path)

    assert config.camera_index == 1
    assert config.database_path == tmp_path / "faceattend.sqlite3"
    assert config.manifest.embedding.metadata.name == "buffalo_l_recognition"


def test_local_configuration_does_not_default_to_developer_camera(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FACEATTEND_CAMERA_INDEX", raising=False)

    assert LocalAppConfig.from_environment(tmp_path).camera_index == 0


def test_first_frame_is_strictly_later_than_its_liveness_session_start() -> None:
    first_frame_ns = 1_234_567_890

    assert _start_before_first_frame(first_frame_ns) < first_frame_ns // 1_000_000


def test_real_factory_verifies_pinned_models_before_camera_capture(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = replace(LocalAppConfig.from_environment(root), database_path=tmp_path / "app.sqlite3")
    application = LocalApplication(config)
    application.configure_registration(RegistrationInput("Ada", True))

    processor = application.processor_factory(DesktopMode.REGISTRATION)

    processor.close()
