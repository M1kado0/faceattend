"""Explicit local-only configuration for the desktop application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from faceattend.vision.face_analyzer import ModelArtifact, RuntimeModelManifest
from faceattend.vision.model_lifecycle import file_sha256
from faceattend.vision.types import ModelMetadata


@dataclass(frozen=True, slots=True)
class LocalAppConfig:
    database_path: Path
    camera_index: int
    detector: ModelArtifact
    embedding: ModelArtifact
    passive: ModelArtifact
    landmarker: ModelArtifact
    passive_threshold: float = 0.85
    match_threshold: float = 0.75
    ambiguity_margin: float = 0.05
    capture_interval_ms: int = 15
    preview_interval_ms: int = 33
    inference_interval_ms: int = 100
    max_frame_age_ms: int = 500

    @property
    def manifest(self) -> RuntimeModelManifest:
        return RuntimeModelManifest(self.detector, self.embedding, self.passive, self.landmarker)

    @classmethod
    def from_environment(cls, root: Path) -> LocalAppConfig:
        """Load local ``.env`` settings, then apply environment overrides.

        Hashes are calculated at startup and then verified before model use. A
        release should pin them in its launch configuration instead of trusting
        a mutable model directory.
        """
        # A developer's shell environment remains authoritative. This makes a
        # local .env convenient without preventing one-off command overrides.
        load_dotenv(root / ".env", override=False)
        model_dir = Path(os.environ.get("FACEATTEND_MODEL_DIR", root / "models"))
        database = Path(
            os.environ.get("FACEATTEND_DATABASE_PATH", root / "data" / "faceattend.sqlite3")
        )

        def artifact(filename: str, name: str, version: str) -> ModelArtifact:
            path = model_dir / filename
            return ModelArtifact(path, ModelMetadata(name, version, file_sha256(path)))

        return cls(
            database_path=database,
            # Do not default to a machine-specific camera. Set it in local .env.
            camera_index=int(os.environ.get("FACEATTEND_CAMERA_INDEX", "0")),
            detector=artifact("det_10g.onnx", "buffalo_l_detector", "buffalo_l"),
            embedding=artifact("w600k_r50.onnx", "buffalo_l_recognition", "buffalo_l"),
            passive=artifact("MiniFASNetV2.onnx", "MiniFASNetV2", "MiniFASNetV2"),
            landmarker=artifact(
                "face_landmarker_v2_with_blendshapes.task",
                "face_landmarker_v2_with_blendshapes",
                "v2",
            ),
            passive_threshold=float(os.environ.get("FACEATTEND_PASSIVE_THRESHOLD", "0.85")),
            match_threshold=float(os.environ.get("FACEATTEND_MATCH_THRESHOLD", "0.75")),
            ambiguity_margin=float(os.environ.get("FACEATTEND_AMBIGUITY_MARGIN", "0.05")),
        )
