"""Headless MiniFASNet passive presentation-attack detector adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

from faceattend.vision.model_lifecycle import LazyModel
from faceattend.vision.types import (
    EvidenceDecision,
    FaceObservation,
    Frame,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)


class PassiveLivenessError(ValueError):
    """Raised when passive-liveness input or model output is invalid."""


class MiniFASNetPassiveLivenessDetector:
    """Evaluate one or more tracked faces using the existing MiniFASNet model.

    A temporal window is accepted only when every frame's score reaches the
    threshold. This conservative aggregation prevents one good frame from
    hiding a failed frame during registration or check-in.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_name: str = "MiniFASNetV2",
        model_version: str = "MiniFASNetV2",
        model_checksum: str = "unknown",
        threshold: float = 0.85,
        scale: float = 2.7,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if scale <= 0.0:
            raise ValueError("scale must be positive")

        self._metadata = ModelMetadata(model_name, model_version, model_checksum)
        self._threshold = threshold

        def load() -> Any:
            if model_loader is not None:
                return model_loader()
            # Import lazily so this headless boundary does not load ONNX Runtime
            # until the detector is actually used.
            minifasnet_module = import_module("ml.liveness.minifasnet")
            return minifasnet_module.MiniFASNet(model_path, scale=scale)

        self._model = LazyModel(load)

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    def evaluate(
        self,
        frames: Sequence[Frame],
        faces: Sequence[FaceObservation],
    ) -> LivenessEvidence:
        if not frames or not faces:
            raise PassiveLivenessError("at least one frame and face are required")
        if len(frames) != len(faces):
            raise PassiveLivenessError("one face observation is required per frame")

        scores: list[float] = []
        for frame, face in zip(frames, faces, strict=True):
            bbox = face.bbox
            bbox_xyxy = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
            try:
                result = self._model.get().predict(frame.pixels, bbox_xyxy)
                score = float(result["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PassiveLivenessError("passive model returned invalid output") from exc
            if not 0.0 <= score <= 1.0:
                raise PassiveLivenessError("passive model score must be between 0 and 1")
            scores.append(score)

        score = min(scores)
        passed = score >= self._threshold
        return LivenessEvidence(
            kind=LivenessKind.PASSIVE,
            decision=EvidenceDecision.PASSED if passed else EvidenceDecision.FAILED,
            score=score,
            threshold=self._threshold,
            model_version=self._metadata.version,
            reason=None if passed else "liveness_score_below_threshold",
        )
