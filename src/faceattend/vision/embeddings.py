"""Headless InsightFace recognition-model adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from insightface.model_zoo import get_model  # type: ignore[import-untyped]

from faceattend.vision.model_lifecycle import LazyModel, file_sha256
from faceattend.vision.types import Float32Array, ModelMetadata, UInt8Array


class EmbeddingError(ValueError):
    """Raised when an aligned face cannot produce a usable embedding."""


class InsightFaceEmbedder:
    """Generate normalized embeddings from already-aligned face images.

    The input image must be an HWC, three-channel ``uint8`` array in BGR order,
    which is the format produced by the OpenCV alignment pipeline. Detection,
    alignment, liveness, and persistence deliberately live outside this class.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_name: str = "buffalo_l_recognition",
        model_version: str = "buffalo_l",
        model_checksum: str = "unknown",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        checksum = model_checksum
        if checksum == "unknown":
            checksum = file_sha256(model_path)
        self._metadata = ModelMetadata(model_name, model_version, checksum)

        def load() -> Any:
            if model_loader is not None:
                return model_loader()
            model = get_model(model_path, providers=list(providers))
            if model is None:
                raise EmbeddingError(f"could not load recognition model: {model_path}")
            model.prepare(ctx_id=-1)
            return model

        self._model = LazyModel(load)

    @property
    def model_name(self) -> str:
        return self._metadata.name

    @property
    def model_version(self) -> str:
        return self._metadata.version

    @property
    def model_checksum(self) -> str:
        return self._metadata.checksum

    def embed(self, aligned_face: UInt8Array) -> Float32Array:
        image = np.asarray(aligned_face)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise EmbeddingError("aligned face must be an HWC uint8 BGR image")

        raw_embedding = np.asarray(self._model.get().get_feat(image), dtype=np.float32)
        embedding = np.ascontiguousarray(raw_embedding.reshape(-1), dtype=np.float32)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise EmbeddingError("recognition model returned an invalid embedding")

        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            raise EmbeddingError("recognition model returned a zero embedding")
        embedding /= norm
        return embedding
