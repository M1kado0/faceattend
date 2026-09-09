"""Headless direct InsightFace detection-model adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from insightface.model_zoo import get_model  # type: ignore[import-untyped]

from faceattend.vision.model_lifecycle import LazyModel, file_sha256
from faceattend.vision.quality import measure_quality
from faceattend.vision.types import (
    BoundingBox,
    FaceObservation,
    Frame,
    ModelMetadata,
)


class FaceDetectionError(ValueError):
    """Raised when the configured detector cannot be loaded or used."""


def _landmarks(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise FaceDetectionError("detector landmarks must have shape (N, 2)")
    return np.ascontiguousarray(array[:, :2], dtype=np.float32)


def _observation(
    bbox: Any,
    score: Any,
    keypoints: Any,
    frame: Frame,
) -> FaceObservation:
    bbox_values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox_values.size < 4 or not np.isfinite(bbox_values[:4]).all():
        raise FaceDetectionError("detector returned an invalid bounding box")

    x_min, y_min, x_max, y_max = (float(value) for value in bbox_values[:4])
    box = BoundingBox(x_min, y_min, x_max, y_max)
    points = _landmarks(keypoints)
    return FaceObservation(
        bbox=box,
        detector_score=float(score),
        landmarks=points,
        quality=measure_quality(frame, box, points),
    )


class InsightFaceDetector:
    """Use only InsightFace's detection ONNX model and its five keypoints."""

    def __init__(
        self,
        *,
        model_path: str = "models/det_10g.onnx",
        model_name: str = "insightface-detector",
        model_version: str = "buffalo_l-det_10g",
        model_checksum: str = "unknown",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        det_size: tuple[int, int] = (640, 640),
        det_threshold: float = 0.5,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        checksum = model_checksum
        if checksum == "unknown":
            checksum = file_sha256(model_path)
        self._metadata = ModelMetadata(model_name, model_version, checksum)
        self._det_size = det_size
        self._det_threshold = det_threshold

        def load() -> Any:
            if model_loader is not None:
                return model_loader()
            try:
                model = get_model(model_path, providers=list(providers))
            except (AssertionError, OSError) as exc:
                raise FaceDetectionError(f"could not load detector: {model_path}") from exc
            if model is None:
                raise FaceDetectionError(f"could not load detector: {model_path}")
            model.prepare(
                ctx_id=-1,
                input_size=det_size,
                det_thresh=det_threshold,
            )
            return model

        self._model = LazyModel(load)

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    def detect(self, frame: Frame) -> list[FaceObservation]:
        try:
            bboxes, keypoints = self._model.get().detect(
                frame.pixels,
                input_size=self._det_size,
                max_num=0,
                metric="default",
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise FaceDetectionError("detector inference failed") from exc

        if bboxes is None or len(bboxes) == 0:
            return []
        return [
            _observation(
                bbox,
                bbox[4] if len(bbox) > 4 else 0.0,
                None if keypoints is None else keypoints[index],
                frame,
            )
            for index, bbox in enumerate(bboxes)
        ]
