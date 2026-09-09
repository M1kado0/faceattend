"""Local ONNX wrapper for the MiniFASNetV2 model artifact."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]


class MiniFASNetModel:
    """Run the pinned local MiniFASNetV2 model on one BGR face crop."""

    def __init__(self, model_path: str, *, scale: float = 2.7) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.scale = scale
        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        self.input_size = tuple(input_config.shape[2:])
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: np.ndarray, bbox_xyxy: list[float]) -> dict[str, object]:
        x1, y1, x2, y2 = bbox_xyxy
        x, y = int(x1), int(y1)
        width, height = int(x2 - x1), int(y2 - y1)
        if width <= 0 or height <= 0:
            raise ValueError("face bounding box must have positive dimensions")
        image_height, image_width = image.shape[:2]
        scale = min((image_height - 1) / height, (image_width - 1) / width, self.scale)
        center_x, center_y = x + width / 2, y + height / 2
        crop_x1 = max(0, int(center_x - width * scale / 2))
        crop_y1 = max(0, int(center_y - height * scale / 2))
        crop_x2 = min(image_width - 1, int(center_x + width * scale / 2))
        crop_y2 = min(image_height - 1, int(center_y + height * scale / 2))
        crop = image[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
        if crop.size == 0:
            raise ValueError("face crop is empty")
        resized = cv2.resize(crop, self.input_size[::-1]).astype(np.float32)
        tensor = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probabilities = _softmax(np.asarray(logits, dtype=np.float32))
        label = int(np.argmax(probabilities))
        return {"score": float(probabilities[0, 1]), "label_idx": label}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / exponentials.sum(axis=1, keepdims=True), dtype=np.float32)
