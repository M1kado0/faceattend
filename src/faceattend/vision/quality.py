"""Measured face-crop quality. Defaults are provisional, not calibrated PAD policy."""

from dataclasses import dataclass, fields
from math import isfinite

import cv2
import numpy as np

from faceattend.vision.types import BoundingBox, FaceQuality, Float32Array, Frame


@dataclass(frozen=True, slots=True)
class QualityConfig:
    min_brightness: float = 0.12
    max_brightness: float = 0.90
    max_clipped_fraction: float = 0.50
    min_sharpness: float = 20.0
    min_face_pixels: float = 60.0
    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.85
    max_center_offset: float = 0.40
    eye_dark_ratio: float = 0.45
    eye_glare_fraction: float = 0.25
    lower_face_std: float = 5.0

    def __post_init__(self) -> None:
        if any(
            not isfinite(getattr(self, f.name)) or getattr(self, f.name) < 0 for f in fields(self)
        ):
            raise ValueError("quality thresholds must be finite and nonnegative")
        if not self.min_brightness <= self.max_brightness <= 1:
            raise ValueError("invalid brightness bounds")
        if not self.min_area_ratio <= self.max_area_ratio <= 1:
            raise ValueError("invalid area bounds")
        if self.max_clipped_fraction > 1 or self.eye_glare_fraction > 1:
            raise ValueError("fractions must not exceed one")


def measure_quality(
    frame: Frame,
    box: BoundingBox,
    landmarks: Float32Array,
    config: QualityConfig | None = None,
) -> FaceQuality:
    """BGR ROI mean, clipping and 128px Laplacian variance; geometric gates.

    Eye darkness/glare and a textureless lower face are advisory signals only.
    They cannot establish glasses/mask presence and never substitute for PAD.
    """
    c = config or QualityConfig()
    h, w = frame.pixels.shape[:2]
    bounds = (box.x_min, box.y_min, box.x_max, box.y_max)
    if not all(isfinite(v) for v in bounds) or box.x_max <= box.x_min or box.y_max <= box.y_min:
        return FaceQuality(False, 0, 0, 0, "invalid_face_box")
    x1, y1 = max(0, int(box.x_min)), max(0, int(box.y_min))
    x2, y2 = min(w, int(box.x_max)), min(h, int(box.y_max))
    if x2 <= x1 or y2 <= y1:
        return FaceQuality(False, 0, 0, 0, "face_outside_frame")
    gray = cv2.cvtColor(frame.pixels[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean() / 255)
    dark, bright = float(np.mean(gray <= 8)), float(np.mean(gray >= 247))
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    area = (x2 - x1) * (y2 - y1) / (w * h)
    offset = max(
        abs((box.x_min + box.x_max) / (2 * w) - 0.5), abs((box.y_min + box.y_max) / (2 * h) - 0.5)
    )
    checks = (
        (box.x_min < 0 or box.y_min < 0 or box.x_max > w or box.y_max > h, "face_clipped"),
        (min(x2 - x1, y2 - y1) < c.min_face_pixels or area < c.min_area_ratio, "face_too_small"),
        (area > c.max_area_ratio, "face_too_large"),
        (offset > c.max_center_offset, "face_off_center"),
        (brightness < c.min_brightness or dark > c.max_clipped_fraction, "underexposed"),
        (brightness > c.max_brightness or bright > c.max_clipped_fraction, "overexposed"),
        (sharpness < c.min_sharpness, "blurry_face"),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    warnings: list[str] = []
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        reason = "invalid_alignment_landmarks"
    elif np.any(landmarks < (x1, y1)) or np.any(landmarks >= (x2, y2)):
        reason = "landmarks_outside_face"
    else:
        # Landmark-centred regions avoid fixed skin-colour assumptions.
        distance = float(np.linalg.norm(landmarks[1] - landmarks[0]))
        for point in landmarks[:2]:
            px, py = point - (x1, y1)
            r = max(1, int(distance * 0.18))
            eye = gray[max(0, int(py) - r) : int(py) + r + 1, max(0, int(px) - r) : int(px) + r + 1]
            if eye.size and float(eye.mean()) < float(gray.mean()) * c.eye_dark_ratio:
                warnings.append("eye_region_dark_possible_occlusion")
            if eye.size and float(np.mean(eye >= 247)) > c.eye_glare_fraction:
                warnings.append("eye_region_glare")
        lower = gray[max(0, int(landmarks[2, 1]) - y1) :]
        if lower.size and float(lower.std()) < c.lower_face_std:
            warnings.append("lower_face_low_texture_possible_occlusion")
    return FaceQuality(
        reason is None,
        sharpness,
        brightness,
        area,
        reason,
        dark,
        bright,
        offset,
        tuple(dict.fromkeys(warnings)),
    )
