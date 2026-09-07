from dataclasses import replace

import cv2
import numpy as np
import pytest

from faceattend.vision.quality import QualityConfig, measure_quality
from faceattend.vision.types import BoundingBox, Frame


def test_measured_sharpness_distinguishes_blur_and_reports_exposure() -> None:
    pixels = np.random.default_rng(1).integers(40, 210, (200, 200, 3), dtype=np.uint8)
    box = BoundingBox(40, 40, 160, 160)
    points = np.array([[70, 80], [130, 80], [100, 100], [80, 130], [120, 130]], np.float32)
    frame = Frame(pixels, 0, 0)
    sharp = measure_quality(frame, box, points)
    blurred = measure_quality(
        replace(frame, pixels=cv2.GaussianBlur(pixels, (31, 31), 10)), box, points
    )
    assert sharp.passed and sharp.sharpness > blurred.sharpness
    assert blurred.reason == "blurry_face"
    assert 0.4 < sharp.brightness < 0.6
    assert sharp.face_area_ratio == pytest.approx(0.36)
    for intensity, reason in ((0, "underexposed"), (255, "overexposed")):
        quality = measure_quality(
            replace(frame, pixels=np.full_like(pixels, intensity)), box, points
        )
        assert quality.reason == reason


@pytest.mark.parametrize(
    "box,reason",
    [
        (BoundingBox(80, 80, 100, 100), "face_too_small"),
        (BoundingBox(0, 0, 200, 200), "face_too_large"),
        (BoundingBox(-1, 40, 160, 160), "face_clipped"),
        (BoundingBox(130, 70, 195, 140), "face_off_center"),
    ],
)
def test_geometric_quality_gates(box: BoundingBox, reason: str) -> None:
    pixels = np.random.default_rng(2).integers(40, 210, (200, 200, 3), dtype=np.uint8)
    points = np.array([[box.x_min + 5, box.y_min + 5]] * 5, np.float32)
    quality = measure_quality(
        Frame(pixels, 0, 0), box, points, QualityConfig(max_center_offset=0.25)
    )
    assert quality.reason == reason


def test_occlusion_signals_are_advisory_not_glasses_or_mask_diagnoses() -> None:
    pixels = np.random.default_rng(3).integers(40, 210, (200, 200, 3), dtype=np.uint8)
    pixels[70:91, 60:81] = 0
    pixels[70:91, 120:141] = 255
    pixels[100:160, 40:160] = 110
    points = np.array([[70, 80], [130, 80], [100, 100], [80, 130], [120, 130]], np.float32)
    q = measure_quality(Frame(pixels, 0, 0), BoundingBox(40, 40, 160, 160), points)
    assert q.passed
    assert "eye_region_dark_possible_occlusion" in q.warnings
    assert "eye_region_glare" in q.warnings
    assert "lower_face_low_texture_possible_occlusion" in q.warnings


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_quality_config_rejects_invalid_thresholds(value: float) -> None:
    with pytest.raises(ValueError):
        QualityConfig(min_sharpness=value)
