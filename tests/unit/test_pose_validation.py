"""Tests for controlled head-pose validation summaries."""

import pytest

from faceattend.evaluation.head_pose import PoseValidationSample, validate_pose_samples
from faceattend.vision.types import HeadPose


def test_validation_reports_axis_errors_and_pass_rate() -> None:
    summary = validate_pose_samples(
        [
            PoseValidationSample(HeadPose(0.0, 0.0, 0.0), HeadPose(1.0, -2.0, 0.5)),
            PoseValidationSample(HeadPose(20.0, 0.0, 5.0), HeadPose(24.0, 1.0, 5.0)),
        ],
        tolerance_degrees=3.0,
    )

    assert summary.sample_count == 2
    assert summary.passed_count == 1
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.mean_absolute_error_degrees == HeadPose(2.5, 1.5, 0.25)
    assert summary.maximum_absolute_error_degrees == HeadPose(4.0, 2.0, 0.5)


def test_empty_validation_set_is_explicit() -> None:
    summary = validate_pose_samples([], tolerance_degrees=2.0)

    assert summary.sample_count == 0
    assert summary.pass_rate == 0.0
    assert summary.mean_absolute_error_degrees == HeadPose(0.0, 0.0, 0.0)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        validate_pose_samples([], tolerance_degrees=tolerance)


def test_non_finite_pose_is_rejected() -> None:
    sample = PoseValidationSample(HeadPose(0.0, 0.0, 0.0), HeadPose(float("nan"), 0.0, 0.0))

    with pytest.raises(ValueError, match="finite"):
        validate_pose_samples([sample], tolerance_degrees=2.0)
