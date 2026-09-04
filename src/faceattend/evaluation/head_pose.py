"""Compare estimated head poses with controlled reference poses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from faceattend.vision.types import HeadPose


@dataclass(frozen=True, slots=True)
class PoseValidationSample:
    """One known-pose observation used for validation."""

    expected: HeadPose
    estimated: HeadPose
    label: str = ""


@dataclass(frozen=True, slots=True)
class PoseValidationSummary:
    """Aggregate error and pass/fail results for a validation set."""

    sample_count: int
    passed_count: int
    mean_absolute_error_degrees: HeadPose
    maximum_absolute_error_degrees: HeadPose
    tolerance_degrees: float

    @property
    def pass_rate(self) -> float:
        """Return the fraction of samples within tolerance on every axis."""
        if self.sample_count == 0:
            return 0.0
        return self.passed_count / self.sample_count


def _axis_errors(expected: HeadPose, estimated: HeadPose) -> tuple[float, float, float]:
    values = (
        abs(estimated.yaw_degrees - expected.yaw_degrees),
        abs(estimated.pitch_degrees - expected.pitch_degrees),
        abs(estimated.roll_degrees - expected.roll_degrees),
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("pose values must be finite")
    return values


def validate_pose_samples(
    samples: list[PoseValidationSample] | tuple[PoseValidationSample, ...],
    *,
    tolerance_degrees: float,
) -> PoseValidationSummary:
    """Summarize pose errors against controlled reference angles.

    A sample passes only when yaw, pitch, and roll are all within the supplied
    tolerance. This function does not decide whether a person is live; it only
    validates the numerical pose estimate.
    """
    if tolerance_degrees < 0 or not isfinite(tolerance_degrees):
        raise ValueError("tolerance_degrees must be finite and non-negative")

    if not samples:
        zero = HeadPose(0.0, 0.0, 0.0)
        return PoseValidationSummary(0, 0, zero, zero, tolerance_degrees)

    errors = [_axis_errors(sample.expected, sample.estimated) for sample in samples]
    passed_count = sum(
        all(error <= tolerance_degrees for error in sample_errors)
        for sample_errors in errors
    )
    mean = HeadPose(
        yaw_degrees=sum(error[0] for error in errors) / len(errors),
        pitch_degrees=sum(error[1] for error in errors) / len(errors),
        roll_degrees=sum(error[2] for error in errors) / len(errors),
    )
    maximum = HeadPose(
        yaw_degrees=max(error[0] for error in errors),
        pitch_degrees=max(error[1] for error in errors),
        roll_degrees=max(error[2] for error in errors),
    )
    return PoseValidationSummary(
        sample_count=len(samples),
        passed_count=passed_count,
        mean_absolute_error_degrees=mean,
        maximum_absolute_error_degrees=maximum,
        tolerance_degrees=tolerance_degrees,
    )
