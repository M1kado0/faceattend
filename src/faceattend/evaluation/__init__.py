"""Headless evaluation helpers for reproducible CV measurements."""

from faceattend.evaluation.head_pose import (
    PoseValidationSample,
    PoseValidationSummary,
    validate_pose_samples,
)
from faceattend.evaluation.liveness import (
    LivenessEvaluationSummary,
    LivenessTrial,
    PassiveThresholdCandidate,
    Rate,
    passive_threshold_candidates,
    summarize_liveness_trials,
    trial_from_record,
)
from faceattend.evaluation.recognition import (
    RecognitionOperatingPoint,
    VerificationEvaluationSummary,
    VerificationScore,
    summarize_verification_scores,
    verification_score_from_record,
)

__all__ = [
    "PoseValidationSample",
    "PoseValidationSummary",
    "validate_pose_samples",
    "LivenessEvaluationSummary",
    "LivenessTrial",
    "PassiveThresholdCandidate",
    "Rate",
    "passive_threshold_candidates",
    "summarize_liveness_trials",
    "trial_from_record",
    "RecognitionOperatingPoint",
    "VerificationEvaluationSummary",
    "VerificationScore",
    "summarize_verification_scores",
    "verification_score_from_record",
]
