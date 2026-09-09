"""Score-only face-recognition evaluation helpers.

The functions in this module deliberately operate on labelled similarity scores,
not embeddings, images, or person identifiers.  A trial collection tool may
produce these labels locally; this module only computes reproducible aggregate
operating points from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from faceattend.evaluation.liveness import Rate


@dataclass(frozen=True, slots=True)
class VerificationScore:
    """One labelled cosine-similarity comparison for offline evaluation."""

    score: float
    is_genuine: bool
    condition: str

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not -1.0 <= self.score <= 1.0:
            raise ValueError("cosine similarity score must be finite and between -1 and 1")
        if not self.condition.strip():
            raise ValueError("condition is required")


@dataclass(frozen=True, slots=True)
class RecognitionOperatingPoint:
    """A threshold with false-match and false-non-match counts."""

    threshold: float
    fmr: Rate
    fnmr: Rate


@dataclass(frozen=True, slots=True)
class VerificationEvaluationSummary:
    """Aggregate verification metrics for a fixed local evaluation split."""

    genuine_count: int
    impostor_count: int
    fmr: Rate
    fnmr: Rate
    eer: float | None
    roc_points: tuple[RecognitionOperatingPoint, ...]
    condition_counts: dict[str, int]


def summarize_verification_scores(
    scores: tuple[VerificationScore, ...] | list[VerificationScore],
    *,
    threshold: float,
) -> VerificationEvaluationSummary:
    """Summarize labelled comparison scores at a stated decision threshold.

    A match is accepted when ``score >= threshold``.  The returned EER is a
    discrete score-sweep approximation, so it is a compact diagnostic rather
    than a substitute for a plotted ROC/DET curve or confidence intervals.
    """
    if not isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between -1 and 1")
    values = tuple(scores)
    genuine = tuple(score for score in values if score.is_genuine)
    impostor = tuple(score for score in values if not score.is_genuine)
    if not genuine:
        raise ValueError("verification evaluation requires a genuine denominator")
    if not impostor:
        raise ValueError("verification evaluation requires an impostor denominator")

    operating_point = _operating_point(values, threshold)
    roc_points = tuple(
        _operating_point(values, score_threshold)
        for score_threshold in sorted({score.score for score in values}, reverse=True)
    )
    closest_eer = min(
        roc_points,
        key=lambda point: abs((point.fmr.rate or 0.0) - (point.fnmr.rate or 0.0)),
    )
    condition_counts: dict[str, int] = {}
    for score in values:
        condition_counts[score.condition] = condition_counts.get(score.condition, 0) + 1

    return VerificationEvaluationSummary(
        genuine_count=len(genuine),
        impostor_count=len(impostor),
        fmr=operating_point.fmr,
        fnmr=operating_point.fnmr,
        eer=((closest_eer.fmr.rate or 0.0) + (closest_eer.fnmr.rate or 0.0)) / 2.0,
        roc_points=roc_points,
        condition_counts=condition_counts,
    )


def verification_score_from_record(record: Mapping[str, Any]) -> VerificationScore:
    """Parse a non-identifying local comparison-score record."""
    try:
        return VerificationScore(
            score=float(record["score"]),
            is_genuine=bool(record["is_genuine"]),
            condition=str(record["condition"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid verification score record") from exc


def _operating_point(
    scores: tuple[VerificationScore, ...], threshold: float
) -> RecognitionOperatingPoint:
    genuine = tuple(score for score in scores if score.is_genuine)
    impostor = tuple(score for score in scores if not score.is_genuine)
    false_matches = sum(score.score >= threshold for score in impostor)
    false_non_matches = sum(score.score < threshold for score in genuine)
    return RecognitionOperatingPoint(
        threshold=threshold,
        fmr=Rate(false_matches, len(impostor)),
        fnmr=Rate(false_non_matches, len(genuine)),
    )
