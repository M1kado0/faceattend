"""Aggregate-only metrics for local active-liveness and PAD trial records.

These helpers consume the JSONL fields produced by the local trial runner. They
never accept image pixels, landmarks, templates, or identities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class LivenessTrial:
    """One non-identifying trial record needed for Phase 7 metrics."""

    trial_id: str
    split: str
    presentation: str
    active_status: str
    active_reason: str | None
    overall_accepted: bool
    passive_minimum: float | None
    passive_processing_failures: int
    duration_ms: float

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.split.strip() or not self.presentation.strip():
            raise ValueError("trial id, split, and presentation are required")
        if not isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration must be finite and non-negative")
        if self.passive_minimum is not None and (
            not isfinite(self.passive_minimum) or not 0.0 <= self.passive_minimum <= 1.0
        ):
            raise ValueError("passive minimum must be between zero and one")
        if self.passive_processing_failures < 0:
            raise ValueError("passive processing failures cannot be negative")

    @property
    def is_bona_fide(self) -> bool:
        return self.presentation == "bona_fide"


@dataclass(frozen=True, slots=True)
class Rate:
    """A metric value that always carries its denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator < 0 or self.numerator > self.denominator:
            raise ValueError("invalid rate counts")

    @property
    def rate(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    @property
    def wilson_95(self) -> tuple[float, float] | None:
        """Approximate 95% binomial interval with the denominator still visible."""
        if self.denominator == 0:
            return None
        z = 1.959963984540054
        proportion = self.numerator / self.denominator
        denominator = 1.0 + z**2 / self.denominator
        center = (proportion + z**2 / (2.0 * self.denominator)) / denominator
        margin = z * sqrt(
            (proportion * (1.0 - proportion) + z**2 / (4.0 * self.denominator))
            / self.denominator
        ) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True, slots=True)
class LatencySummary:
    sample_count: int
    p50: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True, slots=True)
class LivenessEvaluationSummary:
    trial_count: int
    bona_fide_count: int
    active_completion_count: int
    active_timeout_count: int
    false_failure_count: int
    completion_rate: Rate
    false_failure_rate: Rate
    timeout_rate: Rate
    bpcer: float | None
    bpcer_rate: Rate
    apcer_by_attack: dict[str, Rate]
    acer_by_attack: dict[str, float | None]
    failure_to_process_count: int
    failure_to_process_rate: Rate
    duration_ms: LatencySummary
    active_failure_reasons: dict[str, int]


@dataclass(frozen=True, slots=True)
class PassiveThresholdCandidate:
    """One validation-only operating point; never an automatic policy change."""

    threshold: float
    bpcer: Rate
    mean_apcer: float | None


def summarize_liveness_trials(
    trials: tuple[LivenessTrial, ...] | list[LivenessTrial],
    *,
    passive_threshold: float,
) -> LivenessEvaluationSummary:
    """Compute active and PAD metrics without silently dropping denominators.

    BPCER is calculated only over bona-fide attempts for which a PAD score was
    produced. Frames that did not reach PAD are separately reported as failures
    to process; active-liveness failures remain part of the overall false-failure
    rate rather than being mislabelled as a PAD classification error.
    """
    if not isfinite(passive_threshold) or not 0.0 <= passive_threshold <= 1.0:
        raise ValueError("passive threshold must be between zero and one")
    values = tuple(trials)
    bona_fide = tuple(trial for trial in values if trial.is_bona_fide)
    attacks = tuple(trial for trial in values if not trial.is_bona_fide)
    active_completed = sum(trial.active_status == "completed" for trial in bona_fide)
    timeouts = sum("timeout" in (trial.active_reason or "") for trial in bona_fide)
    false_failures = sum(not trial.overall_accepted for trial in bona_fide)
    active_reasons: dict[str, int] = {}
    for trial in bona_fide:
        if trial.active_status != "completed" and trial.active_reason is not None:
            active_reasons[trial.active_reason] = active_reasons.get(trial.active_reason, 0) + 1

    bona_fide_pad = tuple(trial for trial in bona_fide if trial.passive_minimum is not None)
    bpcer_errors = sum(_pad_rejected(trial, passive_threshold) for trial in bona_fide_pad)
    bpcer_rate = Rate(bpcer_errors, len(bona_fide_pad))

    apcer_by_attack: dict[str, Rate] = {}
    acer_by_attack: dict[str, float | None] = {}
    for attack in sorted({trial.presentation for trial in attacks}):
        samples = tuple(trial for trial in attacks if trial.presentation == attack)
        accepted = sum(not _pad_rejected(trial, passive_threshold) for trial in samples)
        apcer = Rate(accepted, len(samples))
        apcer_by_attack[attack] = apcer
        acer_by_attack[attack] = (
            None
            if apcer.rate is None or bpcer_rate.rate is None
            else (apcer.rate + bpcer_rate.rate) / 2.0
        )

    fto_count = sum(trial.passive_processing_failures > 0 for trial in values)
    return LivenessEvaluationSummary(
        trial_count=len(values),
        bona_fide_count=len(bona_fide),
        active_completion_count=active_completed,
        active_timeout_count=timeouts,
        false_failure_count=false_failures,
        completion_rate=Rate(active_completed, len(bona_fide)),
        false_failure_rate=Rate(false_failures, len(bona_fide)),
        timeout_rate=Rate(timeouts, len(bona_fide)),
        bpcer=bpcer_rate.rate,
        bpcer_rate=bpcer_rate,
        apcer_by_attack=apcer_by_attack,
        acer_by_attack=acer_by_attack,
        failure_to_process_count=fto_count,
        failure_to_process_rate=Rate(fto_count, len(values)),
        duration_ms=_latency_summary([trial.duration_ms for trial in values]),
        active_failure_reasons=active_reasons,
    )


def trial_from_record(record: Mapping[str, Any]) -> LivenessTrial:
    """Parse a JSONL metrics object emitted by ``run_liveness_trial.py``."""
    try:
        return LivenessTrial(
            trial_id=str(record["trial_id"]),
            split=str(record["split"]),
            presentation=str(record["presentation"]),
            active_status=str(record["active_status"]),
            active_reason=_optional_string(record.get("active_reason")),
            overall_accepted=bool(record["overall_accepted"]),
            passive_minimum=_optional_float(record.get("passive_minimum")),
            passive_processing_failures=int(record.get("passive_processing_failures", 0)),
            duration_ms=float(record["duration_ms"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid liveness trial record") from exc


def passive_threshold_candidates(
    trials: tuple[LivenessTrial, ...] | list[LivenessTrial],
) -> tuple[PassiveThresholdCandidate, ...]:
    """Return every score-derived PAD operating point for validation review.

    Callers must keep validation and held-out test records separate. Selecting
    and freezing one candidate is an explicit project/ADR decision, not an
    automatic side effect of this function.
    """
    values = tuple(trials)
    thresholds = sorted(
        {
            trial.passive_minimum
            for trial in values
            if trial.passive_minimum is not None
        }
    )
    candidates: list[PassiveThresholdCandidate] = []
    for threshold in thresholds:
        summary = summarize_liveness_trials(values, passive_threshold=threshold)
        attack_rates = [
            rate.rate for rate in summary.apcer_by_attack.values() if rate.rate is not None
        ]
        candidates.append(
            PassiveThresholdCandidate(
                threshold,
                summary.bpcer_rate,
                None if not attack_rates else sum(attack_rates) / len(attack_rates),
            )
        )
    return tuple(candidates)


def _latency_summary(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(0, None, None, None)
    ordered = sorted(values)
    return LatencySummary(
        len(ordered),
        float(median(ordered)),
        _percentile(ordered, 0.95),
        _percentile(ordered, 0.99),
    )


def _pad_rejected(trial: LivenessTrial, threshold: float) -> bool:
    """Return true when PAD cannot accept a trial at the stated threshold."""
    return (
        trial.passive_minimum is None
        or trial.passive_minimum < threshold
        or trial.passive_processing_failures > 0
    )


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("optional score must be numeric")
    return float(value)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
