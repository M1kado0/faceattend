"""Tests for aggregate-only Phase 7 liveness evaluation metrics."""

from __future__ import annotations

import pytest

from faceattend.evaluation.liveness import (
    LivenessTrial,
    Rate,
    passive_threshold_candidates,
    summarize_liveness_trials,
)


def _trial(
    trial_id: str,
    *,
    presentation: str,
    active_status: str = "completed",
    active_reason: str | None = None,
    overall_accepted: bool = True,
    passive_minimum: float | None = 0.9,
    passive_processing_failures: int = 0,
    duration_ms: float = 1_000.0,
) -> LivenessTrial:
    return LivenessTrial(
        trial_id=trial_id,
        split="validation",
        presentation=presentation,
        active_status=active_status,
        active_reason=active_reason,
        overall_accepted=overall_accepted,
        passive_minimum=passive_minimum,
        passive_processing_failures=passive_processing_failures,
        duration_ms=duration_ms,
    )


def test_summary_reports_active_and_pad_rates_with_explicit_denominators() -> None:
    summary = summarize_liveness_trials(
        (
            _trial("bf-pass", presentation="bona_fide", duration_ms=1_000),
            _trial(
                "bf-timeout",
                presentation="bona_fide",
                active_status="timeout",
                active_reason="challenge_timeout",
                overall_accepted=False,
                passive_minimum=None,
                duration_ms=2_000,
            ),
            _trial(
                "bf-pad-reject",
                presentation="bona_fide",
                overall_accepted=False,
                passive_minimum=0.20,
            ),
            _trial("print-pass", presentation="print", overall_accepted=True, passive_minimum=0.95),
            _trial(
                "print-reject",
                presentation="print",
                overall_accepted=False,
                passive_minimum=0.10,
                passive_processing_failures=1,
                duration_ms=3_000,
            ),
        ),
        passive_threshold=0.85,
    )

    assert summary.bona_fide_count == 3
    assert summary.active_completion_count == 2
    assert summary.active_timeout_count == 1
    assert summary.false_failure_count == 2
    assert summary.bpcer == 0.5
    assert summary.apcer_by_attack["print"].denominator == 2
    assert summary.apcer_by_attack["print"].rate == 0.5
    assert summary.acer_by_attack["print"] == 0.5
    assert summary.failure_to_process_count == 1
    assert summary.duration_ms.p50 == 1_000.0


def test_summary_rejects_missing_or_invalid_trial_values() -> None:
    with pytest.raises(ValueError, match="duration"):
        _trial("bad", presentation="bona_fide", duration_ms=-1.0)


def test_threshold_candidates_only_expose_validation_operating_points() -> None:
    candidates = passive_threshold_candidates(
        (
            _trial("bf", presentation="bona_fide", passive_minimum=0.9),
            _trial("attack", presentation="print", passive_minimum=0.2),
        )
    )

    assert [candidate.threshold for candidate in candidates] == [0.2, 0.9]
    assert candidates[0].bpcer.rate == 0.0
    assert candidates[1].mean_apcer == 0.0


def test_rate_exposes_denominator_aware_uncertainty() -> None:
    interval = Rate(2, 10).wilson_95

    assert interval is not None
    assert interval[0] < 0.2 < interval[1]
    assert Rate(0, 0).wilson_95 is None
