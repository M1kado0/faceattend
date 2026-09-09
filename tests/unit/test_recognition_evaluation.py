"""Score-only tests for face-recognition evaluation metrics."""

from __future__ import annotations

from faceattend.evaluation.recognition import (
    VerificationScore,
    summarize_verification_scores,
    verification_score_from_record,
)


def test_verification_summary_reports_operating_point_and_eer() -> None:
    summary = summarize_verification_scores(
        (
            VerificationScore(0.95, True, "normal"),
            VerificationScore(0.85, True, "normal"),
            VerificationScore(0.70, True, "dim"),
            VerificationScore(0.80, False, "normal"),
            VerificationScore(0.20, False, "normal"),
            VerificationScore(0.10, False, "dim"),
        ),
        threshold=0.75,
    )

    assert summary.genuine_count == 3
    assert summary.impostor_count == 3
    assert summary.fmr.numerator == 1
    assert summary.fmr.denominator == 3
    assert summary.fnmr.numerator == 1
    assert summary.fnmr.denominator == 3
    assert summary.eer is not None
    assert len(summary.roc_points) == 6


def test_verification_summary_requires_both_genuine_and_impostor_scores() -> None:
    try:
        summarize_verification_scores((VerificationScore(0.9, True, "normal"),), threshold=0.75)
    except ValueError as exc:
        assert "impostor" in str(exc)
    else:
        raise AssertionError("verification evaluation requires an impostor denominator")


def test_verification_score_record_has_no_identity_requirement() -> None:
    score = verification_score_from_record(
        {"score": 0.92, "is_genuine": True, "condition": "normal"}
    )

    assert score.score == 0.92
    assert score.is_genuine
