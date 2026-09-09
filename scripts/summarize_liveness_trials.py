"""Summarize aggregate-only Phase 7 liveness trial JSONL.

Example:
  PYTHONPATH=src uv run python scripts/summarize_liveness_trials.py \
    --input local-biometric-recordings/liveness-trials.jsonl --split validation

The input directory is ignored by Git. This command never reads or writes image
data, embeddings, landmarks, or attendance records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from faceattend.evaluation.liveness import (  # noqa: E402
    LivenessEvaluationSummary,
    Rate,
    summarize_liveness_trials,
    trial_from_record,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Ignored local JSONL metrics file."
    )
    parser.add_argument("--split", choices=("proposal", "validation", "test"), required=True)
    parser.add_argument("--passive-threshold", type=float, default=0.85)
    return parser.parse_args()


def _rate(rate: Rate) -> dict[str, int | float | tuple[float, float] | None]:
    return {
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "rate": rate.rate,
        "wilson_95": rate.wilson_95,
    }


def _render(summary: LivenessEvaluationSummary) -> dict[str, Any]:
    return {
        "trial_count": summary.trial_count,
        "bona_fide_count": summary.bona_fide_count,
        "active_completion": _rate(summary.completion_rate),
        "bona_fide_overall_rejection": _rate(summary.false_failure_rate),
        "active_timeout": _rate(summary.timeout_rate),
        "active_failure_reasons": summary.active_failure_reasons,
        "pad_bpcer": _rate(summary.bpcer_rate),
        "pad_apcer_by_attack": {
            attack: _rate(rate) for attack, rate in summary.apcer_by_attack.items()
        },
        "pad_acer_by_attack": summary.acer_by_attack,
        "failure_to_process": _rate(summary.failure_to_process_rate),
        "duration_ms": {
            "sample_count": summary.duration_ms.sample_count,
            "p50": summary.duration_ms.p50,
            "p95": summary.duration_ms.p95,
            "p99": summary.duration_ms.p99,
        },
        "interpretation": {
            "bpcer_denominator": "only bona-fide trials with a PAD score",
            "failure_to_process": "reported separately; it is not silently counted as PAD accuracy",
            "security_claim": "none; results require enough fresh validation/test trials",
        },
    }


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise SystemExit(f"input file does not exist: {args.input}")
    records: list[object] = []
    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON on line {number}") from exc
    trials = tuple(
        trial_from_record(record)
        for record in records
        if isinstance(record, dict) and record.get("split") == args.split
    )
    summary = summarize_liveness_trials(trials, passive_threshold=args.passive_threshold)
    print(json.dumps(_render(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
