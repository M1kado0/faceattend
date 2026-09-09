"""Summarize non-identifying verification-score records from one local split.

Each JSONL record must contain ``split``, ``score``, ``is_genuine``, and
``condition``. Do not put person names, embedding values, or image paths in this
file. The script only reports aggregate verification metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from faceattend.evaluation.recognition import (  # noqa: E402
    summarize_verification_scores,
    verification_score_from_record,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("proposal", "validation", "test"))
    parser.add_argument("--threshold", required=True, type=float)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise SystemExit(f"input file does not exist: {args.input}")
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = tuple(
        verification_score_from_record(record)
        for record in records
        if isinstance(record, dict) and record.get("split") == args.split
    )
    summary = summarize_verification_scores(scores, threshold=args.threshold)
    print(
        json.dumps(
            {
                "split": args.split,
                "genuine_count": summary.genuine_count,
                "impostor_count": summary.impostor_count,
                "fmr": {
                    "numerator": summary.fmr.numerator,
                    "denominator": summary.fmr.denominator,
                    "rate": summary.fmr.rate,
                },
                "fnmr": {
                    "numerator": summary.fnmr.numerator,
                    "denominator": summary.fnmr.denominator,
                    "rate": summary.fnmr.rate,
                },
                "eer_discrete_approximation": summary.eer,
                "condition_counts": summary.condition_counts,
                "warning": (
                    "score-only aggregate; threshold selection belongs to validation, not test"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
