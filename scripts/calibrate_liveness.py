"""Print validation-only PAD threshold candidates from ignored trial JSONL.

This script never changes ``FACEATTEND_PASSIVE_THRESHOLD`` or writes project
configuration. Selecting a threshold requires review of denominators, attack
coverage, and a separate held-out test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from faceattend.evaluation.liveness import (  # noqa: E402
    passive_threshold_candidates,
    trial_from_record,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Ignored local JSONL metrics file."
    )
    parser.add_argument("--split", choices=("validation",), default="validation")
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
    trials = tuple(
        trial_from_record(record)
        for record in records
        if isinstance(record, dict) and record.get("split") == args.split
    )
    candidates = passive_threshold_candidates(trials)
    print(
        json.dumps(
            {
                "split": args.split,
                "trial_count": len(trials),
                "candidates": [
                    {
                        "threshold": candidate.threshold,
                        "bpcer": {
                            "numerator": candidate.bpcer.numerator,
                            "denominator": candidate.bpcer.denominator,
                            "rate": candidate.bpcer.rate,
                            "wilson_95": candidate.bpcer.wilson_95,
                        },
                        "mean_apcer": candidate.mean_apcer,
                    }
                    for candidate in candidates
                ],
                "warning": "review only; this command does not freeze or apply a threshold",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    main()
