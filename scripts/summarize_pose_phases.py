"""Summarize labeled MediaPipe pose captures and print threshold candidates.

    uv run python scripts/summarize_pose_phases.py pose.csv

Thresholds are candidates for a later experiment, not automatic policy.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    if not args.csv_path.is_file():
        print(f"ERROR: CSV not found: {args.csv_path}")
        return 1

    samples: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with args.csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            samples[row["phase"]].append(
                (float(row["yaw"]), float(row["pitch"]), float(row["roll"]))
            )
    if not samples:
        print("ERROR: CSV contains no pose samples")
        return 1

    labels = ("yaw", "pitch", "roll")
    medians: dict[str, np.ndarray] = {}
    print("MediaPipe phase statistics (degrees):")
    for phase in sorted(samples):
        values = np.asarray(samples[phase], dtype=np.float64)
        medians[phase] = np.median(values, axis=0)
        print(
            f"{phase:>7}: "
            + ", ".join(
                f"{label}=median {medians[phase][index]:.2f}, "
                f"range {values[:, index].min():.2f}..{values[:, index].max():.2f}"
                for index, label in enumerate(labels)
            )
        )

    neutral = medians.get("neutral")
    if neutral is None:
        print("No neutral phase; cannot suggest directional thresholds.")
        return 0

    print("Candidate hysteresis thresholds (validate on new captures):")
    for phase, axis in (("left", 0), ("right", 0), ("up", 1), ("down", 1)):
        target = medians.get(phase)
        if target is None:
            continue
        distance = abs(target[axis] - neutral[axis])
        enter = distance * 0.65
        exit_threshold = distance * 0.45
        print(
            f"{phase:>7} {labels[axis]}: "
            f"enter={enter:.2f}°, exit={exit_threshold:.2f}°"
        )
    print("Candidate dwell: require at least 3 consecutive accepted frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
