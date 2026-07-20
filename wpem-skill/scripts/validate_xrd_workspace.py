#!/usr/bin/env python3
"""Validate the filesystem inputs required by a PyWPEM XRD refinement."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def validate_two_column_csv(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        return [f"cannot read {path}: {exc}"]
    if len(rows) < 2:
        return [f"{path} must contain at least two data rows"]
    previous_angle: float | None = None
    for number, row in enumerate(rows[:50], start=1):
        if len(row) < 2:
            errors.append(f"{path}:{number} has fewer than two columns")
            continue
        try:
            angle, intensity = float(row[0].strip()), float(row[1].strip())
        except ValueError:
            errors.append(f"{path}:{number} must contain numeric 2theta and intensity")
            continue
        if intensity < 0:
            errors.append(f"{path}:{number} has negative intensity")
        if previous_angle is not None and angle <= previous_angle:
            errors.append(f"{path}:{number} has non-increasing 2theta")
        previous_angle = angle
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--phases", type=int, required=True)
    parser.add_argument("--require-refinement", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    if args.phases < 1:
        errors.append("--phases must be at least 1")
    root = args.work_dir.expanduser().resolve()
    intensity = root / "intensity.csv"
    if not root.is_dir():
        errors.append(f"workspace does not exist: {root}")
    elif not intensity.is_file():
        errors.append(f"missing experimental pattern: {intensity}")
    else:
        errors.extend(validate_two_column_csv(intensity))

    if args.require_refinement and root.is_dir():
        for relative in ("ConvertedDocuments/no_bac_intensity.csv", "ConvertedDocuments/bac.csv"):
            path = root / relative
            if not path.is_file():
                errors.append(f"missing background-processing output: {path}")
            else:
                errors.extend(validate_two_column_csv(path))
        for phase in range(args.phases):
            peak = root / f"peak{phase}.csv"
            if not peak.is_file():
                errors.append(f"missing initial peak file for phase {phase}: {peak}")
            elif "2theta/TOF" not in peak.read_text(errors="replace").splitlines()[0]:
                errors.append(f"{peak} header must contain '2theta/TOF'")

    if errors:
        print("XRD workspace preflight failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"XRD workspace preflight passed: {root} ({args.phases} phase(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
