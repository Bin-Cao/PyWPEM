#!/usr/bin/env python3
"""Normalize a parseable XRD text table to WPEM's two-column CSV format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_row(line: str) -> list[str]:
    """Parse a comma, tab, semicolon, or whitespace-delimited row."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", ";")):
        return []
    for delimiter in (",", "\t", ";"):
        if delimiter in stripped:
            return next(csv.reader([stripped], delimiter=delimiter))
    return stripped.split()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--angle-column", type=int, default=0, metavar="N")
    parser.add_argument("--intensity-column", type=int, default=1, metavar="N")
    args = parser.parse_args()

    if args.angle_column < 0 or args.intensity_column < 0:
        parser.error("column indexes must be non-negative")
    if args.angle_column == args.intensity_column:
        parser.error("angle and intensity columns must differ")
    if not args.source.is_file():
        parser.error(f"source file does not exist: {args.source}")
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must differ from source to preserve the original file")

    rows: list[tuple[float, float]] = []
    data_started = False
    highest_column = max(args.angle_column, args.intensity_column)
    try:
        lines = args.source.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        parser.error(f"source must be a UTF-8 text table: {exc}")
    except OSError as exc:
        parser.error(f"cannot read source: {exc}")

    for line_number, line in enumerate(lines, start=1):
        fields = parse_row(line)
        if not fields:
            continue
        if len(fields) <= highest_column:
            if data_started:
                parser.error(f"line {line_number} has too few columns")
            continue
        try:
            angle = float(fields[args.angle_column].strip())
            intensity = float(fields[args.intensity_column].strip())
        except ValueError:
            if data_started:
                parser.error(
                    f"line {line_number} has non-numeric selected columns; "
                    "choose the correct columns or clean the source"
                )
            continue
        data_started = True
        rows.append((angle, intensity))

    if len(rows) < 2:
        parser.error("fewer than two numeric XRD rows were found")
    previous_angle: float | None = None
    for line_number, (angle, intensity) in enumerate(rows, start=1):
        if intensity < 0:
            parser.error(f"data row {line_number} has negative intensity")
        if previous_angle is not None and angle <= previous_angle:
            parser.error(f"data row {line_number} has non-increasing 2theta")
        previous_angle = angle

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} XRD rows to {args.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
