"""
Derive the policy rate path, and named policy eras, from the statements.

Queries like "during the hiking cycle" need date bounds, but those bounds are a
fact about what the FOMC did -- so they are read out of the corpus rather than
hardcoded. Every policy statement names the target range it set, e.g.

    "the Committee decided to raise the target range for the federal funds
     rate to 4-1/4 to 4-1/2 percent"

so the whole path, and the turning points in it, fall out of parsing that one
sentence per meeting.

Usage:
    python rates.py            # print the derived rate path and eras
"""

import argparse
import collections
import json
import pathlib
import re

CHUNKS_PATH = pathlib.Path(__file__).resolve().parent / "data/chunks.jsonl"

# The range is stated as two fractions: "4-1/4 to 4-1/2 percent". Some
# statements put "by 1/2 percentage point" in between, so the bound is found
# with a non-greedy gap rather than an immediate match.
TARGET_RANGE = re.compile(
    r"target range for the federal funds rate(?:[^.]*?)"
    r"(?:at|to)\s+(\d[\d\-/]*)\s+to\s+(\d[\d\-/]*)\s+percent",
    re.IGNORECASE,
)

HIKE, CUT, HOLD = "hiking", "cutting", "holding"


def parse_fraction(token):
    """'4-1/4' -> 4.25, '1/4' -> 0.25, '5' -> 5.0."""
    token = token.strip()
    whole, _, fraction = token.partition("-")
    if "/" in whole and not fraction:          # a bare fraction like "1/4"
        numerator, _, denominator = whole.partition("/")
        return int(numerator) / int(denominator)
    total = float(whole)
    if fraction:
        numerator, _, denominator = fraction.partition("/")
        total += int(numerator) / int(denominator)
    return total


def parse_target_range(text):
    """Return (low, high) of the target range a statement sets, or None."""
    match = TARGET_RANGE.search(text)
    if not match:
        return None
    try:
        return parse_fraction(match.group(1)), parse_fraction(match.group(2))
    except (ValueError, ZeroDivisionError):
        return None


def rate_path(chunks):
    """meeting_date -> midpoint of the target range, oldest first."""
    path = {}
    for chunk in sorted(chunks, key=lambda c: c["meeting_date"]):
        if chunk["doc_type"] != "statement" or chunk["meeting_date"] in path:
            continue
        bounds = parse_target_range(chunk["text"])
        if bounds:
            path[chunk["meeting_date"]] = round(sum(bounds) / 2, 4)
    return collections.OrderedDict(sorted(path.items()))


def moves(path):
    """Per meeting, the direction of the change from the meeting before it."""
    dates = list(path)
    out = []
    for previous, current in zip(dates, dates[1:]):
        delta = path[current] - path[previous]
        out.append((current, HIKE if delta > 0 else CUT if delta < 0 else HOLD, delta))
    return out


def eras(path):
    """
    Group the path into hiking, cutting and holding runs.

    A cycle runs from the meeting that made the first move in a direction to the
    meeting that made the last one before the direction reversed; intervening
    holds stay inside the cycle. A run of holds between cycles is its own era.
    """
    changes = moves(path)
    if not changes:
        return []

    out = []
    current_kind, start, end = None, None, None

    for date, kind, _ in changes:
        if kind == HOLD:
            if current_kind in (HIKE, CUT):
                continue                      # a pause inside a cycle
            if current_kind == HOLD:
                end = date
            else:
                current_kind, start, end = HOLD, date, date
            continue

        if kind == current_kind:
            end = date
            continue

        if current_kind is not None:
            out.append({"kind": current_kind, "start": start, "end": end})
        current_kind, start, end = kind, date, date

    if current_kind is not None:
        out.append({"kind": current_kind, "start": start, "end": end})
    return out


def load_path(path=CHUNKS_PATH):
    with path.open(encoding="utf-8") as handle:
        return rate_path([json.loads(line) for line in handle if line.strip()])


def named_eras(path):
    """
    Map query phrases to date windows, derived from the path above.

    Only the most recent cycle of each kind is named, which is what a phrase
    like "the hiking cycle" means in a question asked today.
    """
    found = eras(path)
    names = {}
    for kind, labels in (
        (HIKE, ["hiking cycle", "hiking campaign", "tightening cycle",
                "rate hikes", "hiking"]),
        (CUT, ["cutting cycle", "easing cycle", "rate cuts", "cutting"]),
    ):
        matching = [e for e in found if e["kind"] == kind]
        if not matching:
            continue
        latest = matching[-1]
        for label in labels:
            names[label] = (latest["start"], latest["end"])
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=pathlib.Path, default=CHUNKS_PATH)
    args = parser.parse_args()

    path = load_path(args.chunks)
    print(f"Target range midpoint, parsed from {len(path)} policy statements\n")
    previous = None
    for date, midpoint in path.items():
        marker = ""
        if previous is not None:
            delta = midpoint - previous
            marker = f"  {'+' if delta > 0 else ''}{delta:.2f}" if delta else "  --"
        print(f"  {date}   {midpoint:5.3f}%{marker}")
        previous = midpoint

    print("\nPolicy eras derived from that path\n")
    for era in eras(path):
        print(f"  {era['kind']:<8} {era['start']} .. {era['end']}")

    print("\nPhrases a query may use\n")
    for label, (start, end) in sorted(named_eras(path).items()):
        print(f"  {label:<20} {start} .. {end}")


if __name__ == "__main__":
    main()
