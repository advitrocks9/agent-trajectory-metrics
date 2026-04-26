#!/usr/bin/env python3
"""Count messages by role in a mini-SWE-agent v2 trajectory."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def count_messages(trajectory: dict) -> Counter:
    return Counter(m["role"] for m in trajectory["messages"])


LABELS = [
    ("system", "System messages:"),
    ("user", "User messages:"),
    ("assistant", "Assistant messages:"),
    ("tool", "Tool messages:"),
]


def format_counts(counts: Counter) -> str:
    width = max(2, max((len(str(v)) for v in counts.values()), default=2))
    rows = [(label, counts.get(role, 0)) for role, label in LABELS]
    body = [f"{label:<19} {n:>{width}}" for label, n in rows]
    rule = "=" * len(body[0])
    total = sum(counts.get(r, 0) for r, _ in LABELS)
    return "\n".join(body + [rule, f"Total messages:     {total:>{width}}"])


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"usage: {argv[0]} [trajectory.json]", file=sys.stderr)
        return 2
    text = (
        Path(argv[1]).read_text()
        if len(argv) == 2 and argv[1] != "-"
        else sys.stdin.read()
    )
    print(format_counts(count_messages(json.loads(text))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
