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


CANONICAL = {role for role, _ in LABELS}


def format_counts(counts: Counter) -> str:
    width = max(2, max((len(str(v)) for v in counts.values()), default=2))
    rows = [(label, counts.get(role, 0)) for role, label in LABELS]
    # mini-SWE-agent v2 trajectories carry an extra `exit` message holding
    # the final patch; surface it (and any other non-canonical roles) instead
    # of dropping them silently from the total.
    for r in counts:
        if r not in CANONICAL:
            rows.append((f"{r.capitalize()} messages:", counts[r]))
    body = [f"{label:<19} {n:>{width}}" for label, n in rows]
    rule = "=" * len(body[0])
    total = sum(counts.values())
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
