#!/usr/bin/env python3
"""Count messages by role in a mini-SWE-agent-v2 trajectory.

Reads a single trajectory JSON from a path or stdin and prints role counts
in the layout the swebench.com task brief asks for, plus any non-canonical
roles the trajectory happens to contain (mini-SWE-agent v2 emits an `exit`
message holding the final patch).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def role_of(msg: dict) -> str:
    if "role" in msg:
        return msg["role"]
    # GPT-5-2-Codex submissions store raw OpenAI Responses API objects in
    # the messages array with no role field.
    if msg.get("object") == "response" or msg.get("type") == "response":
        return "assistant"
    if msg.get("type") == "function_call_output":
        return "tool"
    return "unknown"


CANONICAL = ("system", "user", "assistant", "tool")


def format_counts(counts: Counter) -> str:
    width = max(2, max((len(str(v)) for v in counts.values()), default=2))
    rows = [(f"{r.capitalize()} messages:", counts.get(r, 0)) for r in CANONICAL]
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
    counts = Counter(role_of(m) for m in json.loads(text)["messages"])
    print(format_counts(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
