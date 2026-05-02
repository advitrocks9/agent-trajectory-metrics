"""Thrash-depth feature: longest streak of identical non-edit commands.

A thrash is the agent re-running the same read or test (or grep, or git
status) without changing the working tree in between. It is the cleanest
"stuck" signal in a bash trajectory: the agent has new information from
the last command and is re-asking the same question.

Implementation:

  thrash_depth(t) = max k s.t. the trajectory contains k consecutive
                    identical non-edit bash commands

Identity is exact-string after stripping leading/trailing whitespace; an
intervening edit (sed -i, heredoc redirect, apply_patch, ...) resets the
streak. Read / search / test / git / install / other commands all count
toward streaks; only edit breaks them.

Adds `thrash_depth` and `thrash_at_10` (depth restricted to the first 10
non-edit commands, for in-flight use) as new feature columns.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from features import classify
from prefix_features import assistant_turns

ROOT = Path(__file__).parent
TRAJ_DIR = ROOT / "trajectories"
DATA = ROOT / "data"

DISPLAY = {
    "20260217_mini-v2.0.0_claude-4-5-opus-high": "Claude 4.5 Opus high",
    "20260217_mini-v2.0.0_gemini-3-flash-high": "Gemini 3 Flash high",
    "20260217_mini-v2.0.0_minimax-2-5-high": "MiniMax M2.5 high",
    "20260217_mini-v2.0.0_claude-4-6-opus": "Claude 4.6 Opus",
    "20260219_mini-v2.0.0_gpt-5-2-codex": "GPT-5-2-Codex",
}


def thrash_depth(commands: list[str], early_cap: int | None = None) -> int:
    """Longest consecutive-identical run, edits reset the streak."""
    max_run = 0
    run = 1
    prev = None
    seen_non_edit = 0
    for cmd in commands:
        a = classify(cmd)
        if a == "edit":
            prev = None
            run = 1
            continue
        seen_non_edit += 1
        if early_cap is not None and seen_non_edit > early_cap:
            break
        c = cmd.strip()
        if prev == c and c:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
        prev = c
    return max_run


def repeat_rate(commands: list[str]) -> float:
    """Fraction of non-edit commands that exactly repeat an earlier non-edit
    command. Captures interleaved cycles that the strict run-length misses
    (e.g., `cat`, `python`, `cat`, `python`, ...).
    """
    non_edits = [c.strip() for c in commands if classify(c) != "edit" and c.strip()]
    if len(non_edits) < 2:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for c in non_edits:
        if c in seen:
            repeats += 1
        seen.add(c)
    return repeats / len(non_edits)


def max_window_repeat(commands: list[str], window: int = 10) -> int:
    """Across all length-`window` windows of non-edit commands, the maximum
    count of any single command. Higher = more local repetition.
    """
    non_edits = [c.strip() for c in commands if classify(c) != "edit" and c.strip()]
    if len(non_edits) < window:
        if len(non_edits) < 2:
            return 0
        return Counter(non_edits).most_common(1)[0][1]
    best = 0
    for i in range(len(non_edits) - window + 1):
        c = Counter(non_edits[i:i + window])
        best = max(best, c.most_common(1)[0][1])
    return best


def trajectory_commands(traj: dict) -> list[str]:
    cmds = []
    for turn_cmds in assistant_turns(traj):
        cmds.extend(turn_cmds)
    return cmds


def main() -> int:
    src = DATA / "features_with_both_sim.csv"
    rows = list(csv.DictReader(src.open()))
    by_inst: dict[tuple[str, str], dict] = {(r["model"], r["instance_id"]): r for r in rows}

    n = 0
    for folder, model in DISPLAY.items():
        for path in sorted((TRAJ_DIR / folder).glob("*.traj.json")):
            iid = path.name.replace(".traj.json", "")
            traj = json.loads(path.read_text())
            cmds = trajectory_commands(traj)
            r = by_inst.get((model, iid))
            if r is None:
                continue
            r["thrash_depth"] = str(thrash_depth(cmds))
            r["thrash_at_10"] = str(thrash_depth(cmds, early_cap=10))
            r["repeat_rate"] = f"{repeat_rate(cmds):.4f}"
            r["window10_max"] = str(max_window_repeat(cmds, window=10))
            n += 1

    cols = list(rows[0].keys())
    for c in ("thrash_depth", "thrash_at_10", "repeat_rate", "window10_max"):
        if c not in cols:
            cols.append(c)
    with src.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"updated {src}: thrash_depth and thrash_at_10 added for {n} rows", file=sys.stderr)

    print()
    print("Thrash depth distribution by model")
    print("=" * 64)
    print(f"  {'Model':<22} {'mean':>6} {'med':>5} {'p90':>5} {'max':>5}")
    by_model: dict[str, list[int]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(int(r["thrash_depth"]))
    for m, ds in by_model.items():
        ds_s = sorted(ds)
        print(f"  {m:<22} {sum(ds)/len(ds):>6.2f} {ds_s[len(ds)//2]:>5} {ds_s[int(0.9*(len(ds)-1))]:>5} {ds_s[-1]:>5}")

    print()
    print("repeat_rate vs resolved (fraction of non-edit commands that repeat earlier)")
    print("=" * 78)
    print(f"  {'Model':<22} {'mean (res)':>12} {'mean (unres)':>14} {'gap':>6}")
    for m in ("Claude 4.5 Opus high", "Gemini 3 Flash high", "MiniMax M2.5 high", "Claude 4.6 Opus"):
        rs = [r for r in rows if r["model"] == m and r["resolved"] in {"True", "False"}]
        d_r = [float(r["repeat_rate"]) for r in rs if r["resolved"] == "True"]
        d_u = [float(r["repeat_rate"]) for r in rs if r["resolved"] == "False"]
        gap = sum(d_u) / len(d_u) - sum(d_r) / len(d_r)
        print(f"  {m:<22} {sum(d_r)/len(d_r):>12.3f} {sum(d_u)/len(d_u):>14.3f} {gap:>+6.3f}")

    print()
    print("window10_max vs resolved (max repeat count in any 10-cmd window)")
    print("=" * 78)
    print(f"  {'Model':<22} {'mean (res)':>12} {'mean (unres)':>14} {'gap':>6}")
    for m in ("Claude 4.5 Opus high", "Gemini 3 Flash high", "MiniMax M2.5 high", "Claude 4.6 Opus"):
        rs = [r for r in rows if r["model"] == m and r["resolved"] in {"True", "False"}]
        d_r = [int(r["window10_max"]) for r in rs if r["resolved"] == "True"]
        d_u = [int(r["window10_max"]) for r in rs if r["resolved"] == "False"]
        gap = sum(d_u) / len(d_u) - sum(d_r) / len(d_r)
        print(f"  {m:<22} {sum(d_r)/len(d_r):>12.2f} {sum(d_u)/len(d_u):>14.2f} {gap:>+6.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
