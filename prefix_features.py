"""Per-trajectory features computed from only the first k assistant turns.

This is the honest version of the in-flight signal. The original
features.py reads the whole trajectory; here we truncate to a prefix and
ask: from what's visible by turn k, can we predict the eventual outcome?

Patch shape and patch similarity are not available mid-flight (no
submission yet), so they are dropped from the feature set.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from features import _is_repo_path, classify, edit_target, role_of, ACTIONS

ROOT = Path(__file__).parent
TRAJ_DIR = ROOT / "trajectories"
META_PATH = ROOT / "data" / "leaderboard.json"
OUT_PATH = ROOT / "data" / "prefix_features.csv"

DISPLAY = {
    "20260217_mini-v2.0.0_claude-4-5-opus-high": "Claude 4.5 Opus high",
    "20260217_mini-v2.0.0_gemini-3-flash-high": "Gemini 3 Flash high",
    "20260217_mini-v2.0.0_minimax-2-5-high": "MiniMax M2.5 high",
    "20260217_mini-v2.0.0_claude-4-6-opus": "Claude 4.6 Opus",
    "20260219_mini-v2.0.0_gpt-5-2-codex": "GPT-5-2-Codex",
}

# Checkpoints in assistant-turn space.
PREFIX_KS = [5, 10, 20, 30, 50, 75]


def _call_args(t: dict) -> str:
    args = (t.get("function") or {}).get("arguments") or t.get("arguments") or "{}"
    try:
        d = json.loads(args)
    except json.JSONDecodeError:
        return ""
    cmd = d.get("command") or d.get("cmd") or ""
    return " ".join(cmd) if isinstance(cmd, list) else str(cmd)


def assistant_turns(trajectory: dict):
    """Walk the trajectory and yield, for each assistant turn, the list of
    bash commands issued in that turn. Standard schema = the assistant msg's
    tool_calls; codex schema = the function_call entries on a response item.
    """
    for msg in trajectory["messages"]:
        if msg.get("role") == "assistant":
            cmds = [_call_args(t) for t in (msg.get("tool_calls") or [])]
            yield cmds
        elif msg.get("object") == "response" or msg.get("type") == "response":
            cmds = [
                _call_args(o) for o in (msg.get("output") or [])
                if o.get("type") == "function_call"
            ]
            yield cmds


def features_at_prefix(turns: list[list[str]]) -> dict:
    """Aggregate features from the first len(turns) assistant turns."""
    counts = Counter()
    edited_files: dict[str, int] = {}
    repo_edits = 0
    repo_re_edits = 0
    n_calls = 0
    cmd_set: set[str] = set()
    for cmds in turns:
        for c in cmds:
            n_calls += 1
            cmd_set.add(c.strip())
            a = classify(c)
            counts[a] += 1
            if a != "edit":
                continue
            tgt = edit_target(c) or ""
            if not _is_repo_path(tgt):
                continue
            repo_edits += 1
            if tgt in edited_files:
                repo_re_edits += 1
            edited_files[tgt] = edited_files.get(tgt, 0) + 1
    edit_churn = repo_re_edits / repo_edits if repo_edits else 0.0
    call_unique_ratio = (len(cmd_set) / n_calls) if n_calls else 1.0
    out = {
        "n_assistant_prefix": len(turns),
        "n_calls_prefix": n_calls,
        "n_repo_edits_prefix": repo_edits,
        "n_edit_files_prefix": len(edited_files),
        "edit_churn_prefix": edit_churn,
        "call_unique_ratio_prefix": call_unique_ratio,
    }
    out.update({f"n_{a}_prefix": counts.get(a, 0) for a in ACTIONS})
    return out


def main() -> int:
    meta = json.loads(META_PATH.read_text())
    leaderboard: dict[tuple[str, str], dict] = {}
    for row in meta:
        for iid, d in (row.get("per_instance_details") or {}).items():
            leaderboard[(row["folder"], iid)] = d

    fieldnames = ["model", "folder", "instance_id", "k", "reached", "resolved"]
    sample_keys = features_at_prefix([[]]).keys()
    fieldnames += list(sample_keys)

    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n_rows = 0
        for folder, model in DISPLAY.items():
            for path in sorted((TRAJ_DIR / folder).glob("*.traj.json")):
                instance_id = path.name.replace(".traj.json", "")
                traj = json.loads(path.read_text())
                turns = list(assistant_turns(traj))
                lb = leaderboard.get((folder, instance_id), {})
                resolved = lb.get("resolved", "")
                for k in PREFIX_KS:
                    reached = len(turns) >= k
                    feats = features_at_prefix(turns[:k])
                    w.writerow({
                        "model": model,
                        "folder": folder,
                        "instance_id": instance_id,
                        "k": k,
                        "reached": int(reached),
                        "resolved": resolved,
                        **feats,
                    })
                    n_rows += 1
        print(f"wrote {OUT_PATH} ({n_rows} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
