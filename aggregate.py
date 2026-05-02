"""Run the feature extractor over every downloaded trajectory.

Joins per-instance leaderboard fields (resolved, cost, api_calls) where
available. GPT-5-2-Codex's row on swebench.com lacks per_instance_details,
so its `resolved` column comes back blank.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path

from features import ACTIONS, features_from_path

ROOT = Path(__file__).parent
TRAJ_DIR = ROOT / "trajectories"
META_PATH = ROOT / "data" / "leaderboard.json"
OUT_PATH = ROOT / "data" / "features.csv"

DISPLAY = {
    "20260217_mini-v2.0.0_claude-4-5-opus-high": "Claude 4.5 Opus high",
    "20260217_mini-v2.0.0_gemini-3-flash-high": "Gemini 3 Flash high",
    "20260217_mini-v2.0.0_minimax-2-5-high": "MiniMax M2.5 high",
    "20260217_mini-v2.0.0_claude-4-6-opus": "Claude 4.6 Opus",
    "20260219_mini-v2.0.0_gpt-5-2-codex": "GPT-5-2-Codex",
}

COLUMNS = [
    "model", "folder", "instance_id",
    "n_messages", "n_assistant", "n_tool", "n_calls",
    "n_read", "n_edit", "n_scratch_writes",
    "n_search", "n_test", "n_git", "n_install", "n_other",
    "n_repo_edits", "n_edit_files", "edit_churn", "call_unique_ratio",
    "patch_lines", "patch_files",
    "exit_status", "instance_cost", "api_calls",
    "resolved",
]


def per_instance(meta: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in meta:
        for iid, d in (row.get("per_instance_details") or {}).items():
            out[(row["folder"], iid)] = d
    return out


def main() -> int:
    meta = json.loads(META_PATH.read_text())
    leaderboard = per_instance(meta)

    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        rows = 0
        for folder, model in DISPLAY.items():
            folder_dir = TRAJ_DIR / folder
            files = sorted(folder_dir.glob("*.traj.json"))
            if not files:
                print(f"warn: no trajectories under {folder_dir}", file=sys.stderr)
                continue
            for path in files:
                tf = features_from_path(path)
                lb = leaderboard.get((folder, tf.instance_id), {})
                w.writerow({
                    "model": model,
                    "folder": folder,
                    "instance_id": tf.instance_id,
                    "n_messages": tf.n_messages,
                    "n_assistant": tf.n_assistant,
                    "n_tool": tf.n_tool,
                    "n_calls": tf.n_calls,
                    **{
                        ("n_scratch_writes" if a == "scratch_write" else f"n_{a}"):
                            tf.counts.get(a, 0)
                        for a in ACTIONS
                    },
                    "n_repo_edits": tf.n_repo_edits,
                    "n_edit_files": tf.n_edit_files,
                    "edit_churn": f"{tf.edit_churn:.4f}",
                    "call_unique_ratio": f"{tf.call_unique_ratio:.4f}",
                    "patch_lines": tf.patch_lines,
                    "patch_files": tf.patch_files,
                    "exit_status": tf.exit_status or "",
                    "instance_cost": tf.instance_cost if tf.instance_cost is not None else "",
                    "api_calls": tf.api_calls if tf.api_calls is not None else "",
                    "resolved": lb.get("resolved", ""),
                })
                rows += 1
        print(f"wrote {OUT_PATH} ({rows} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
