"""Pull each model's submitted patch out of its trajectories and save them
to disk so the GPU embedder can read them flat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "trajectories"
DST = ROOT / "data" / "predicted"

FOLDERS = {
    "20260217_mini-v2.0.0_claude-4-5-opus-high": "claude-4-5-opus-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high": "gemini-3-flash-high",
    "20260217_mini-v2.0.0_minimax-2-5-high": "minimax-2-5-high",
    "20260217_mini-v2.0.0_claude-4-6-opus": "claude-4-6-opus",
    "20260219_mini-v2.0.0_gpt-5-2-codex": "gpt-5-2-codex",
}


def main() -> int:
    for folder, slug in FOLDERS.items():
        out_dir = DST / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = n_empty = 0
        for path in sorted((SRC / folder).glob("*.traj.json")):
            traj = json.loads(path.read_text())
            sub = (traj.get("info") or {}).get("submission") or ""
            iid = path.name.replace(".traj.json", "")
            (out_dir / f"{iid}.patch").write_text(sub)
            if sub.strip():
                n_written += 1
            else:
                n_empty += 1
        print(f"{slug}: {n_written} patches, {n_empty} empty", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
