"""For every (model, instance_id), compute cosine similarity between the
agent's submitted patch and the SWE-bench Verified ground-truth patch.

Reads embeddings.npy + embeddings_index.csv produced by embed_remote.py
(the GPU-side script), writes a `patch_sim` column joined into
data/features.csv.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SLUG = {
    "Claude 4.5 Opus high": "claude-4-5-opus-high",
    "Gemini 3 Flash high": "gemini-3-flash-high",
    "MiniMax M2.5 high": "minimax-2-5-high",
    "Claude 4.6 Opus": "claude-4-6-opus",
    "GPT-5-2-Codex": "gpt-5-2-codex",
}


def main() -> int:
    emb = np.load(DATA / "embeddings.npy")
    rows = list(csv.DictReader((DATA / "embeddings_index.csv").open()))
    print(f"loaded {emb.shape}, {len(rows)} index rows", file=sys.stderr)

    # Build (group, instance_id) -> embedding row
    idx: dict[tuple[str, str], int] = {(r["group"], r["instance_id"]): int(r["row"]) for r in rows}

    # Compute patch_sim per (model, instance_id) by cosine sim against ground truth
    sims: dict[tuple[str, str], float] = {}
    missing = defaultdict(int)
    for (group, iid), row in idx.items():
        if group == "ground_truth":
            continue
        gt_row = idx.get(("ground_truth", iid))
        if gt_row is None:
            missing[group] += 1
            continue
        # Both vectors are L2-normalised by the embedder, so cosine = dot.
        sims[(group, iid)] = float(np.dot(emb[row], emb[gt_row]))
    if missing:
        print(f"missing ground-truth for: {dict(missing)}", file=sys.stderr)

    # Read features.csv, append patch_sim, write to a new file
    features_path = DATA / "features.csv"
    out_path = DATA / "features_with_sim.csv"
    with features_path.open() as fin, out_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        cols = reader.fieldnames + ["patch_sim"]
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()
        n_with = 0
        for r in reader:
            slug = SLUG.get(r["model"], "")
            sim = sims.get((slug, r["instance_id"]))
            r["patch_sim"] = f"{sim:.4f}" if sim is not None else ""
            n_with += sim is not None
            writer.writerow(r)
    print(f"wrote {out_path} ({n_with} rows with patch_sim)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
