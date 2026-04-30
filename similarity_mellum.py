"""Add a `patch_sim_mellum` column from the Mellum embeddings.

Same shape as similarity.py but reads embeddings_mellum.npy /
embeddings_mellum_index.csv. Reads features_with_sim.csv (Qwen-augmented)
and writes features_with_both_sim.csv.
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
    emb = np.load(DATA / "embeddings_mellum.npy")
    rows = list(csv.DictReader((DATA / "embeddings_mellum_index.csv").open()))
    print(f"loaded mellum embeddings {emb.shape}", file=sys.stderr)

    idx = {(r["group"], r["instance_id"]): int(r["row"]) for r in rows}
    sims: dict[tuple[str, str], float] = {}
    missing = defaultdict(int)
    for (group, iid), row in idx.items():
        if group == "ground_truth":
            continue
        gt = idx.get(("ground_truth", iid))
        if gt is None:
            missing[group] += 1
            continue
        sims[(group, iid)] = float(np.dot(emb[row], emb[gt]))
    if missing:
        print(f"missing ground-truth for: {dict(missing)}", file=sys.stderr)

    src = DATA / "features_with_sim.csv"
    dst = DATA / "features_with_both_sim.csv"
    with src.open() as fin, dst.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        cols = reader.fieldnames + ["patch_sim_mellum"]
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()
        n_with = 0
        for r in reader:
            slug = SLUG.get(r["model"], "")
            sim = sims.get((slug, r["instance_id"]))
            r["patch_sim_mellum"] = f"{sim:.4f}" if sim is not None else ""
            n_with += sim is not None
            writer.writerow(r)
    print(f"wrote {dst} ({n_with} with patch_sim_mellum)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
