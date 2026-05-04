"""Cheap textual baseline for patch_sim.

Jaccard overlap of *changed* lines between the predicted patch and the
ground-truth patch. No LM, no GPU; just diff parsing.

The point of this baseline is to ask: how much of the embedding-based
patch-similarity signal is the LM doing real semantic work, vs how much
is just "the predicted diff and the ground-truth diff share lines"?

Each entry in the set is keyed by `(file, op, payload)`. The first version
of this script pooled adds and deletes into the same string set, so a
deletion of `import x` and an addition of `import x` would collapse into
one element and inflate the score. Cross-file payload collisions
(boilerplate lines like `from __future__ import annotations`) had the
same effect. Keeping the file path and the +/- op separate kills both
classes of false match.

Adds a `patch_overlap` column to features_with_both_sim.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
GT_DIR = DATA / "patches"
PRED_ROOT = DATA / "predicted"

SLUG = {
    "Claude 4.5 Opus high": "claude-4-5-opus-high",
    "Gemini 3 Flash high": "gemini-3-flash-high",
    "MiniMax M2.5 high": "minimax-2-5-high",
    "Claude 4.6 Opus": "claude-4-6-opus",
    "GPT-5-2-Codex": "gpt-5-2-codex",
}


def changed_lines(diff_text: str) -> set[tuple[str, str, str]]:
    """`(file, op, payload)` triples for every +/- line in a unified diff.

    Tracks the current `+++ b/<path>` so cross-file payload collisions
    don't inflate the Jaccard. Keeps `+` and `-` distinct so a deletion
    and an addition of the same line don't collapse together. Leading
    indentation is preserved in the payload (only trailing whitespace
    is stripped) -- in Python, indentation is semantic, so a predicted
    line at the wrong indentation is not the same line.
    """
    out: set[tuple[str, str, str]] = set()
    cur_file = ""
    for ln in diff_text.splitlines():
        if ln.startswith("+++ "):
            # `+++ b/path/to/file.py` -> `path/to/file.py`
            rest = ln[4:].strip()
            cur_file = rest[2:] if rest.startswith(("a/", "b/")) else rest
            continue
        if ln.startswith("--- ") or ln.startswith("@@"):
            continue
        if not ln or ln[0] not in "+-":
            continue
        body = ln[1:].rstrip()
        if not body.strip():
            # Pure-whitespace edits don't carry information.
            continue
        out.add((cur_file, ln[0], body))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    src = DATA / "features_with_both_sim.csv"
    dst = DATA / "features_with_both_sim.csv.tmp"
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(src.open()))
    n_with = n_skipped = 0
    for r in rows:
        slug = SLUG.get(r["model"])
        gt_path = GT_DIR / f"{r['instance_id']}.patch"
        pred_path = PRED_ROOT / slug / f"{r['instance_id']}.patch" if slug else None
        if pred_path is None or not gt_path.exists() or not pred_path.exists():
            r["patch_overlap"] = ""
            n_skipped += 1
            continue
        a = changed_lines(gt_path.read_text())
        b = changed_lines(pred_path.read_text())
        r["patch_overlap"] = f"{jaccard(a, b):.4f}"
        n_with += 1

    cols = list(rows[0].keys())
    if "patch_overlap" not in cols:
        cols.append("patch_overlap")
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    dst.replace(src)
    print(f"updated {src}: {n_with} with patch_overlap, {n_skipped} skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
