"""Compare feature subsets in the LOMO classifier.

Asks: how much does each information source add to the AUC?

  base    -> trajectory-shape features only (no patch features at all)
  +shape  -> base + final patch line/file count
  +qwen   -> +shape + cosine similarity with Qwen2.5-Coder-1.5B embeddings
  +mellum -> +shape + cosine with JetBrains/Mellum-4b-sft-python
  +both   -> +shape + Qwen sim + Mellum sim

Same logistic regression, same leave-one-model-out folds.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"

LABELLED = [
    "Claude 4.5 Opus high",
    "Gemini 3 Flash high",
    "MiniMax M2.5 high",
    "Claude 4.6 Opus",
]

SHAPE_FEATURES = [
    "n_calls",
    "n_read", "n_edit", "n_search", "n_test", "n_git", "n_install",
    "n_repo_edits", "n_edit_files", "edit_churn", "call_unique_ratio",
]

PATCH_SHAPE_FEATURES = ["patch_lines", "patch_files"]

SUBSETS = {
    "base":    SHAPE_FEATURES,
    "+shape":  SHAPE_FEATURES + PATCH_SHAPE_FEATURES,
    "+qwen":   SHAPE_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim"],
    "+mellum": SHAPE_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim_mellum"],
    "+both":   SHAPE_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim", "patch_sim_mellum"],
}


def to_xy(rows: list[dict], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(rows), len(features)), dtype=np.float64)
    y = np.empty(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        for j, f in enumerate(features):
            v = r.get(f, "")
            X[i, j] = float(v) if v not in ("", None) else np.nan
        y[i] = 1 if r["resolved"] == "True" else 0
    return X, y


def lomo_auc(rows: list[dict], features: list[str]) -> tuple[float, list[float]]:
    aucs = []
    for held in LABELLED:
        train = [r for r in rows if r["model"] != held]
        test = [r for r in rows if r["model"] == held]
        Xtr, ytr = to_xy(train, features)
        Xte, yte = to_xy(test, features)
        mu = np.nanmean(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr)
        Xte = np.where(np.isnan(Xte), mu, Xte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(yte, proba))
    return float(np.mean(aucs)), aucs


def main() -> int:
    csv_path = DATA / "features_with_both_sim.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}; run similarity_mellum.py first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r["model"] in LABELLED and r["resolved"] in {"True", "False"}]

    print(f"labelled rows: {len(rows)}")
    print()
    print("Feature subset            mean AUC   per-fold (Opus5h, Gemini, MiniMax, Opus6)")
    print("-" * 76)
    for name, feats in SUBSETS.items():
        mean, aucs = lomo_auc(rows, feats)
        print(f"  {name:<22}  {mean:>7.3f}    {' '.join(f'{a:.3f}' for a in aucs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
