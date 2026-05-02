"""Compare feature subsets in the LOMO classifier, with bootstrap CIs.

Asks: how much does each information source add to the AUC, and how big
are the error bars?

  base       -> trajectory-shape features only (no patch features at all)
  +shape     -> base + final patch line/file count
  +overlap   -> +shape + Jaccard similarity over patch hunk lines (no LM)
  +qwen      -> +shape + cosine with Qwen2.5-Coder-1.5B embeddings
  +mellum    -> +shape + cosine with JetBrains/Mellum-4b-sft-python
  +both LMs  -> +shape + Qwen sim + Mellum sim

For each subset I report mean LOMO AUC across 4 folds plus the 95% CI
from a bootstrap of the test-set predictions within each fold (1,000
resamples per fold, percentile method).
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

THRASH_FEATURES = ["thrash_depth", "thrash_at_10", "repeat_rate", "window10_max"]
PATCH_SHAPE_FEATURES = ["patch_lines", "patch_files"]

SUBSETS = {
    "base":      SHAPE_FEATURES,
    "+thrash":   SHAPE_FEATURES + THRASH_FEATURES,
    "+shape":    SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES,
    "+overlap":  SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES + ["patch_overlap"],
    "+qwen":     SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim"],
    "+mellum":   SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim_mellum"],
    "+both LMs": SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES + ["patch_sim", "patch_sim_mellum"],
    "+all":      SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES + ["patch_overlap", "patch_sim", "patch_sim_mellum"],
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


def lomo_auc(
    rows: list[dict], features: list[str], n_boot: int = 1000, seed: int = 1
) -> tuple[float, tuple[float, float], list[float]]:
    """Mean LOMO AUC + 95% bootstrap CI on the pooled out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    fold_aucs: list[float] = []
    pooled_y: list[int] = []
    pooled_p: list[float] = []
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
        fold_aucs.append(roc_auc_score(yte, proba))
        pooled_y.extend(yte.tolist())
        pooled_p.extend(proba.tolist())
    pooled_y_arr = np.array(pooled_y)
    pooled_p_arr = np.array(pooled_p)
    n = len(pooled_y_arr)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = roc_auc_score(pooled_y_arr[idx], pooled_p_arr[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(np.mean(fold_aucs)), (float(lo), float(hi)), fold_aucs


def main() -> int:
    csv_path = DATA / "features_with_both_sim.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}; run similarity_mellum.py first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r["model"] in LABELLED and r["resolved"] in {"True", "False"}]

    print(f"labelled rows: {len(rows)}")
    print()
    print(f"{'Feature subset':<14}  {'mean AUC':>9}  {'95% CI':>14}   per-fold (Opus5h, Gemini, MiniMax, Opus6)")
    print("-" * 92)
    rows_for_overlap = [r for r in rows if "patch_overlap" in r]
    if not rows_for_overlap:
        print("  (run patch_overlap.py first to populate the overlap column)")
        return 1
    for name, feats in SUBSETS.items():
        mean, (lo, hi), aucs = lomo_auc(rows, feats)
        print(f"  {name:<12}  {mean:>9.3f}  [{lo:.3f}, {hi:.3f}]   {' '.join(f'{a:.3f}' for a in aucs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
