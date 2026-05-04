"""Compare feature subsets in the LOMO classifier, with bootstrap CIs.

Asks: how much does each information source add to the AUC, and how big
are the error bars?

The baseline is `shape+thrash`: action counts, edit churn, and the four
thrash columns. Every "+X" row adds the named feature on top of the same
shape+thrash baseline, so each delta is the contribution of X above
shape+thrash, not above shape alone. (An earlier version of this script
silently rolled thrash into every row but labelled them as if thrash
weren't there, which over-credited the LM / Jaccard rows.)

  shape          -> action counts + edit churn + edit files (no thrash, no patch)
  shape+thrash   -> shape + thrash_depth, thrash_at_10, repeat_rate, window10_max
  shape+thrash+patch
                 -> shape+thrash + patch line/file count
  +overlap       -> shape+thrash+patch + Jaccard over (file, +/-, payload)
  +qwen          -> shape+thrash+patch + cosine to Qwen2.5-Coder-1.5B
  +mellum        -> shape+thrash+patch + cosine to Mellum-4b-sft-python
  +both LMs      -> shape+thrash+patch + Qwen sim + Mellum sim
  +all           -> shape+thrash+patch + overlap + Qwen + Mellum

For each subset I report mean LOMO AUC across 4 folds plus the 95% CI
from a bootstrap of the pooled out-of-fold predictions (1,000 resamples,
percentile method, resampled by `instance_id` so the 4 model rows for
each task move together). I also include a paired per-instance bootstrap
of the Mellum-vs-Qwen AUC delta because their CIs overlap and the
earlier "Mellum edges Qwen" framing rested on overlapping marginal CIs,
which doesn't actually establish a difference.
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

_BASE = SHAPE_FEATURES + THRASH_FEATURES + PATCH_SHAPE_FEATURES

SUBSETS = {
    "shape":              SHAPE_FEATURES,
    "shape+thrash":       SHAPE_FEATURES + THRASH_FEATURES,
    "shape+thrash+patch": _BASE,
    "+overlap":           _BASE + ["patch_overlap"],
    "+qwen":              _BASE + ["patch_sim"],
    "+mellum":            _BASE + ["patch_sim_mellum"],
    "+both LMs":          _BASE + ["patch_sim", "patch_sim_mellum"],
    "+all":               _BASE + ["patch_overlap", "patch_sim", "patch_sim_mellum"],
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


def pooled_predictions(
    rows: list[dict], features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Returns (y, p, instance_ids, per-fold AUCs). Predictions are
    out-of-fold and pooled in fold order."""
    fold_aucs: list[float] = []
    pooled_y: list[int] = []
    pooled_p: list[float] = []
    pooled_iids: list[str] = []
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
        pooled_iids.extend([r["instance_id"] for r in test])
    return np.array(pooled_y), np.array(pooled_p), np.array(pooled_iids), fold_aucs


def lomo_auc(
    rows: list[dict], features: list[str], n_boot: int = 1000, seed: int = 1
) -> tuple[float, tuple[float, float], list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Mean LOMO AUC + 95% bootstrap CI on the pooled out-of-fold predictions.

    The bootstrap resamples unique `instance_id` clusters (each contributes
    its 4 model rows together), mirroring `paired_delta_ci`. The earlier
    version sampled rows independently, which understates variance because
    same-instance rows across the 4 models share task difficulty.
    """
    y, p, iids, fold_aucs = pooled_predictions(rows, features)
    rng = np.random.default_rng(seed)
    by_iid: dict[str, list[int]] = {}
    for i, iid in enumerate(iids):
        by_iid.setdefault(iid, []).append(i)
    keys = list(by_iid.keys())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample_keys = rng.choice(len(keys), size=len(keys), replace=True)
        idx = []
        for k in sample_keys:
            idx.extend(by_iid[keys[k]])
        idx_arr = np.array(idx)
        boot[b] = roc_auc_score(y[idx_arr], p[idx_arr])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(np.mean(fold_aucs)), (float(lo), float(hi)), fold_aucs, y, p, iids


def paired_delta_ci(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, iids: np.ndarray,
    n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap of AUC(p_a) - AUC(p_b) resampling unique instance_ids
    so the same task contributes its 4 model rows together. Returns
    (point_delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    by_iid: dict[str, list[int]] = {}
    for i, iid in enumerate(iids):
        by_iid.setdefault(iid, []).append(i)
    keys = list(by_iid.keys())
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sample_keys = rng.choice(len(keys), size=len(keys), replace=True)
        idx = []
        for k in sample_keys:
            idx.extend(by_iid[keys[k]])
        idx_arr = np.array(idx)
        deltas[b] = roc_auc_score(y[idx_arr], p_a[idx_arr]) - roc_auc_score(y[idx_arr], p_b[idx_arr])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> int:
    csv_path = DATA / "features_with_both_sim.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}; run similarity_mellum.py first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r["model"] in LABELLED and r["resolved"] in {"True", "False"}]

    print(f"labelled rows: {len(rows)}")
    print()
    print(f"{'Feature subset':<22}  {'mean AUC':>9}  {'95% CI':>14}   per-fold (Opus5h, Gemini, MiniMax, Opus6)")
    print("-" * 100)
    rows_for_overlap = [r for r in rows if "patch_overlap" in r]
    if not rows_for_overlap:
        print("  (run patch_overlap.py first to populate the overlap column)")
        return 1
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, feats in SUBSETS.items():
        mean, (lo, hi), aucs, y, p, iids = lomo_auc(rows, feats)
        preds[name] = (y, p, iids)
        print(f"  {name:<20}  {mean:>9.3f}  [{lo:.3f}, {hi:.3f}]   {' '.join(f'{a:.3f}' for a in aucs)}")

    # Paired bootstraps on instance_id. Done as deltas, not as overlapping
    # marginal CIs (which is what the v1 prose used).
    y_q, p_q, iid_q = preds["+qwen"]
    y_m, p_m, iid_m = preds["+mellum"]
    y_o, p_o, iid_o = preds["+overlap"]
    assert np.array_equal(y_q, y_m) and np.array_equal(iid_q, iid_m)
    assert np.array_equal(y_q, y_o) and np.array_equal(iid_q, iid_o)
    print()
    print("Paired AUC deltas, resampled by instance_id (n_boot=2000):")
    for name_a, name_b, p_a, p_b in [
        ("+mellum", "+qwen",   p_m, p_q),
        ("+mellum", "+overlap", p_m, p_o),
        ("+overlap", "+qwen",  p_o, p_q),
    ]:
        point, lo, hi = paired_delta_ci(y_q, p_a, p_b, iid_q)
        sig = "includes 0" if lo <= 0 <= hi else "excludes 0"
        print(f"  {name_a:>9} vs {name_b:<9} delta = {point:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
