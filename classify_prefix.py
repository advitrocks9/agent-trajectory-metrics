"""Predict eventual resolution from features visible at turn k.

Same logistic regression as classify.py but the inputs are the truncated
features from prefix_features.csv. For each k in {5, 10, 20, 30, 50, 75},
restrict to trajectories that reached k assistant turns, and fit
leave-one-model-out across the four labelled models.

This is the honest "in-flight" version of the classifier in classify.py.
That one cheats by using patch_lines / patch_files / patch_sim, which are
only available after submission. This one uses only what an online
scaffold can read.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"

PREFIX_FEATURES = [
    "n_calls_prefix",
    "n_repo_edits_prefix",
    "n_edit_files_prefix",
    "edit_churn_prefix",
    "call_unique_ratio_prefix",
    "n_read_prefix", "n_edit_prefix", "n_search_prefix",
    "n_test_prefix", "n_git_prefix", "n_install_prefix",
]

LABELLED = [
    "Claude 4.5 Opus high",
    "Gemini 3 Flash high",
    "MiniMax M2.5 high",
    "Claude 4.6 Opus",
]


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[f]) for f in PREFIX_FEATURES] for r in rows])
    y = np.array([1 if r["resolved"] == "True" else 0 for r in rows])
    return X, y


def fit_eval(train: list[dict], test: list[dict]) -> tuple[float, float, float]:
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]
    if len(set(yte)) < 2:
        return float("nan"), float("nan"), float("nan")
    return roc_auc_score(yte, proba), brier_score_loss(yte, proba), proba.mean()


def within_model_cv_auc(rows: list[dict], n_splits: int = 5) -> tuple[float, list[float]]:
    """Per-model GroupKFold CV AUC keyed by instance_id, averaged across models.

    Earlier this used `np.random.shuffle`-style folds, which understates
    the variance of CV estimates because different folds can hold rows
    from the same instance the model has seen at a different prefix
    length. With one row per (model, instance, k) and a fixed k subset
    here, the shuffle isn't a leak per se, but GroupKFold matches the
    per-instance variance the LOMO numbers are tested against.
    """
    aucs: list[float] = []
    for m in LABELLED:
        sub = [r for r in rows if r["model"] == m]
        if len(sub) < 50 or len({r["resolved"] for r in sub}) < 2:
            continue
        groups = np.array([r["instance_id"] for r in sub])
        n_groups = len(set(groups.tolist()))
        if n_groups < n_splits:
            continue
        kf = GroupKFold(n_splits=n_splits)
        idx_arr = np.arange(len(sub))
        for train_i, test_i in kf.split(idx_arr, groups=groups):
            test = [sub[i] for i in test_i]
            train = [sub[i] for i in train_i]
            if len({r["resolved"] for r in test}) < 2:
                continue
            auc, _, _ = fit_eval(train, test)
            if not np.isnan(auc):
                aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan"), aucs


def main() -> int:
    rows = list(csv.DictReader((DATA / "prefix_features.csv").open()))
    rows = [r for r in rows if r["model"] in LABELLED and r["resolved"] in {"True", "False"}]

    print("Prefix-conditional LOMO classifier")
    print("=" * 72)
    print(f"{'k':>3} {'fold':<22} {'n_test':>7} {'AUC':>6} {'Brier':>7} {'p_mean':>7}")
    summary: dict[int, list[float]] = {}
    for k in [5, 10, 20, 30, 50, 75]:
        # Only trajectories that actually reached turn k contribute.
        sub = [r for r in rows if int(r["k"]) == k and r["reached"] == "1"]
        aucs = []
        for held in LABELLED:
            train = [r for r in sub if r["model"] != held]
            test = [r for r in sub if r["model"] == held]
            if len(test) < 20 or len(set(r["resolved"] for r in test)) < 2:
                print(f"{k:>3} {held:<22} {len(test):>7}   skipped (too few/single class)")
                continue
            auc, brier, pmean = fit_eval(train, test)
            print(f"{k:>3} {held:<22} {len(test):>7} {auc:>6.3f} {brier:>7.3f} {pmean:>7.3f}")
            aucs.append(auc)
        if aucs:
            summary[k] = aucs
            print(f"{k:>3} {'mean':<22} {'':>7} {np.mean(aucs):>6.3f}")
            print()

    print("Mean LOMO AUC by prefix length k:")
    for k, aucs in summary.items():
        print(f"  k={k:>3}  mean AUC = {np.mean(aucs):.3f}  range across folds = [{min(aucs):.3f}, {max(aucs):.3f}]  n_folds={len(aucs)}")

    print()
    print("Sanity: within-model 5-fold CV (no transfer)")
    print("=" * 72)
    print(f"{'k':>3} {'mean AUC':>10}  {'n_folds':>8}")
    for k in [5, 10, 20, 30, 50, 75]:
        sub = [r for r in rows if int(r["k"]) == k and r["reached"] == "1"]
        mean, all_aucs = within_model_cv_auc(sub)
        print(f"{k:>3} {mean:>10.3f}  {len(all_aucs):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
