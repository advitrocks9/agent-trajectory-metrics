"""Logistic regression on trajectory-shape features -> P(resolved).

Cross-validation is leave-one-model-out: train on four models, evaluate on
the fifth. This is a stricter test than within-model CV; it forces the
features to carry signal that generalises across very different agent
behaviours rather than memorising one model's quirks.

GPT-5-2-Codex has no per-instance resolved flags on the leaderboard, so it
sits out the labelled work but still gets predictions.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# Features used for the classifier. Patch features (lines, files, sim) are
# trajectory-quality signals; the action counts and edit churn are
# trajectory-shape signals. We deliberately exclude n_assistant on the
# main run so the model can't just learn "Claude is short, MiniMax is long".
FEATURES = [
    "n_calls",
    "n_read", "n_edit", "n_search", "n_test", "n_git", "n_install",
    "n_repo_edits", "n_edit_files", "edit_churn", "call_unique_ratio",
    "patch_lines", "patch_files",
    "patch_sim",
]

LABELLED_MODELS = [
    "Claude 4.5 Opus high",
    "Gemini 3 Flash high",
    "MiniMax M2.5 high",
    "Claude 4.6 Opus",
]


def load_rows() -> list[dict]:
    return list(csv.DictReader((DATA / "features_with_sim.csv").open()))


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    y = np.empty(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        for j, f in enumerate(FEATURES):
            v = r[f]
            X[i, j] = float(v) if v not in ("", None) else np.nan
        y[i] = 1 if r["resolved"] == "True" else 0
    return X, y


def fit_predict(train: list[dict], test: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    # Mean-impute missing patch_sim (rows with empty submission).
    mu = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)[:, 1]
    coefs = dict(zip(FEATURES, clf.coef_[0]))
    return yte, proba, coefs


def main() -> int:
    rows = load_rows()
    labelled = [r for r in rows if r["model"] in LABELLED_MODELS and r["resolved"] in {"True", "False"}]
    print(f"labelled rows: {len(labelled)}", file=sys.stderr)

    print()
    print("Leave-one-model-out cross-validation")
    print("=" * 64)
    print(f"{'held-out model':<24} {'n_test':>7} {'AUC':>6} {'Brier':>7} {'baseline AUC':>13}")
    auc_pooled, brier_pooled = [], []
    for held in LABELLED_MODELS:
        train = [r for r in labelled if r["model"] != held]
        test = [r for r in labelled if r["model"] == held]
        yte, proba, _ = fit_predict(train, test)
        if len(set(yte)) < 2:
            print(f"{held:<24} skipped (single class)")
            continue
        auc = roc_auc_score(yte, proba)
        brier = brier_score_loss(yte, proba)
        # Baseline: predict the held-out model's mean training resolve rate
        baseline = np.mean([1 if r["resolved"] == "True" else 0 for r in train])
        baseline_auc = roc_auc_score(yte, np.full_like(proba, baseline))
        print(f"{held:<24} {len(test):>7} {auc:>6.3f} {brier:>7.3f} {baseline_auc:>13.3f}")
        auc_pooled.append(auc)
        brier_pooled.append(brier)
    print(f"{'mean across folds':<24} {'':>7} {np.mean(auc_pooled):>6.3f} {np.mean(brier_pooled):>7.3f}")

    print()
    print("Coefficients (trained on all four labelled models)")
    print("=" * 64)
    _, _, coefs = fit_predict(labelled, labelled)
    for feat, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {feat:<22} {c:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
