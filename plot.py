"""Survival curves with bootstrap CIs + a feature-coefficient bar chart.

Outputs PNGs under plots/. Uses matplotlib only.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

LABELLED = ["Claude 4.5 Opus high", "Gemini 3 Flash high", "MiniMax M2.5 high", "Claude 4.6 Opus"]
COLOURS = {
    "Claude 4.5 Opus high": "#cc7722",
    "Gemini 3 Flash high": "#3873b8",
    "MiniMax M2.5 high":   "#2a8b3a",
    "Claude 4.6 Opus":     "#b03060",
    "GPT-5-2-Codex":       "#5f5f5f",
}


def survival_curve(
    turns: np.ndarray, resolved: np.ndarray, ks: np.ndarray, n_boot: int = 1000, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each k in ks, return (point estimate, lo, hi, n_at_risk).

    P(resolve | reached turn k) bootstrapped over the observed instances.
    """
    rng = np.random.default_rng(seed)
    n = len(turns)
    pt = np.zeros(len(ks))
    lo = np.zeros(len(ks))
    hi = np.zeros(len(ks))
    at = np.zeros(len(ks), dtype=int)
    for j, k in enumerate(ks):
        mask = turns >= k
        at[j] = mask.sum()
        if at[j] == 0:
            pt[j] = lo[j] = hi[j] = np.nan
            continue
        pt[j] = resolved[mask].mean()
        boot = np.empty(n_boot)
        idx_pool = np.where(mask)[0]
        for b in range(n_boot):
            sample = rng.choice(idx_pool, size=len(idx_pool), replace=True)
            boot[b] = resolved[sample].mean()
        lo[j], hi[j] = np.quantile(boot, [0.025, 0.975])
    return pt, lo, hi, at


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    src = DATA / "features_with_both_sim.csv"
    if not src.exists():
        src = DATA / "features_with_sim.csv"
    rows = list(csv.DictReader(src.open()))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    # Plot 1: survival curves with bootstrap 95% CIs
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = np.arange(0, 121, 5)
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"}]
        turns = np.array([int(r["n_assistant"]) for r in rs])
        resolved = np.array([1 if r["resolved"] == "True" else 0 for r in rs])
        pt, lo, hi, _ = survival_curve(turns, resolved, ks)
        ax.plot(ks, pt, label=m, color=COLOURS[m], linewidth=2)
        ax.fill_between(ks, lo, hi, color=COLOURS[m], alpha=0.18, linewidth=0)
    ax.set_xlabel("turn count k (assistant messages)")
    ax.set_ylabel("P(resolve  |  trajectory reached turn k)")
    ax.set_title("Conditional resolve probability by depth (95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 120)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "survival.png", dpi=150)
    plt.close(fig)

    # Plot 2: edit churn distribution by model + resolved status
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    xpos = np.arange(len(LABELLED))
    res_means, unres_means, res_err, unres_err = [], [], [], []
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"} and int(r["n_repo_edits"]) > 0]
        ch_res = np.array([float(r["edit_churn"]) for r in rs if r["resolved"] == "True"])
        ch_un = np.array([float(r["edit_churn"]) for r in rs if r["resolved"] == "False"])
        res_means.append(ch_res.mean() if len(ch_res) else 0)
        unres_means.append(ch_un.mean() if len(ch_un) else 0)
        res_err.append(ch_res.std(ddof=1) / np.sqrt(len(ch_res)) if len(ch_res) > 1 else 0)
        unres_err.append(ch_un.std(ddof=1) / np.sqrt(len(ch_un)) if len(ch_un) > 1 else 0)
    ax.bar(xpos - width / 2, res_means, width, yerr=res_err, label="resolved", color="#3873b8")
    ax.bar(xpos + width / 2, unres_means, width, yerr=unres_err, label="unresolved", color="#cc4422")
    ax.set_xticks(xpos)
    ax.set_xticklabels([m.replace(" high", "") for m in LABELLED], rotation=15, ha="right")
    ax.set_ylabel("mean edit churn")
    ax.set_title("Edit churn (re-edit / repo-edit) by model and outcome")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "edit_churn.png", dpi=150)
    plt.close(fig)

    # Plot 3: patch_sim distribution (resolved vs unresolved) per model
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box_data = []
    box_labels = []
    box_colors = []
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"} and r["patch_sim"]]
        sim_res = np.array([float(r["patch_sim"]) for r in rs if r["resolved"] == "True"])
        sim_un = np.array([float(r["patch_sim"]) for r in rs if r["resolved"] == "False"])
        box_data.extend([sim_res, sim_un])
        short = m.replace(" high", "")
        box_labels.extend([f"{short}\nres", f"{short}\nunres"])
        box_colors.extend(["#3873b8", "#cc4422"])
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, fontsize=8)
    ax.set_ylabel("cosine sim to ground-truth patch")
    ax.set_title("Patch similarity to ground truth, by outcome")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS / "patch_sim.png", dpi=150)
    plt.close(fig)

    # Plot 4: Mellum vs Qwen patch-similarity gap, by model + outcome.
    has_mellum = any(r.get("patch_sim_mellum") for r in rows)
    if has_mellum:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.18
        xpos = np.arange(len(LABELLED))
        for i, (col, label, colour, off) in enumerate([
            ("patch_sim", "Qwen res", "#3873b8", -1.5),
            ("patch_sim", "Qwen unres", "#9bbedd", -0.5),
            ("patch_sim_mellum", "Mellum res", "#cc7722", 0.5),
            ("patch_sim_mellum", "Mellum unres", "#e6b87a", 1.5),
        ]):
            vals, errs = [], []
            for m in LABELLED:
                rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"} and r.get(col)]
                target = "True" if "res" in label and "unres" not in label else "False"
                xs = np.array([float(r[col]) for r in rs if r["resolved"] == target])
                vals.append(xs.mean() if len(xs) else 0)
                errs.append(xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0)
            ax.bar(xpos + off * width, vals, width, yerr=errs, label=label, color=colour)
        ax.set_xticks(xpos)
        ax.set_xticklabels([m.replace(" high", "") for m in LABELLED], rotation=15, ha="right")
        ax.set_ylabel("mean cosine sim to ground-truth patch")
        ax.set_title("Patch similarity by encoder: Mellum gives a wider resolved/unresolved gap")
        ax.set_ylim(0.82, 1.0)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(PLOTS / "mellum_vs_qwen.png", dpi=150)
        plt.close(fig)

    # Plot 5: prefix-conditional classifier AUC vs k (LOMO mean ± fold range).
    prefix_path = DATA / "prefix_features.csv"
    if prefix_path.exists():
        from classify_prefix import PREFIX_FEATURES, fit_eval, LABELLED as PF_LABELLED
        prows = list(csv.DictReader(prefix_path.open()))
        prows = [r for r in prows if r["model"] in PF_LABELLED and r["resolved"] in {"True", "False"}]
        ks_pf = [5, 10, 20, 30, 50]
        means, los, his = [], [], []
        for k in ks_pf:
            sub = [r for r in prows if int(r["k"]) == k and r["reached"] == "1"]
            aucs = []
            for held in PF_LABELLED:
                train = [r for r in sub if r["model"] != held]
                test = [r for r in sub if r["model"] == held]
                if len(test) < 20 or len(set(r["resolved"] for r in test)) < 2:
                    continue
                auc, _, _ = fit_eval(train, test)
                aucs.append(auc)
            means.append(np.mean(aucs))
            los.append(min(aucs))
            his.append(max(aucs))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks_pf, means, "o-", color="#3a3a3a", linewidth=2, label="LOMO mean AUC")
        ax.fill_between(ks_pf, los, his, color="#3a3a3a", alpha=0.18, linewidth=0,
                        label="fold range")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="chance (AUC=0.5)")
        ax.set_xlabel("prefix length k (assistant turns visible to the classifier)")
        ax.set_ylabel("LOMO AUC of P(eventual resolved | prefix-features)")
        ax.set_title("In-flight classifier AUC decays with prefix length")
        ax.set_ylim(0.35, 0.7)
        ax.set_xticks(ks_pf)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / "prefix_auc.png", dpi=150)
        plt.close(fig)
    written = sum(1 for p in PLOTS.glob("*.png"))
    print(f"wrote {written} PNGs under {PLOTS}/")


if __name__ == "__main__":
    main()
