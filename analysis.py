"""Numbers cited in report.md. Single source of truth.

Run after aggregate.py + similarity.py.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MODELS = [
    "Claude 4.5 Opus high",
    "Gemini 3 Flash high",
    "MiniMax M2.5 high",
    "Claude 4.6 Opus",
    "GPT-5-2-Codex",
]
LABELLED = MODELS[:4]

FOLDER_OF = {
    "Claude 4.5 Opus high": "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "Gemini 3 Flash high": "20260217_mini-v2.0.0_gemini-3-flash-high",
    "MiniMax M2.5 high": "20260217_mini-v2.0.0_minimax-2-5-high",
    "Claude 4.6 Opus": "20260217_mini-v2.0.0_claude-4-6-opus",
    "GPT-5-2-Codex": "20260219_mini-v2.0.0_gpt-5-2-codex",
}


def median(xs):
    return statistics.median(xs) if xs else 0.0


def quantile(xs, q):
    s = sorted(xs); return s[int(q * (len(s) - 1))] if s else 0.0


def avg_ranks(xs):
    n = len(xs); order = sorted(range(n), key=lambda i: xs[i]); ranks = [0.0]*n
    i = 0
    while i < n:
        j = i
        while j+1 < n and xs[order[j+1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    rx, ry = avg_ranks(xs), avg_ranks(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    sx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    sy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num / (sx*sy) if sx and sy else 0.0


def main():
    for cand in ("features_with_both_sim.csv", "features_with_sim.csv", "features.csv"):
        csv_path = DATA / cand
        if csv_path.exists():
            break
    rows = list(csv.DictReader(csv_path.open()))
    meta = {r["folder"]: r for r in json.loads((DATA / "leaderboard.json").read_text())}
    by_model = defaultdict(list)
    for r in rows: by_model[r["model"]].append(r)

    print("Top-of-leaderboard, mini-SWE-agent v2 on SWE-bench Verified")
    print("=" * 78)
    print(f"{'Model':<22} {'pass@1':>7} {'$/inst':>8} {'asst med':>9} {'edits':>6} {'churn':>6} {'patch L':>8}")
    for m in MODELS:
        rs = by_model[m]
        leader = meta[FOLDER_OF[m]]
        a = sorted(int(r["n_assistant"]) for r in rs)
        e = sorted(int(r["n_edit"]) for r in rs)
        c = sorted(float(r["edit_churn"]) for r in rs if int(r["n_repo_edits"]) > 0)
        pl = sorted(int(r["patch_lines"]) for r in rs)
        print(f"{m:<22} {leader['resolved']:>6.1f}% ${leader['cost']/len(rs):>6.3f} "
              f"{median(a):>9.0f} {median(e):>6.0f} {median(c):>6.2f} {median(pl):>8.0f}")

    print()
    print("Conditional resolve probability given trajectory has reached turn k")
    print("=" * 78)
    ks = [0, 10, 20, 30, 50, 75, 100]
    print(f"{'Model':<22} " + "  ".join(f"k>={k:>3}" for k in ks))
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"}]
        cells = []
        for k in ks:
            reached = [r for r in rs if int(r["n_assistant"]) >= k]
            if not reached: cells.append("    -"); continue
            res = sum(1 for r in reached if r["resolved"] == "True") / len(reached)
            cells.append(f"{res:>5.0%}")
        print(f"{m:<22} {'   '.join(cells)}")

    print()
    print("Edit churn (re-edit / repo-edit) by outcome, only trajectories with >=1 repo edit")
    print("=" * 78)
    print(f"{'Model':<22} {'n':>4} {'churn (resolved)':>18} {'churn (unresolved)':>20}")
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"} and int(r["n_repo_edits"]) > 0]
        ch_r = [float(r["edit_churn"]) for r in rs if r["resolved"] == "True"]
        ch_u = [float(r["edit_churn"]) for r in rs if r["resolved"] == "False"]
        print(f"{m:<22} {len(rs):>4} "
              f"{statistics.mean(ch_r) if ch_r else 0:>10.3f} (n={len(ch_r)})  "
              f"{statistics.mean(ch_u) if ch_u else 0:>10.3f} (n={len(ch_u)})")

    print()
    print("Spearman ρ (tie-corrected) between feature and resolved (binary)")
    print("=" * 78)
    feats = ["n_assistant", "n_edit", "edit_churn", "n_test", "patch_lines", "patch_sim", "patch_sim_mellum"]
    print(f"{'Model':<22} " + "  ".join(f"{f:>11}" for f in feats))
    for m in LABELLED:
        rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"}]
        ys = [1 if r["resolved"] == "True" else 0 for r in rs]
        cells = []
        for f in feats:
            xs = [float(r[f]) if r[f] not in ("", None) else float("nan") for r in rs]
            valid = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x)]
            if len(valid) < 50:
                cells.append(f"{'-':>11}")
                continue
            vx, vy = zip(*valid)
            rho = spearman(list(vx), list(vy))
            cells.append(f"{rho:+11.3f}")
        print(f"{m:<22} {'  '.join(cells)}")

    print()
    print("Patch similarity by outcome (Qwen vs Mellum)")
    print("=" * 78)
    for col, label in [("patch_sim", "Qwen-Coder-1.5B"), ("patch_sim_mellum", "Mellum-4B-sft-py")]:
        print(f"  {label}:")
        print(f"    {'Model':<22} {'mean res':>10} {'mean unres':>11} {'gap':>7}")
        for m in LABELLED:
            rs = [r for r in by_model[m] if r["resolved"] in {"True", "False"} and r.get(col)]
            s_r = [float(r[col]) for r in rs if r["resolved"] == "True"]
            s_u = [float(r[col]) for r in rs if r["resolved"] == "False"]
            if s_r and s_u:
                gap = statistics.mean(s_r) - statistics.mean(s_u)
                print(f"    {m:<22} {statistics.mean(s_r):>10.3f} {statistics.mean(s_u):>11.3f} {gap:>+7.3f}")
        print()

    print()
    print("Cost per resolved task (leaderboard cost / number resolved)")
    print("=" * 78)
    for m in MODELS:
        leader = meta[FOLDER_OF[m]]
        n_res = leader["resolved"] / 100 * len(by_model[m])
        print(f"{m:<22} ${leader['cost']/n_res:>5.3f} per resolved (${leader['cost']:.0f} / {n_res:.0f})")

    print()
    print("Per-repo summary (8 most common repos in SWE-bench Verified)")
    print("=" * 78)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["resolved"] not in {"True", "False"}:
            continue
        by_repo[r["instance_id"].split("__", 1)[0]].append(r)
    common_repos = sorted(by_repo.items(), key=lambda kv: -len(kv[1]))[:8]
    print(f"  {'repo':<14} {'n':>4} {'pass@1':>7} {'turns med':>10} {'patch_sim_m':>12} {'patch_overlap':>14}")
    for repo, rs in common_repos:
        n = len(rs)
        pass1 = sum(1 for r in rs if r["resolved"] == "True") / n * 100
        turns = sorted(int(r["n_assistant"]) for r in rs)
        sims = [float(r["patch_sim_mellum"]) for r in rs if r.get("patch_sim_mellum")]
        ovs = [float(r["patch_overlap"]) for r in rs if r.get("patch_overlap")]
        print(
            f"  {repo:<14} {n:>4} {pass1:>6.1f}% {turns[n//2]:>10} "
            f"{statistics.mean(sims):>12.3f} {statistics.mean(ovs):>14.3f}"
        )

    print()
    print("Hit-the-step-cap = LimitsExceeded")
    print("=" * 78)
    total = solved = 0
    for m in MODELS:
        rs = [r for r in by_model[m] if r["exit_status"] == "LimitsExceeded"]
        n_res = sum(1 for r in rs if r["resolved"] == "True")
        if rs:
            total += len(rs); solved += n_res
            print(f"{m:<22} {len(rs):>3} hit, {n_res} resolved")
    print(f"{'TOTAL':<22} {total:>3} hit, {solved} resolved")


if __name__ == "__main__":
    main()
