"""Compare leaderboard `resolved` flags to a local SWE-bench harness run.

I sampled 10 django instances and ran the harness in docker on a 4090 box
for each of the five models. This script reads the per-(model, instance)
resolved/unresolved breakdown from the harness reports and prints a
side-by-side table against what the swebench.com leaderboard says.

If the two columns disagree the leaderboard is wrong (or my docker
harness is wrong). Either way it is worth knowing.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EVAL_DIR = DATA / "eval_reports"

SLUG_TO_MODEL = {
    "claude-4-5-opus-high": "Claude 4.5 Opus high",
    "gemini-3-flash-high": "Gemini 3 Flash high",
    "minimax-2-5-high": "MiniMax M2.5 high",
    "claude-4-6-opus": "Claude 4.6 Opus",
    "gpt-5-2-codex": "GPT-5-2-Codex",
}


def main() -> int:
    rows = list(csv.DictReader((DATA / "features_with_both_sim.csv").open()))
    by_pair = {(r["model"], r["instance_id"]): r for r in rows}

    reports = list(EVAL_DIR.glob("*.json"))
    if not reports:
        print(f"no reports under {EVAL_DIR}", file=sys.stderr)
        return 1

    agree = disagree = unlabelled = 0
    print(f"{'model':<22} {'instance':<28} {'leaderboard':>12}  {'docker':>8}  {'match':>6}")
    print("-" * 88)
    for path in sorted(reports):
        slug = path.name.split(".eval_")[0]
        model = SLUG_TO_MODEL.get(slug)
        if model is None:
            continue
        report = json.loads(path.read_text())
        resolved = set(report["resolved_ids"])
        unresolved = set(report["unresolved_ids"])
        for iid in sorted(resolved | unresolved):
            lb = by_pair.get((model, iid), {}).get("resolved", "")
            docker = "True" if iid in resolved else "False"
            if lb in {"True", "False"}:
                ok = "✓" if lb == docker else "✗"
                if lb == docker: agree += 1
                else: disagree += 1
            else:
                ok = "?"
                unlabelled += 1
            print(f"{model:<22} {iid:<28} {lb:>12}  {docker:>8}  {ok:>6}")
    n = agree + disagree
    print("-" * 88)
    if n:
        print(f"agreement: {agree}/{n} ({agree/n:.0%}); {disagree} disagree; {unlabelled} unlabelled (codex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
