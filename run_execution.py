"""Killer experiment: run the SWE-bench docker harness on a stratified
50-instance subset across the four labelled models and extract per-test
pass counts.

Why: the Task 2 classifier rests on diff-shape and embedding features.
The implicit question reviewers will ask is "does any of this beat just
running the test suite?". This script builds an `exec_<slug>.json` per
model containing the harness verdict + tests-pass-fraction per
instance, then `merge_execution.py` joins the four into
`data/execution.csv` for use as a feature in classify_compare.

Usage:
  python run_execution.py predictions    # writes data/exec_predictions/<slug>.jsonl
  python run_execution.py harness        # runs swebench.harness.run_evaluation per model
  python run_execution.py merge          # builds data/execution.csv from the harness reports

Why three steps: the harness can be slow (4-8 hours), so the predictions
pass is cheap and idempotent and the harness pass can resume from
cached image layers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRED_ROOT = DATA / "predicted"
SUBSET_PATH = DATA / "exec_subset.json"
PRED_OUT = DATA / "exec_predictions"
HARNESS_OUT = DATA / "exec_reports"

SLUG_BY_MODEL = {
    "Claude 4.5 Opus high": "claude-4-5-opus-high",
    "Gemini 3 Flash high": "gemini-3-flash-high",
    "MiniMax M2.5 high": "minimax-2-5-high",
    "Claude 4.6 Opus": "claude-4-6-opus",
}


def cmd_predictions() -> int:
    PRED_OUT.mkdir(exist_ok=True)
    iids = json.loads(SUBSET_PATH.read_text())
    print(f"{len(iids)} instances in subset", file=sys.stderr)
    for model, slug in SLUG_BY_MODEL.items():
        rows = []
        for iid in iids:
            patch_file = PRED_ROOT / slug / f"{iid}.patch"
            if not patch_file.exists():
                print(f"missing {patch_file}", file=sys.stderr)
                continue
            patch = patch_file.read_text()
            rows.append({
                "instance_id": iid,
                "model_name_or_path": slug,
                "model_patch": patch,
            })
        out = PRED_OUT / f"{slug}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out} ({len(rows)} predictions)", file=sys.stderr)
    return 0


def cmd_harness() -> int:
    """Run the SWE-bench harness per model.

    Importing inside the function so `python run_execution.py predictions`
    on a box without docker still works.
    """
    import subprocess

    HARNESS_OUT.mkdir(exist_ok=True)
    iids = json.loads(SUBSET_PATH.read_text())
    for model, slug in SLUG_BY_MODEL.items():
        pred_file = PRED_OUT / f"{slug}.jsonl"
        if not pred_file.exists():
            print(f"missing {pred_file}; run `predictions` first", file=sys.stderr)
            return 1
        run_id = f"exec_{slug}"
        # The harness writes <run_id>.<model_name>.json in cwd, plus
        # logs/run_evaluation/<run_id>/<model>/<instance>/...
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(pred_file),
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--run_id", run_id,
            "--instance_ids", *iids,
            "--max_workers", "4",
            "--cache_level", "instance",
        ]
        print("$", " ".join(cmd[:6]), "...", file=sys.stderr)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"harness exited {rc} for {slug}", file=sys.stderr)
        # Move report into HARNESS_OUT
        report = ROOT / f"{slug}.{run_id}.json"
        if report.exists():
            target = HARNESS_OUT / f"{slug}.{run_id}.json"
            report.rename(target)
            print(f"  -> {target}", file=sys.stderr)
    return 0


def cmd_merge() -> int:
    """Pull tests-pass numbers out of the harness logs.

    For each (model, instance) in HARNESS_OUT, look at
    `logs/run_evaluation/<run_id>/<model>/<iid>/report.json`. That JSON has
    `tests_status` with `FAIL_TO_PASS.success`, `FAIL_TO_PASS.failure`,
    `PASS_TO_PASS.success`, `PASS_TO_PASS.failure`. We compute
    tests_pass_frac = (FTP.success + PTP.success) / total.
    """
    import csv

    iids = json.loads(SUBSET_PATH.read_text())
    log_root = ROOT / "logs" / "run_evaluation"
    out_path = DATA / "execution.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "instance_id", "harness_resolved",
            "ftp_success", "ftp_failure", "ptp_success", "ptp_failure",
            "tests_pass_frac",
        ])
        n_rows = 0
        for model, slug in SLUG_BY_MODEL.items():
            run_id = f"exec_{slug}"
            for iid in iids:
                report_path = log_root / run_id / slug / iid / "report.json"
                if not report_path.exists():
                    w.writerow([model, iid, "", "", "", "", "", ""])
                    continue
                report = json.loads(report_path.read_text())
                inst_blob = report.get(iid, {})
                resolved = inst_blob.get("resolved", False)
                ts = inst_blob.get("tests_status", {})
                ftp = ts.get("FAIL_TO_PASS", {})
                ptp = ts.get("PASS_TO_PASS", {})
                fs = len(ftp.get("success", []))
                ff = len(ftp.get("failure", []))
                ps = len(ptp.get("success", []))
                pf = len(ptp.get("failure", []))
                total = fs + ff + ps + pf
                frac = (fs + ps) / total if total else ""
                w.writerow([
                    model, iid, "True" if resolved else "False",
                    fs, ff, ps, pf,
                    f"{frac:.4f}" if isinstance(frac, float) else "",
                ])
                n_rows += 1
        print(f"wrote {out_path} ({n_rows} rows with harness data)", file=sys.stderr)
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"predictions", "harness", "merge"}:
        print(__doc__, file=sys.stderr)
        return 1
    return {"predictions": cmd_predictions, "harness": cmd_harness, "merge": cmd_merge}[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
