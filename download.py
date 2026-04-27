"""Pull every mini-SWE-agent v2 Verified trajectory for the top-5 models.

Reads the leaderboard JSON embedded in https://www.swebench.com/ to find
each submission folder and the 500 instance IDs that make up SWE-bench
Verified, then anonymously pulls the trajectories from the bash-only S3
prefix. ~1.3 GB total. Reuses already-downloaded files.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
TRAJ_DIR = ROOT / "trajectories"
META_PATH = ROOT / "data" / "leaderboard.json"

LEADERBOARD_URL = "https://www.swebench.com/"
S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"

TARGET_FOLDERS = [
    "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_minimax-2-5-high",
    "20260217_mini-v2.0.0_claude-4-6-opus",
    "20260219_mini-v2.0.0_gpt-5-2-codex",
]


def http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "agent-traj-metrics/0.1"})
    with urlopen(req, timeout=60) as r:
        return r.read()


def fetch_leaderboard() -> list[dict]:
    META_PATH.parent.mkdir(exist_ok=True)
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    html = http_get(LEADERBOARD_URL).decode("utf-8")
    m = re.search(
        r'<script type="application/json" id="leaderboard-data">\s*(\[.*?\])\s*</script>',
        html, re.DOTALL,
    )
    if not m:
        raise RuntimeError("could not find leaderboard-data script tag")
    splits = json.loads(m.group(1))
    verified = next(s for s in splits if s["name"] == "Verified")
    wanted = {f: None for f in TARGET_FOLDERS}
    for r in verified["results"]:
        if r["folder"] in wanted:
            wanted[r["folder"]] = r
    rows = list(wanted.values())
    META_PATH.write_text(json.dumps(rows, indent=2))
    return rows


def download_one(folder: str, instance_id: str) -> tuple[str, str, str | None]:
    out = TRAJ_DIR / folder / f"{instance_id}.traj.json"
    if out.exists() and out.stat().st_size > 0:
        return folder, instance_id, None
    url = f"{S3_BASE}/bash-only/{folder}/trajs/{instance_id}/{instance_id}.traj.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_bytes(http_get(url))
    except Exception as e:
        return folder, instance_id, f"{type(e).__name__}: {e}"
    return folder, instance_id, None


def main() -> int:
    rows = fetch_leaderboard()
    # GPT-5-2-Codex's leaderboard row has no per_instance_details; reuse the
    # other models' instance set since SWE-bench Verified is fixed.
    instance_ids = next(
        sorted(r["per_instance_details"].keys())
        for r in rows if r.get("per_instance_details")
    )
    jobs = [(r["folder"], iid) for r in rows for iid in instance_ids]
    print(f"{len(jobs)} trajectories across {len(rows)} models", file=sys.stderr)

    done = 0
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        futures = [ex.submit(download_one, f, i) for f, i in jobs]
        for fut in as_completed(futures):
            folder, iid, err = fut.result()
            done += 1
            if err:
                failures.append((folder, iid, err))
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} ({len(failures)} failed)", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} failures (first 10):", file=sys.stderr)
        for f, i, e in failures[:10]:
            print(f"  {f}/{i}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
