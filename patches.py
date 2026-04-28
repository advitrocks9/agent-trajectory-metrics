"""Pull SWE-bench Verified ground-truth patches from HuggingFace.

Stores them under data/patches/{instance_id}.patch so we can compare them
to each model's submission later.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "patches"
URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/"
    "resolve/main/data/test-00000-of-00001.parquet"
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    req = Request(URL, headers={"User-Agent": "agent-traj-metrics/0.1"})
    print("downloading swe-bench verified parquet...", file=sys.stderr)
    raw = urlopen(req, timeout=120).read()
    print(f"  {len(raw):_} bytes", file=sys.stderr)

    # pyarrow handles parquet without pulling pandas as a dependency
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(raw))
    df = table.to_pandas()
    print(f"  {len(df)} rows; columns: {list(df.columns)[:8]}...", file=sys.stderr)

    n = 0
    for row in df[["instance_id", "patch"]].itertuples(index=False):
        path = OUT_DIR / f"{row.instance_id}.patch"
        path.write_text(row.patch)
        n += 1
    print(f"wrote {n} patches to {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
