"""Embed a subset of patches with Mellum, in both raw and FIM input formats.

The full 3000-patch FIM re-embed needs the GPU box; on the local 16GB Mac
it would take many hours under MPS. To keep the FIM-vs-raw comparison
honest without re-embedding the world, this script:

  1. Reads `data/exec_subset.json` -- the 50 instance IDs picked for the
     killer experiment.
  2. Embeds the ground truth + 4-model predictions for each of those
     50 instances (250 patches), once with --input-format raw and once
     with --input-format fim, plus a separate 250-patch random sample
     drawn from the same 4-model pool, so the comparison covers more
     than the killer-experiment subset.

Output: `data/embeddings_mellum_subset_<format>.npy` plus index CSVs
matching the existing `embeddings_mellum_index.csv` schema.

Then `compare_mellum_formats.py` joins them and reports the
resolved/unresolved cosine gap, both formats, on the same rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embed_mellum import MODEL_ID, fim_wrap

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def collect_subset(data: Path, exec_subset: list[str], extra_n: int, seed: int = 7) -> list[tuple[str, str, str, str]]:
    """Return rows of (kind, group, instance_id, path).

    For exec_subset, include ground truth + every model's prediction.
    For the extra random sample, draw `extra_n` (model, iid) pairs
    uniformly from the predicted pool, plus their ground truths.
    """
    rng = random.Random(seed)
    rows: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, group: str, iid: str, path: Path) -> None:
        key = (kind, group, iid)
        if key in seen or not path.exists():
            return
        rows.append((kind, group, iid, str(path)))
        seen.add(key)

    pred_root = data / "predicted"
    model_dirs = [d for d in sorted(pred_root.iterdir()) if d.is_dir() and d.name != "gpt-5-2-codex"]

    for iid in exec_subset:
        gt = data / "patches" / f"{iid}.patch"
        add("gt", "ground_truth", iid, gt)
        for m in model_dirs:
            add("pred", m.name, iid, m / f"{iid}.patch")

    # Extra random sample of (model, iid) pairs
    pool: list[tuple[Path, str, str]] = []
    for m in model_dirs:
        for p in sorted(m.glob("*.patch")):
            pool.append((p, m.name, p.stem))
    rng.shuffle(pool)
    added = 0
    for p, group, iid in pool:
        if added >= extra_n:
            break
        if (data / "patches" / f"{iid}.patch").exists():
            add("gt", "ground_truth", iid, data / "patches" / f"{iid}.patch")
            add("pred", group, iid, p)
            added += 1
    return rows


def encode_batch(model, tokenizer, texts: list[str], device: str, fmt: str, max_len: int) -> np.ndarray:
    if fmt == "fim":
        texts = [fim_wrap(t) for t in texts]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=False)
    h = out.last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().to(torch.float32).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-subset", type=Path, default=DATA / "exec_subset.json")
    ap.add_argument("--extra-n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    exec_subset = json.loads(args.exec_subset.read_text())
    rows = collect_subset(DATA, exec_subset, args.extra_n)
    print(f"{len(rows)} patches to embed (raw + fim, both)", file=sys.stderr)

    for fmt in ("raw", "fim"):
        print(f"\n== {fmt} ==", file=sys.stderr)
        embs = []
        for i in range(0, len(rows), args.batch):
            batch = rows[i : i + args.batch]
            texts = [Path(r[3]).read_text() or " " for r in batch]
            embs.append(encode_batch(model, tokenizer, texts, device, fmt, args.max_len))
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(batch)}/{len(rows)}", file=sys.stderr)
        arr = np.vstack(embs)
        np.save(DATA / f"embeddings_mellum_subset_{fmt}.npy", arr)
        with (DATA / f"embeddings_mellum_subset_{fmt}_index.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "kind", "group", "instance_id"])
            for j, (kind, group, iid, _) in enumerate(rows):
                w.writerow([j, kind, group, iid])
        print(f"wrote embeddings_mellum_subset_{fmt}.npy shape={arr.shape}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
