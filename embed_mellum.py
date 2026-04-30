"""Re-embed every patch with JetBrains Mellum-4b-sft-python.

Same shape as embed_remote.py but with a JetBrains-trained Python code
model so we can ask: does a code-completion model fine-tuned for the
Python ecosystem produce a tighter signal than a general code LM (Qwen),
on a benchmark that is all Python?

Run on the GPU box; pull embeddings.npy back the same way.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "JetBrains/Mellum-4b-sft-python"


def encode_batch(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    # Mellum-4B needs ~7.5GB for weights alone; the GPU shares with two
    # other jobs, so trim context aggressively. Patches almost never exceed
    # 1024 tokens after truncation anyway.
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=False)
    h = out.last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().to(torch.float32).numpy()


def collect(root: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for p in sorted((root / "patches").glob("*.patch")):
        rows.append(("gt", "ground_truth", p.stem, str(p)))
    pred_root = root / "predicted"
    for slug_dir in sorted(pred_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        for p in sorted(slug_dir.glob("*.patch")):
            rows.append(("pred", slug_dir.name, p.stem, str(p)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-emb", type=Path, default=Path("embeddings_mellum.npy"))
    ap.add_argument("--out-idx", type=Path, default=Path("embeddings_mellum_index.csv"))
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    rows = collect(args.data)
    print(f"{len(rows)} patches to embed with {MODEL_ID}", file=sys.stderr)

    embs: list[np.ndarray] = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        texts = [Path(r[3]).read_text() or " " for r in batch]
        embs.append(encode_batch(model, tokenizer, texts, device))
        if (i // args.batch) % 40 == 0:
            print(f"  {i + len(batch)}/{len(rows)}", file=sys.stderr)

    arr = np.vstack(embs)
    np.save(args.out_emb, arr)
    with args.out_idx.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "kind", "group", "instance_id"])
        for j, (kind, group, iid, _) in enumerate(rows):
            w.writerow([j, kind, group, iid])
    print(f"wrote {args.out_emb} shape={arr.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
