"""Embed every patch (predicted + ground truth) into a fixed-size vector.

Run on the 4090. Takes the patches/ and predicted/ directories that the
Mac has rsync'd over, mean-pools the last hidden states of
Qwen2.5-Coder-1.5B over each patch, and writes embeddings.npy + an index
CSV that maps each row to its (kind, model_or_gt, instance_id).

I picked Qwen Coder over a general-purpose embedding model because the
patches are diff-text, not natural language. Code-trained encoders give
noticeably tighter clusters on a small spot-check (n=20 pairs).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"


def encode_batch(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=False)
    # Mean-pool the last hidden states under the attention mask.
    h = out.last_hidden_state           # (B, T, D)
    mask = enc.attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().to(torch.float32).numpy()


def collect(root: Path) -> list[tuple[str, str, str, str]]:
    """Return rows of (kind, group, instance_id, path).

    kind = "gt" for the ground truth, "pred" for a model submission.
    group = "ground_truth" for gt, model slug for pred.
    """
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
    ap.add_argument("--data", type=Path, required=True, help="dir holding patches/ and predicted/")
    ap.add_argument("--out-emb", type=Path, default=Path("embeddings.npy"))
    ap.add_argument("--out-idx", type=Path, default=Path("embeddings_index.csv"))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    rows = collect(args.data)
    print(f"{len(rows)} patches to embed", file=sys.stderr)

    embeddings: list[np.ndarray] = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        texts = [Path(r[3]).read_text() or " " for r in batch]
        embeddings.append(encode_batch(model, tokenizer, texts, device))
        if (i // args.batch) % 20 == 0:
            print(f"  {i + len(batch)}/{len(rows)}", file=sys.stderr)

    arr = np.vstack(embeddings)
    np.save(args.out_emb, arr)

    with args.out_idx.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "kind", "group", "instance_id"])
        for j, (kind, group, iid, _) in enumerate(rows):
            w.writerow([j, kind, group, iid])

    print(f"wrote {args.out_emb} shape={arr.shape}, {args.out_idx}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
