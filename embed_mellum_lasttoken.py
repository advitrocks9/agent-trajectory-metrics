"""Last-token-pooled embeddings from Mellum-4b-sft-python.

embed_mellum.py mean-pools the last hidden state. Mellum is a causal LM,
so the last non-pad token's hidden state has seen the whole sequence and
is the conventional choice for sentence-level representations from a
decoder. This script does that, on top of the same FIM input format.

Usage matches embed_mellum.py; produces embeddings_mellum_last.npy +
embeddings_mellum_last_index.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embed_mellum import MODEL_ID, collect, fim_wrap


def encode_batch(
    model, tokenizer, texts: list[str], device: str, input_format: str, max_len: int
) -> np.ndarray:
    if input_format == "fim":
        texts = [fim_wrap(t) for t in texts]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=False)
    h = out.last_hidden_state  # (B, T, D)
    # Last non-pad token per row. Pad is right by default in tokenisers.
    seq_lens = enc.attention_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
    pooled = h[torch.arange(h.size(0)), seq_lens]
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().to(torch.float32).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-emb", type=Path, default=Path("embeddings_mellum_last.npy"))
    ap.add_argument("--out-idx", type=Path, default=Path("embeddings_mellum_last_index.csv"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--input-format", choices=("fim", "raw"), default="fim")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}; pool: last-token; input-format: {args.input_format}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-pad the inputs so the last attended index is the natural EOS-side
    # token (rather than the leftmost pad).
    tokenizer.padding_side = "right"
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    rows = collect(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} patches to embed (last-token pool)", file=sys.stderr)

    embs: list[np.ndarray] = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        texts = [Path(r[3]).read_text() or " " for r in batch]
        embs.append(encode_batch(model, tokenizer, texts, device, args.input_format, args.max_len))
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
