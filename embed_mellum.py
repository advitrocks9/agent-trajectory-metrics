"""Re-embed every patch with JetBrains Mellum-4b-sft-python.

Same shape as embed_remote.py but with a JetBrains-trained Python code
model so we can ask: does a code-completion model fine-tuned for the
Python ecosystem produce a tighter signal than a general code LM (Qwen),
on a benchmark that is all Python?

Notes:
  * Mellum is a fill-in-the-middle code-completion model with 8192
    context. The first version of this script fed bare diff text
    truncated at 1024 tokens, so the model saw inputs the SFT was never
    asked to handle. Now wraps each patch with the model card's
    `<filename>` + `<fim_prefix>` / `<fim_suffix>` / `<fim_middle>`
    template (--input-format fim), which is what the SFT was trained on.
    --input-format raw keeps the old behaviour for ablation.
  * `AutoModel.from_pretrained` discards `lm_head.weight` (logged as
    UNEXPECTED in data/embed_mellum.log). That is correct for embedding
    extraction (we only need the base hidden states) but it does mean the
    SFT-specific specialisation in the LM head is lost.
  * `--batch 1` was a GPU-sharing workaround on prannayk-gpu-1 because
    two other jobs were running at the time; the default is 4.

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


def fim_wrap(patch: str) -> str:
    """Wrap a unified-diff patch in Mellum's FIM completion template.

    The model card shows multi-file FIM inputs of the form
        <filename>foo.py\n<contents>\n<filename>bar.py\n<contents>\n
        <filename>target.py\n<fim_suffix>...<fim_prefix>...<fim_middle>
    Our task is not actual completion; we just want a hidden state that
    reflects the patch in the format the model was trained on. So we
    treat the patch as the "current file" and ask for the middle of it.
    Empty prefix and suffix are deliberate: the patch is the whole
    payload we care about, and the FIM tokens push the activations into
    the SFT distribution.
    """
    return (
        f"<filename>tmp/patch.diff\n"
        f"<fim_suffix><fim_prefix>{patch}<fim_middle>"
    )


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
    # batch=4 is the default; the original 3000-patch run used batch=1
    # because the GPU was shared with two other jobs.
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--input-format", choices=("fim", "raw"), default="fim",
                    help="fim wraps each patch in <filename>+<fim_*> tokens; raw was the v1 behaviour")
    ap.add_argument("--max-len", type=int, default=2048,
                    help="tokenizer max_length; 1024 in v1 was a GPU-sharing workaround, model context is 8192")
    ap.add_argument("--limit", type=int, default=None, help="optional cap on the number of patches encoded")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}; input-format: {args.input_format}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # AutoModel discards lm_head.weight (UNEXPECTED in the load report).
    # That is correct for embedding extraction; we want the base hidden
    # states, not the next-token logits.
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()

    rows = collect(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} patches to embed with {MODEL_ID}", file=sys.stderr)

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
