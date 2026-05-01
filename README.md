# agent-trajectory-metrics

Trajectory analysis for mini-SWE-agent v2 on the SWE-bench Verified leaderboard. Five top models, 500 tasks each, 2,500 trajectories. Two reports under the repo root: `report.md` is the writeup, `paper-summary.md` is a summary of the *Towards a Science of AI Agent Reliability* paper.

## Run

The CLI tool the brief asks for is `traj_metrics.py`:

```bash
python3 traj_metrics.py path/to/trajectory.json
# or pipe stdin
python3 traj_metrics.py < trajectory.json
```

Sample output on `astropy__astropy-12907` from Claude 4.5 Opus high:

```
System messages:     1
User messages:       1
Assistant messages: 32
Tool messages:      32
Exit messages:       1
======================
Total messages:     67
```

The brief's example output stops at `Total: 66`. Real mini-SWE-agent v2 trajectories carry one `exit` message holding the final patch, so the tool surfaces it as its own row to keep the total honest.

## End-to-end pipeline

```bash
python3 download.py            # 5 × 500 trajectories from public S3 (~1.3 GB)
python3 patches.py             # SWE-bench Verified ground-truth patches (HF)
python3 export_patches.py      # extract submitted patches from each trajectory
python3 aggregate.py           # data/features.csv (post-hoc trajectory features)
python3 prefix_features.py     # data/prefix_features.csv (turn-k prefix features)

# rsync data/patches/ data/predicted/ to a CUDA box, then on the box:
#   python3 embed_remote.py --data ./data            # Qwen2.5-Coder-1.5B
#   python3 embed_mellum.py --data ./data --batch 1  # JetBrains/Mellum-4b-sft-python
# rsync embeddings*.npy + embeddings*_index.csv back into ./data/

python3 similarity.py          # adds Qwen patch_sim
python3 similarity_mellum.py   # adds Mellum patch_sim_mellum
python3 classify.py            # post-hoc LOMO classifier (uses final patch)
python3 classify_compare.py    # base / +qwen / +mellum / +both feature subsets
python3 classify_prefix.py     # in-flight LOMO classifier (no final patch)
python3 plot.py                # 5 PNGs under plots/
python3 analysis.py            # every number cited in report.md
```

`data/leaderboard.json`, `data/features_with_both_sim.csv`, `data/embeddings.npy`, `data/embeddings_mellum.npy`, and the analysis transcripts are committed so the report numbers can be re-derived without re-downloading 1.3 GB of trajectories or rerunning the GPU jobs.

## Headline result

| Model                | pass@1 | $/inst | $/resolved | turns med |
| -------------------- | -----: | -----: | ---------: | --------: |
| Claude 4.5 Opus high |  76.8% |  $0.75 |      $0.98 |        29 |
| Gemini 3 Flash high  |  75.8% |  $0.36 |      $0.47 |        54 |
| MiniMax M2.5 high    |  75.8% |  $0.07 |      $0.10 |        52 |
| Claude 4.6 Opus      |  75.6% |  $0.55 |      $0.73 |        23 |
| GPT-5-2-Codex        |  72.8% |  $0.45 |      $0.62 |        31 |

The five top entries score within 4.2 points on pass@1; per-instance cost spans 10x; ranked by cost-per-resolved-task, the leaderboard inverts. A logistic regression on trajectory shape plus Qwen-Coder patch-similarity predicts the resolved flag at LOMO mean AUC 0.74; replacing Qwen with Mellum-4B-sft-python lifts it to 0.77. See `report.md` / `report.pdf`.

## Schema notes

Four of the five models use the standard `{role, content}` mini-SWE-agent v2 schema. GPT-5-2-Codex emits raw OpenAI Responses API objects directly into the `messages` array (no `role` field; `type=response` for assistant turns, `type=function_call_output` for tool returns). `traj_metrics.py` and the feature extractor handle both.

## Files

| Path                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `traj_metrics.py`                     | the CLI the task brief asks for, ~50 LOC                     |
| `features.py`                         | bash-command classifier + per-trajectory feature extractor   |
| `prefix_features.py`                  | features-up-to-turn-k for the in-flight classifier            |
| `download.py`                         | anonymous S3 puller for the 2,500 trajectories               |
| `patches.py`                          | SWE-bench Verified ground-truth patches (HF)                 |
| `export_patches.py`                   | extract submitted patches from trajectories                  |
| `embed_remote.py`                     | runs on the GPU box, Qwen2.5-Coder-1.5B mean-pool             |
| `embed_mellum.py`                     | runs on the GPU box, Mellum-4B-sft-python mean-pool           |
| `similarity.py`, `similarity_mellum.py` | join embeddings into per-(model, instance) cosine similarities |
| `classify.py`                         | post-hoc LOMO classifier (uses patch features)                |
| `classify_compare.py`                 | feature-subset ablation: shape / +qwen / +mellum / +both     |
| `classify_prefix.py`                  | in-flight LOMO classifier at k = 5, 10, 20, 30, 50, 75       |
| `plot.py`                             | survival, edit_churn, patch_sim, mellum_vs_qwen, prefix_auc   |
| `analysis.py`                         | reproduces the numbers in `report.md`                        |
| `report.md` / `.pdf`                  | the writeup, one page                                        |
| `paper-summary.md` / `.pdf`           | summary of arXiv:2602.16666                                   |
| `data/leaderboard.json`               | swebench.com snapshot used here                              |
| `data/features_with_both_sim.csv`     | 2,500 rows × 26 columns (the master features table)          |
| `data/embeddings.npy`                 | (3000, 1536) float32, Qwen-Coder-1.5B                         |
| `data/embeddings_mellum.npy`          | (3000, 3072) float32, Mellum-4B-sft-python                    |
| `plots/`                              | five PNGs referenced from `report.md`                        |
