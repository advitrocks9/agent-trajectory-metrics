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
python3 download.py        # 5 model folders × 500 trajectories from public S3 (~1.3 GB)
python3 patches.py         # SWE-bench Verified ground-truth patches from HuggingFace
python3 export_patches.py  # extract submitted patches from each trajectory
python3 aggregate.py       # data/features.csv, 2,500 rows × 24 columns
# rsync data/patches/ data/predicted/ to a CUDA box, then on the box:
#   python3 embed_remote.py --data ./data
# rsync embeddings.npy + embeddings_index.csv back into ./data/
python3 similarity.py      # data/features_with_sim.csv (adds patch_sim column)
python3 classify.py        # leave-one-model-out logistic regression, prints AUCs
python3 plot.py            # plots/{survival,edit_churn,patch_sim}.png
python3 analysis.py        # every number cited in report.md
```

`data/leaderboard.json`, `data/features.csv`, `data/features_with_sim.csv`, `data/embeddings.npy`, and the analysis transcripts are committed so the report numbers can be re-derived without re-downloading 1.3 GB of trajectories or rerunning the GPU job.

## Headline result

| Model                | pass@1 | $/inst | $/resolved | turns med | LOMO classifier AUC |
| -------------------- | -----: | -----: | ---------: | --------: | ------------------: |
| Claude 4.5 Opus high |  76.8% |  $0.75 |      $0.98 |        29 |                0.78 |
| Gemini 3 Flash high  |  75.8% |  $0.36 |      $0.47 |        54 |                0.67 |
| MiniMax M2.5 high    |  75.8% |  $0.07 |      $0.10 |        52 |                0.77 |
| Claude 4.6 Opus      |  75.6% |  $0.55 |      $0.73 |        23 |                0.74 |
| GPT-5-2-Codex        |  72.8% |  $0.45 |      $0.62 |        31 |                  -- |

The leaderboard ranks by pass@1; the five top entries are within 4.2 points. Per-instance cost spans 10x. Ranked by cost-per-resolved-task, the leaderboard inverts. A 14-feature logistic regression on trajectory shape plus patch-similarity-to-ground-truth predicts the resolved flag at AUC 0.74 across-models under leave-one-model-out CV. See `report.md` / `report.pdf` for the full table and caveats.

## Schema notes

Four of the five models use the standard `{role, content}` mini-SWE-agent v2 schema. GPT-5-2-Codex emits raw OpenAI Responses API objects directly into the `messages` array (no `role` field; `type=response` for assistant turns, `type=function_call_output` for tool returns). `traj_metrics.py` and the feature extractor handle both.

## Files

| Path                            | Purpose                                                    |
| ------------------------------- | ---------------------------------------------------------- |
| `traj_metrics.py`               | The CLI the task brief asks for, ~50 LOC                   |
| `features.py`                   | bash-command classifier + per-trajectory feature extractor |
| `download.py`                   | anonymous S3 puller for the 2,500 trajectories             |
| `patches.py`                    | SWE-bench Verified ground-truth patches (HF)               |
| `export_patches.py`             | extract submitted patches from trajectories                |
| `embed_remote.py`               | runs on the GPU box, mean-pools Qwen2.5-Coder-1.5B         |
| `similarity.py`                 | joins embeddings into features_with_sim.csv                |
| `classify.py`                   | LOMO logistic regression                                   |
| `plot.py`                       | survival curves + edit-churn + patch-sim plots             |
| `analysis.py`                   | reproduces the numbers in `report.md`                      |
| `report.md` / `.pdf`            | the writeup, one page                                      |
| `paper-summary.md` / `.pdf`     | summary of arXiv:2602.16666                                |
| `data/leaderboard.json`         | swebench.com snapshot used here                            |
| `data/features_with_sim.csv`    | 2,500 rows × 25 columns                                    |
| `data/embeddings.npy`           | (3000, 1536) float32                                       |
| `plots/`                        | three PNGs referenced from `report.md`                     |
