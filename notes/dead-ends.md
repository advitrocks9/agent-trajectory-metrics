# Things I tried that didn't work or that I dropped

A scratch log of dead ends and parking-lot ideas. Mostly here so that future-me does not retry them blind.

## Sentence-transformer embedders

Tried to use `jinaai/jina-embeddings-v3` and `BAAI/bge-large-en-v1.5` first because they are actually trained for retrieval (cosine similarity is meaningful). Neither was cached on prannayk-gpu-1 and downloading both would have eaten an evening. Ended up using Qwen2.5-Coder-1.5B instead (already cached) with mean-pooled hidden states. Mellum-4b-sft-python was also already cached, so I used that as the second encoder.

Verdict: it would still be worth trying jina-v3 to see whether the retrieval-trained loss gives a wider resolved/unresolved cosine gap than the autoregressive mean-pool. The textual `patch_overlap` baseline (Jaccard over diff hunks) actually ties Mellum (`+overlap` 0.772 vs `+mellum` 0.770 LOMO AUC), which made me suspect the LM embeddings are mostly recovering surface lexical overlap rather than something deeper.

## Edit churn at the line level

First attempt at edit churn tried to track which *lines* of which files were edited at each step, by simulating the bash command (apply the sed pattern, replay the heredoc, etc.). Gave up after an hour: there are too many edge cases, agents do things like `cat file | grep ... > file` (read the file, transform, write back) which would need a shell-aware parser to reproduce.

Settled for the file-level proxy: `repo_re_edits / repo_edits`. It is coarse but reproducible and not nonsense.

The real `Agentic Edit Churn` metric the project description names probably wants line-level tracking. To do that properly I would either run the agent commands inside a sandbox and diff the working tree after each step (slow but correct), or build a small bash-aware abstract interpreter (fast but lots of work). Both are further than I could go in a week.

## Action-type classifier validation

The bash-verb classifier is regex. I never validated it against hand-labelled commands. Spot-checked ~30 commands and the categories looked right. The `+overlap` finding makes me less worried about its precision: the strongest signals turned out to be patch-level, not action-counting.

## Same-size encoder control for the Mellum vs Qwen comparison

Mellum-4B vs Qwen-Coder-1.5B is not a clean comparison; the +0.029 LOMO lift might be partly the parameter budget. The natural control is Qwen-Coder-3B (closer in size to Mellum-4B) or another ~4B code model. Qwen-Coder-3B is not cached and pulling it pushes /home over the comfort line, so leaving this for follow-up.

## Things that landed late

Two items were on this list as "skipped"; both shipped.

* SWE-bench docker validation. `validate_eval.py` runs the SWE-bench harness on a 10-django sample per labelled model and writes `data/eval_validation.txt`. 39/39 of the labelled (model, instance) pairs match the leaderboard. The killer experiment also extends this to a 50-instance balanced subset across 4 models with per-test-pass-fraction features, see `data/execution.csv` and report.md finding 7.
* Calibration. `plot.py` builds a reliability diagram for the +all LOMO classifier (`plots/calibration.png`) and prints ECE = 0.027 over 10 equal-width bins.
