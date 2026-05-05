# Trajectory metrics for the top of mini-SWE-agent v2

The five top SWE-bench Verified models score within 4.2 points on pass@1 (76.8 -> 72.8); their per-instance cost spans 10x and the cost-per-resolved-task ranking inverts the pass@1 leaderboard (MiniMax $0.10 vs Claude 4.5 Opus high $0.98). The conditional resolve probability `P(resolved | T >= k)` collapses very differently across models: Claude 4.5 Opus high goes 77% -> 9% by k=75 while Gemini 3 Flash holds 76% -> 68% across the same range. The in-flight prefix classifier on the first k assistant turns (no patch features) holds 0.58-0.60 AUC under within-model `GroupKFold(instance_id)` up to k=20 and decays to 0.41 at k=75 under leave-one-model-out: cross-model transfer is what breaks. Pass@1 is saturated at the top of the leaderboard; the trajectory is not.

For each of 2,500 trajectories (5 models * 500 instances) `aggregate.py` extracts action-counts from bash verbs (read / edit / scratch_write / search / test / git / install / other), edit churn over repo paths, and patch shape. Each predicted patch is compared to its SWE-bench gold patch three ways: indentation-aware Jaccard over `(file, +/-, payload)` triples in the unified diff, cosine of mean-pooled Qwen2.5-Coder-1.5B hidden states, and cosine of mean-pooled JetBrains/Mellum-4B-sft-python hidden states (both encoders run on a 4090). GPT-5-2-Codex emits OpenAI Responses API objects in `messages` rather than `{role, content}`, so the parser branches on that.

| Model                 | pass@1 | $/inst | turns med | edits med | churn |
| --------------------- | -----: | -----: | --------: | --------: | ----: |
| Claude 4.5 Opus high  |  76.8% |  $0.75 |        29 |         5 |  0.00 |
| Gemini 3 Flash high   |  75.8% |  $0.36 |        54 |        11 |  0.28 |
| MiniMax M2.5 high     |  75.8% |  $0.07 |        52 |         5 |  0.00 |
| Claude 4.6 Opus       |  75.6% |  $0.55 |        23 |         3 |  0.00 |
| GPT-5-2-Codex         |  72.8% |  $0.45 |        31 |         3 |  0.00 |

**1. Cost per resolved task spans 10x.** Spend ÷ resolves: MiniMax $0.10, Gemini $0.47, GPT-5-2-Codex $0.62, Claude 4.6 Opus $0.73, Claude 4.5 Opus $0.98. This metric inverts the leaderboard.

**2. Conditional resolve probability collapses with depth, at very different rates.** `P(resolved | T >= k)` for Claude 4.5 Opus high goes 77% -> 46% -> 9% -> 0% at `k = 0, 50, 75, 100`. Gemini 3 Flash holds 76% -> 73% -> 68% -> 50% over the same range (`plots/survival.png`, pointwise 95% bootstrap CIs from 1,000 resamples per k). This is a conditional mean, not a Kaplan-Meier survival estimator: trajectories that hit the step cap (`exit_status == "LimitsExceeded"`) are treated as observed failures at their stopping turn, not censored. The bands are pointwise marginal CIs, not a simultaneous band; the cross-curve claim ("Claude 4.5 collapses faster than Gemini") is informal in that strict sense.

**3. Post-hoc diagnostic on patch-vs-gold similarity.** This subsection asks: once you have the submitted patch and the SWE-bench gold patch, how well does patch-vs-gold similarity predict the `resolved` flag? Every "+X" feature here is computed against the gold patch (Jaccard of changed-line triples, Qwen cosine, Mellum cosine), so this is a diagnostic of how predictable `resolved` is once both patches are in hand, not an in-flight classifier. L2-regularised LR with leave-one-model-out across the four labelled models, bootstrap CIs over pooled out-of-fold predictions, resampled by `instance_id` so the 4 model rows for each task move together:

| Feature subset                    | Mean AUC |       95% CI       |
| --------------------------------- | -------: | -----------------: |
| shape                             |    0.659 |     [0.602, 0.665] |
| shape+thrash                      |    0.655 |     [0.596, 0.661] |
| shape+thrash+patch shape          |    0.662 |     [0.608, 0.679] |
| + Jaccard over (file, +/-, payload)| 0.758   |     [0.721, 0.781] |
| + Qwen2.5-Coder-1.5B cosine       |    0.737 |     [0.685, 0.758] |
| + Mellum-4B-sft-py cosine         |    0.764 |     [0.721, 0.787] |
| + Jaccard + Qwen + Mellum         |    0.775 |     [0.742, 0.803] |

Each "+X" row sits on top of the same shape+thrash+patch-shape baseline, so the +X delta is the named feature's contribution above that baseline. The earlier table presented these as additions to shape alone; thrash was silently in every row.

Paired bootstrap on instance_id (n=2000, see `data/classify_compare.txt`):

* Mellum vs Qwen,    +mellum - +qwen    = +0.031, 95% CI [+0.016, +0.046], excludes 0.
* Mellum vs Jaccard, +mellum - +overlap = +0.004, CI [-0.021, +0.029], includes 0.
* Jaccard vs Qwen,   +overlap - +qwen   = +0.027, CI [-0.001, +0.054], includes 0.

Mellum beats Qwen. The indentation-aware Jaccard (now keyed on `(file, +/-, payload)` with leading whitespace preserved -- previous versions stripped indentation) ties Mellum within the paired CI; once indentation matters, the Jaccard edge over Qwen also crosses zero. The interpretation: a 30-line set comparison that respects Python-significant indentation captures most of what mean-pooled Mellum extracts from the same diff, and neither cleanly beats Qwen on this 2000-row pool. A retrieval-trained encoder (jina-v3, BGE) might widen the gap. So might a sandboxed test-pass signal, which is finding 7.

**4. The within-model in-flight signal holds, the cross-model one doesn't.** Prefix-AUC at k=5 is 0.59 under LOMO; at k=50 it's 0.45 and at k=75 it's 0.41. Within-model `GroupKFold(instance_id)` (`classify_prefix.py`, 5 folds) holds 0.59 / 0.60 / 0.60 across k=5, 10, 20, dips to 0.57 at k=30, holds 0.59 at k=50, and collapses at k=75 (0.39, fold count 10 because Claude 4.5 Opus high and Claude 4.6 Opus have too few trajectories that reach 75 turns). Between-model transfer is what fails as trajectories grow long. I have not run the per-model threshold study a production scaffold would need; that's left for follow-up.

**5. Per-repo structure is large.** scikit-learn (n=128, 32/model) resolves at 90.6%, astropy (n=88, 22/model) at 58.0%; mean `patch_overlap` ranges from 0.12 (pylint-dev, n=40) to 0.48 (scikit-learn). django dominates (n=924, 231/model) so pooled numbers track django. mwaskom (n=8) and pallets (n=4) are too small to mean-quote. Repo identity is a confounder for cross-repo transfer claims.

**6. Thrash detection: honest negative result.** Longest run of identical non-edit bash commands (edits reset the streak), and max single-command count in any 10-command window. As a feature in the LOMO classifier it does not help: shape+thrash AUC is 0.655 vs shape alone 0.659. Within-model the `window10_max` mean gap (unresolved minus resolved) is +0.26 / +0.00 / +0.30 / +0.34 for the four labelled models, small and absent in Gemini. Strict thrash (depth >= 5) catches two trajectories in the whole 2,500: scikit-learn-15100 (MiniMax, depth 8) and scikit-learn-25102 (MiniMax, depth 59), both unresolved. Per-model the depth distribution differs, but across models the threshold doesn't transfer.

**7. Test-pass fraction as a feature, on a 50-instance subset.** I picked 50 instances stratified across 0/1/2/3/4-of-4-models-resolve buckets (`data/exec_subset.json`) and built per-model prediction JSONLs (`data/exec_predictions/*.jsonl`) for the SWE-bench harness in a Modal sandbox (`run_execution.py predictions | harness | merge`; the harness step calls `swebench.harness.run_evaluation` with `--cache_level instance --max_workers 4`). The new feature is `tests_pass_frac = (FAIL_TO_PASS.success + PASS_TO_PASS.success) / total`. The harness reports for all four labelled models are in `data/exec_reports/`; one instance (`pylint-dev__pylint-7277`) errored on Modal sandbox creation across all four models and is excluded, leaving 192/200 (model, instance) rows. Re-running `classify_compare.py` on this subset with a `+execution` feature, instance-grouped paired bootstrap (`data/classify_compare_with_execution.txt`):

| Subset                  | mean AUC | 95% CI         | per-fold (Opus5h, Gemini, MiniMax, Opus6) |
|-------------------------|----------|----------------|-------------------------------------------|
| shape (subset only)     | 0.510    | [0.459, 0.619] | 0.563 0.367 0.540 0.569                   |
| +overlap                | 0.657    | [0.568, 0.752] | 0.710 0.589 0.683 0.646                   |
| +mellum                 | 0.592    | [0.509, 0.672] | 0.652 0.434 0.592 0.690                   |
| +all                    | 0.633    | [0.539, 0.729] | 0.687 0.559 0.653 0.633                   |
| +execution              | 0.578    | [0.495, 0.659] | 0.652 0.480 0.523 0.656                   |
| +all+execution          | 0.642    | [0.548, 0.740] | 0.705 0.618 0.622 0.625                   |

`tests_pass_frac` does add signal over shape alone (0.578 vs 0.510) but is weaker than `patch_overlap` (0.657) on this 192-row subset; paired delta `+execution` vs `+overlap` is -0.086, CI [-0.172, -0.003] (excludes 0). Stacking execution on top of all three patch features gives `+all+execution` 0.642 vs `+all` 0.633, paired delta +0.010 with CI [-0.017, +0.036] (includes 0). On this subset the cheap patch-vs-gold Jaccard already captures what the expensive harness run adds; running the tests is redundant when you have the gold patch.

**Validation.** I ran the SWE-bench docker harness locally on a 10-django sample per model (Gemini lost one report to a harness timeout, so 39 (model, instance) pairs across the four labelled models, `data/eval_validation.txt`); 39/39 matched the leaderboard. The leaderboard `resolved` field is the harness output itself, so on the validated subset the classifier predicts the harness verdict directly. The full 2000-row labels are taken on faith from the leaderboard. The classifier's outputs are calibrated to ECE = 0.027 over 10 equal-width bins (`plots/calibration.png`) and slightly overconfident in the 0.10-0.35 range, the regime where one would actually want to escalate or cut.

**Caveats.**
* Patch embeddings are mean-pooled hidden states from `AutoModel`, which discards Mellum's `lm_head.weight` (logged as UNEXPECTED in `data/embed_mellum.log`). That is correct for embedding extraction but the SFT-specific specialisation in the LM head is lost. Mean-pool is also the wrong default for a causal LM: the conventional choice is the last non-pad token's hidden state, which has attended to the full sequence. `embed_mellum_lasttoken.py` is the script for that. I tried to rerun it on the local 16 GB Mac (MPS, fp16, batch=4, max-len=2048) and the process pinned at 16 GB resident memory and made it through 4 of 3000 patches in 14 minutes before I killed it; the run needs the 4090 the original embeddings were produced on. Treat the +mellum row in the table above as the mean-pool, max-len=1024, bare-diff baseline; a last-token, max-len=2048, FIM-formatted re-embed is the obvious next step and would only narrow or widen the Mellum-vs-Jaccard tie further. The committed `embeddings_mellum.npy` was also produced with the bare-diff input, no `<filename>` / `<fim_*>` formatting; that is out-of-distribution for an FIM-trained model. `embed_mellum.py` now defaults to `--input-format fim`.
* Mellum-4B vs Qwen-1.5B is not size-controlled; some of the +0.031 paired delta is parameter count.
* Tokenizer `max_length` differs across the two encoders. `embed_remote.py` runs Qwen at `max_length=2048`; the committed `embeddings_mellum.npy` was produced at `max_length=1024` (a GPU-sharing workaround on the original run; `embed_mellum.py` now defaults to 2048). On long Django diffs that exceed 1024 Qwen tokens, Qwen sees more of the patch than Mellum did. The asymmetry biases the Mellum row downward, so the +mellum vs +qwen paired delta of +0.031 is, if anything, an underestimate of the gap a fair re-run would show. A re-embed at `max_length=2048` is the obvious follow-up; the existing committed Mellum embeddings stay in the report as the as-run number.
* LOMO is across 4 models on the same 500 SWE-bench instances. Task difficulty leaks through trajectory shape (turn count, edits) which is correlated across same-instance rows in train and test folds. The within-model 5-fold GroupKFold in `classify_prefix.py` is leakage-free; the LOMO AUC here is therefore an upper bound on cross-model generalisation.
* The conditional-resolve curves in finding 2 are pointwise per-k bootstraps of `P(resolved | T >= k)`, not a Kaplan-Meier survival estimator. `LimitsExceeded` trajectories are treated as observed failures at their stopping turn rather than censored. KM with Greenwood's formula would give the proper survival construction; renaming the panel to "conditional resolve probability" rather than fitting KM is the conservative choice taken here.
* GPT-5-2-Codex has no per-instance resolved flags on swebench.com and is excluded from findings 2, 3, 4, 6, 7. The cost finding uses leaderboard pass@1 and is the only one that includes it.
