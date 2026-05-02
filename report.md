# Trajectory metrics for the top of mini-SWE-agent v2

The five top SWE-bench Verified models score within 4.2 points on pass@1 (76.8 → 72.8); their per-instance cost spans 10x. A logistic regression on trajectory shape plus patch-content features predicts the resolved flag at LOMO mean AUC 0.78 (95% bootstrap CI [0.76, 0.80]). The strongest single patch-content feature is, surprisingly, the Jaccard overlap of changed lines between predicted and ground-truth diff: it ties Mellum-4B-sft-python and beats Qwen-Coder-1.5B. Pass@1 is saturated at the top of the leaderboard; the trajectory is not.

I ran a Python pipeline over all 2,500 trajectories (5 models × 500 instances). Each tool call is classified by bash verb (read / edit / search / test / git / install / other); each predicted patch is compared to ground truth three ways: Jaccard overlap of changed lines, cosine similarity of mean-pooled Qwen2.5-Coder-1.5B embeddings, and cosine of mean-pooled JetBrains/Mellum-4B-sft-python embeddings (both encoders run on a 4090). GPT-5-2-Codex emits OpenAI Responses API objects in `messages` rather than `{role, content}`, so the parser has a separate branch.

| Model                 | pass@1 | $/inst | turns med | edits med | churn |
| --------------------- | -----: | -----: | --------: | --------: | ----: |
| Claude 4.5 Opus high  |  76.8% |  $0.75 |        29 |         5 |  0.00 |
| Gemini 3 Flash high   |  75.8% |  $0.36 |        54 |        11 |  0.28 |
| MiniMax M2.5 high     |  75.8% |  $0.07 |        52 |         5 |  0.00 |
| Claude 4.6 Opus       |  75.6% |  $0.55 |        23 |         3 |  0.00 |
| GPT-5-2-Codex         |  72.8% |  $0.45 |        31 |         3 |  0.00 |

**1. Cost per resolved task spans 10x.** Spend ÷ resolves: MiniMax $0.10, Gemini $0.47, GPT-5-2-Codex $0.62, Claude 4.6 Opus $0.73, Claude 4.5 Opus $0.98. This metric inverts the leaderboard.

**2. Conditional resolve probability collapses with depth, at very different rates.** `S_m(k) = P(resolved | T ≥ k)` for Claude 4.5 Opus high goes 77% → 46% → 9% → 0% at `k = 0, 50, 75, 100`. Gemini 3 Flash holds 76% → 73% → 68% → 50% over the same range (`plots/survival.png`, 95% bootstrap CIs from 1,000 resamples).

**3. Patch-content features dominate; encoder choice barely matters.** L2-regularised LR with leave-one-model-out across the four labelled models, bootstrap CIs over pooled out-of-fold predictions:

| Feature subset                    | Mean AUC |       95% CI       |
| --------------------------------- | -------: | -----------------: |
| trajectory shape only             |    0.659 |     [0.604, 0.664] |
| + patch shape (lines, files)      |    0.665 |     [0.617, 0.676] |
| + Jaccard over diff hunks         |    0.772 |     [0.744, 0.787] |
| + Qwen2.5-Coder-1.5B cosine       |    0.741 |     [0.702, 0.752] |
| + Mellum-4B-sft-py cosine         |    0.770 |     [0.737, 0.785] |
| + Jaccard + Qwen + Mellum         |    0.784 |     [0.759, 0.803] |

I tested Mellum because it is a JetBrains-trained Python code LM and SWE-bench Verified is all Python. It matches a 30-line Jaccard baseline. Mellum-only edges Qwen-only (0.770 vs 0.741, CIs overlap), but what these autoregressive code LMs capture in their mean-pooled hidden states on diff text is largely the same lexical overlap a hash-set comparison gives you for free. A retrieval-trained encoder might widen the gap. So might a sandboxed test-pass signal, which is the obvious next experiment.

**4. The in-flight signal is real within-model but does not transfer.** A classifier using only shape features visible by turn `k` (no patch features yet) gets within-model 5-fold CV AUC stable at ~0.60 across `k = 5, 10, 20, 30, 50` and drops only at `k = 75`. Under LOMO the same classifier decays from 0.59 at `k=5` to 0.45 at `k=50` (`plots/prefix_auc.png`). The signal exists; it just stops being model-agnostic as trajectories get long. A scaffold that wants to use this in production should learn per-model thresholds, not a shared one.

**5. Per-repo structure is large.** scikit-learn instances resolve at 90.6%, astropy at 58.0%; `patch_overlap` means range from 0.25 (matplotlib) to 0.48 (scikit-learn). Repo identity is a confounder for cross-repo transfer claims.

**6. Thrash detection: honest negative result, narrow within-model signal.** Longest run of identical non-edit bash commands (edits reset the streak), and max single-command count in any 10-command window. As a feature in the LOMO classifier it does not help: `+all` LOMO AUC drops from 0.784 to 0.779 with thrash columns added. Within-model the `window10_max` mean gap (unresolved minus resolved) is +0.26 / +0.01 / +0.31 / +0.34 for the four labelled models, small but consistent everywhere except Gemini. Strict thrash (depth ≥ 5) only catches three trajectories in the whole dataset, all MiniMax, all unresolved (one MiniMax run hits depth 59). Useful as a per-model in-flight flag, not as a portable feature.

**Validation.** I ran the SWE-bench docker harness locally on a 10-django sample per model (Gemini lost one report to a harness timeout, so 39 (model, instance) pairs across the four labelled models, `data/eval_validation.txt`); 39/39 matched the leaderboard. The leaderboard `resolved` field is, as one would hope, the harness output itself, so the classifier here genuinely predicts test-suite passage. The classifier's outputs are reasonably calibrated above 0.5 and slightly overconfident in the 0.10-0.35 range (`plots/calibration.png`), the regime where one would actually want to escalate or cut.

**Caveats.** Patch embeddings are mean-pooled hidden states, not from retrieval-trained encoders; absolute cosine values are anisotropy-compressed near 1.0 and only the relative gap is meaningful. Mellum-4B vs Qwen-1.5B is not size-controlled. GPT-5-2-Codex has no per-instance resolved flags on swebench.com and is excluded from findings 2, 3, 4, 6 (the cost finding uses leaderboard pass@1 and is the only one that includes it).
