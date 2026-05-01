# Trajectory metrics for the top of mini-SWE-agent v2

The five top SWE-bench Verified models score within 4.2 points on pass@1 (76.8 → 72.8); their per-instance cost spans 10x. A logistic regression on trajectory shape plus Qwen-Coder patch-similarity predicts the resolved flag at LOMO mean AUC 0.74; replacing Qwen with JetBrains Mellum-4B-sft-python lifts that to 0.77. Pass@1 is saturated at the top of the leaderboard; the trajectory is not.

I ran a Python pipeline over all 2,500 trajectories (5 models × 500 instances). Each tool call is classified by bash verb (read / edit / search / test / git / install / other). Each submitted patch and the corresponding ground-truth patch are mean-pooled through Qwen2.5-Coder-1.5B and again through Mellum-4B-sft-python on a local 4090; cosine similarity gives two per-instance patch-quality scores. GPT-5-2-Codex emits OpenAI Responses API objects in `messages` rather than `{role, content}`, so the parser has a separate branch.

| Model                 | pass@1 | $/inst | turns med | edits med | churn | patch L |
| --------------------- | -----: | -----: | --------: | --------: | ----: | ------: |
| Claude 4.5 Opus high  |  76.8% |  $0.75 |        29 |         5 |  0.00 |       6 |
| Gemini 3 Flash high   |  75.8% |  $0.36 |        54 |        11 |  0.28 |       6 |
| MiniMax M2.5 high     |  75.8% |  $0.07 |        52 |         5 |  0.00 |       6 |
| Claude 4.6 Opus       |  75.6% |  $0.55 |        23 |         3 |  0.00 |       4 |
| GPT-5-2-Codex         |  72.8% |  $0.45 |        31 |         3 |  0.00 |      15 |

Median `churn` is `re-edits / repo-edits` for trajectories with ≥1 repo edit. The four `0.00` medians have right-skewed distributions (means 0.11-0.19); Gemini is the outlier at 0.30.

**1. Cost per resolved task spans 10x at near-equal accuracy.** Spend ÷ resolves: MiniMax $0.10, Gemini $0.47, GPT-5-2-Codex $0.62, Claude 4.6 Opus $0.73, Claude 4.5 Opus $0.98. This metric inverts the leaderboard.

**2. Conditional resolve probability collapses with depth, at very different rates.** `S_m(k) = P(resolved | T ≥ k)` for Claude 4.5 Opus high goes 77% → 46% → 9% → 0% at `k = 0, 50, 75, 100`. Gemini 3 Flash holds 76% → 73% → 68% → 50% over the same range (`plots/survival.png`, 95% bootstrap CIs over 1,000 resamples). The cost-of-failure / cost-of-success ratio is 1.84x for Claude 4.5 Opus and 1.19x for Gemini, because Claude runs that fail are long, and long Claude runs almost always fail.

**3. Mellum beats Qwen as a patch-quality encoder.** L2-regularised logistic regression with leave-one-model-out across the four labelled models:

| Feature subset                         | Mean AUC |  Opus5h |  Gemini | MiniMax |  Opus6 |
| -------------------------------------- | -------: | ------: | ------: | ------: | -----: |
| trajectory shape only                  |    0.659 |   0.705 |   0.548 |   0.700 |  0.683 |
| + patch shape (lines, files)           |    0.665 |   0.715 |   0.571 |   0.699 |  0.676 |
| + Qwen-Coder-1.5B cosine               |    0.741 |   0.781 |   0.670 |   0.771 |  0.741 |
| + Mellum-4B-sft-py cosine              |    0.770 |   0.805 |   0.711 |   0.789 |  0.773 |
| + both Qwen and Mellum                 |    0.769 |   0.804 |   0.711 |   0.790 |  0.770 |

Mellum and Qwen carry similar information (the joint model adds nothing over Mellum alone), but Mellum encodes it more discriminatively. The resolved/unresolved cosine gap is 0.06-0.10 with Mellum vs 0.04-0.07 with Qwen (`plots/mellum_vs_qwen.png`). A JetBrains-trained Python code LM gives a sharper patch-quality signal on a Python benchmark, despite never being trained for retrieval. Mellum is 4B vs Qwen 1.5B, so some of this is parameter count; controlling for scale is a follow-up.

**4. The in-flight signal decays with prefix length.** A classifier using only shape features visible by turn `k` (no patch features) gets LOMO mean AUC 0.59 at `k = 5`, 0.58 at `k = 20`, 0.54 at `k = 30`, 0.45 at `k = 50` (`plots/prefix_auc.png`). The discriminative features at low `k` (low n_test, low n_calls) get compressed away as trajectories grow long: the surviving population at `k ≥ 50` is dominated by hard cases that all look similar. Most of the operationally-useful in-flight signal is just the survival curve itself, i.e. turn-count alone.

**Caveats.** Patch embeddings are mean-pooled hidden states, not from retrieval-trained encoders; absolute cosine values are anisotropy-compressed near 1.0 and only the relative gap is meaningful. GPT-5-2-Codex has no per-instance resolved flags on swebench.com, so it is excluded from findings 2-4. None of these say whether the patch is correct on the SWE-bench test cases.
