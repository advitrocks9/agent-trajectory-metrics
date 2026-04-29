# Trajectory metrics for the top of mini-SWE-agent v2

The five top models on the SWE-bench Verified mini-SWE-agent v2 leaderboard score within 4.2 points on pass@1 (76.8 → 72.8). Their per-instance cost spans 10x, their median assistant turn counts span 2.3x, and a logistic regression on trajectory shape plus patch-content embeddings predicts the resolved flag at mean AUC 0.74 under leave-one-model-out cross-validation. Pass@1 is saturated at the top of the leaderboard. The trajectory is not.

I ran a Python tool over all 2,500 trajectories (5 models × 500 SWE-bench Verified instances). Four of the five use the standard `{role, content}` schema; GPT-5-2-Codex emits raw OpenAI Responses API objects in the `messages` array, so the parser has a separate branch. Each tool call is classified by bash verb (read / edit / search / test / git / install / other). Each submitted patch and the corresponding SWE-bench ground-truth patch are mean-pooled through Qwen2.5-Coder-1.5B on a local 4090 to give a 1536-d embedding; cosine similarity gives a per-instance patch-quality score.

| Model                 | pass@1 | $/inst | turns med | edits med | churn | patch L |
| --------------------- | -----: | -----: | --------: | --------: | ----: | ------: |
| Claude 4.5 Opus high  |  76.8% |  $0.75 |        29 |         5 |  0.00 |       6 |
| Gemini 3 Flash high   |  75.8% |  $0.36 |        54 |        11 |  0.28 |       6 |
| MiniMax M2.5 high     |  75.8% |  $0.07 |        52 |         5 |  0.00 |       6 |
| Claude 4.6 Opus       |  75.6% |  $0.55 |        23 |         3 |  0.00 |       4 |
| GPT-5-2-Codex         |  72.8% |  $0.45 |        31 |         3 |  0.00 |      15 |

`churn` is the median of `re-edits / repo-edits` for trajectories with at least one repo-path edit. The four models with `0.00` median have right-skewed churn distributions; their *means* are 0.11-0.19. Gemini is the outlier (mean 0.30): it edits the same file repeatedly, by design.

**1. Cost per resolved task spans 10x at near-equal accuracy.** Total leaderboard spend ÷ resolves: MiniMax $0.10, Gemini Flash $0.47, GPT-5-2-Codex $0.62, Claude 4.6 Opus $0.73, Claude 4.5 Opus $0.98. Ranking by this metric inverts the leaderboard. Both rankings are defensible. Only one is reported.

**2. Conditional resolve probability collapses with depth, at very different rates.** Define the survival curve `S_m(k) = P(resolved | T ≥ k)` for model `m` and turn count `T`. Claude 4.5 Opus high goes 77% → 46% → 9% → 0% at `k = 0, 50, 75, 100`. Gemini 3 Flash holds 76% → 73% → 68% → 50% over the same range. (See `plots/survival.png`; bands are 95% bootstrap CIs over 1,000 resamples.) A scaffold reading only the live turn count of a Claude run knows that by turn 75 it has below 10% chance of resolving and should escalate; the same logic is nearly useless on Gemini. This is the same signal that drives finding 1: the cost-of-failure / cost-of-success ratio is 1.84x for Claude 4.5 Opus and only 1.19x for Gemini, because Claude runs that fail are long, and long Claude runs almost always fail.

**3. A 14-feature classifier predicts `resolved` at mean AUC 0.74 across-models.** Features: action-type counts, edit churn, repo-edit count, distinct-files-edited count, call uniqueness, patch line-count and file-count, and patch-vs-ground-truth cosine similarity. Standardised, fit with L2-regularised logistic regression. Held-out fold AUCs: 0.78 (Claude 4.5 Opus high), 0.67 (Gemini 3 Flash), 0.77 (MiniMax), 0.74 (Claude 4.6 Opus); mean Brier 0.16. Patch similarity carries the largest absolute coefficient (+3.08); patch line-count and test-call count carry the largest negatives. Strikingly, the 13-feature model trained without `patch_sim` (so trajectory shape only) still hits mean AUC 0.67. Trajectory shape is genuinely informative; semantic patch similarity adds another 0.07 AUC on top.

**Caveats.** Patch embeddings are L2-normalised mean-pooled hidden states from a code LM, not from a model trained for retrieval; absolute cosine values are anisotropy-compressed near 1.0 and only the relative gap is meaningful. GPT-5-2-Codex has no per-instance resolved flags on swebench.com, so it sits out the labelled work. MiniMax's 250-turn cap is a hard limit from the harness, not the model. None of these say anything about *whether the patch is correct on the test cases that aren't in the trajectory*; that needs the patches actually run, which is the obvious follow-up.
