# Open questions

Things I would chase next if I had another two weeks.

1. **Does patch_sim survive an actual SWE-bench eval?** Run a 50-100 instance subset through the evaluation harness, regress `passes_test_suite ∈ {0, 1}` on `patch_sim_mellum`. If the slope is significant the metric is validated; if not, it is structural overlap only.

2. **Does the survival curve transfer to other scaffolds?** Same five models, but on Claude Code or OpenAI Codex submissions (when those exist on swebench). If `S_m(k)` looks the same shape, the curve is a property of the model. If it looks different, it is a property of the scaffold-model pair, which is a more interesting result.

3. **Same-size Mellum-vs-Qwen.** Need a comparable parameter budget on both sides. Qwen-Coder-3B vs Mellum-4B is the closest available. The +0.029 LOMO lift might shrink to nothing.

4. **Does turn-count transfer across models on the same instance?** For each instance, compute a "difficulty score" = mean turn count across models on that instance. Are the per-model surplus turns relative to that score informative about which model is mismatched to the task?

5. **Mellum's anisotropy.** The cosine values cluster near 1.0. A whitening transform (or just centring + Mahalanobis) might widen the gap. Cheap to try.

6. **Conditional-on-accuracy comparison.** Pair instances where two models have the same outcome. Among `(both resolved)` pairs, compare turn counts; same for `(both failed)`. Removes the resolved-flag confound from the cost-vs-quality story.

7. **Test-driven trajectory features.** Right now I count `n_test` as "ran a python file". But agents that write a regression test, run it, fix until it passes, then run the full suite probably resolve more often than agents that just `pytest` repeatedly. Detecting that pattern needs sequence modelling on the action stream, not bag-of-actions counts.

8. **Cost-aware metric design.** The cost-per-resolved-task ranking is not the same as the pass@1 ranking. There is probably a Pareto frontier of (accuracy, cost, latency) per scaffold. Worth quantifying.
