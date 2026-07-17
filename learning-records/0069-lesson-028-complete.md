# 0069 — Lesson 028 complete: MLP & ResNet Tabular Baselines

**Date:** 2026-07-17
**Status:** Complete (self-reported by the user: "lesson 28 done")
**Curriculum:** Year 1 · Q3 · lecture 028 — the first PyTorch lesson; pivots Q3 from *why trees win*
(the Grinsztajn arc 024–027) into *building the honest neural baseline*.

## How it was marked complete
The user said "lesson 28 done" (alongside the request to create Lesson 029), with no EXIT-ticket text
pasted — so, per the L017–L027 precedent, there is no rubric score for the lab; treated as
self-reported complete. Full teaching content and the verified numbers are in the published record
[[0068-lesson-028-published.md]].

## Retained (assumed, from the lesson design)
- **The residual block:** `ResNetBlock(x) = x + f(x)` where `f = BatchNorm → Linear → ReLU → Dropout →
  Linear → Dropout`. The one line `x +` is the whole difference between a ResNet and a plain MLP.
- **The degradation problem:** a plain deep MLP gets *worse* with depth and its **training** accuracy
  falls too (repro: plain train 1.000 → 0.927 over depth 1→32), which rules out overfitting — an
  optimization failure (He et al. 2015), **not** vanishing gradients (both nets use BatchNorm).
- **Why the skip fixes it:** it makes the identity map free (`f ≈ 0` is trivial), so a deep ResNet can
  always fall back to a shallow solution — adding depth can't hurt.
- **Honest baseline / no universal winner:** on small categorical credit_g a tuned GBDT still leads
  (0.793) and the two neural baselines tie (MLP 0.752 ≈ ResNet 0.743). Gorishniy's contribution is
  methodological — compare against a *properly-tuned* MLP/ResNet or your "SOTA" claim is worthless.

## Next
Lesson 029 (Manual FE vs AutoML — Feurer et al. 2015, Auto-sklearn). Published this same session; see
[[0070-lesson-029-published.md]]. It asks the skeptic's question — "can a machine just do all the
modelling for you?" — and runs the fair AutoML-vs-tuned-XGBoost fight, closing Q3's modelling thread
before the L030 checkpoint.
