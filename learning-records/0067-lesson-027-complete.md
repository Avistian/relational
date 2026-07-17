# 0067 — Lesson 027 complete: Inductive Bias — Uninformative Features

**Date:** 2026-07-17
**Status:** Complete (self-reported by the user: "lesson 27 done")
**Curriculum:** Year 1 · Q3 · lecture 027 — the last of the three inductive-bias mechanism lessons
(the Grinsztajn arc 024–027).

## How it was marked complete
The user said "lesson 27 done" (alongside the request to create Lesson 028), with no EXIT-ticket text
pasted — so, per the L017–L026 precedent, there is no rubric score for the lab; treated as
self-reported complete. Full teaching content and verified numbers are in the (backfilled) published
record [[0066-lesson-027-published.md]].

## Retained (assumed, from the lesson design)
- **Implicit feature selection:** a tree grows greedily, splitting only on the highest-*gain* feature,
  so a pure-noise column (best split ≈ 0 gain) is never chosen near the top — the tree ignores junk as
  a side-effect of how it grows. Root-split gain was ~118× higher on informative than junk features.
- **Why the MLP is fragile:** every feature enters the first linear layer, and (being rotationally
  invariant, L026) the MLP has no gate to tell a useful axis from a useless one; per Ng (2004) it
  needs ≥ linearly more samples per junk feature, so with finite data it leaks capacity onto noise.
- **The two ablations:** adding junk widens the tree–MLP gap (MLP lost 0.084 vs GBT 0.032 over 100
  junk columns, reversing the ranking); removing junk shrinks it (helps the MLP more).
- **The three biases are one story:** smoothness (L025), rotation (L026), uninformative features
  (L027) compound; typical tabular data has all three at once. This closes the Grinsztajn arc.

## Next
Lesson 028 (MLP & ResNet tabular baselines — Gorishniy 2021 §3.2). Published this same session; see
[[0068-lesson-028-published.md]]. It pivots Q3 from *why trees win* into *how to build the honest
neural contestant* that an RDL result must actually beat.
