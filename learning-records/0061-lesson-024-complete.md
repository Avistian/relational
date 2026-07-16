# Lesson 024 complete — The Grinsztajn benchmark

**Date:** 2026-07-16
**Status:** Complete (marked done on the user's word)

User signalled completion: **"lesson 24 done"** (in the same message that asked to build L025).
No EXIT-ticket text was pasted, so — following the L017/L018/L020/L021/L022 precedent — **no numeric
rubric score** is recorded; the unit is marked done on the user's word.

**What L024 covered.** The Grinsztajn benchmark *protocol* (§3–4): the dataset-curation funnel, the
random-search **budget curve** (select by validation, report test, average over orderings) instead of
one tuned number, and **per-dataset min-max normalization** so a 45-dataset suite can be averaged.
Verified live on real credit-g: GBT above MLP at every budget (default gap +0.062 → tuned +0.015).
Lesson + lab: [[0060-lesson-024-published.md]].

**Grinsztajn arc progress.** L024 built the *measuring instrument*. The next three lessons dissect the
*mechanism* — the three inductive biases previewed in L019:
- **L025 — smoothness** (this session): NNs are biased toward smooth functions; tabular targets are
  irregular.
- L026 — rotation (§5.3).
- L027 — uninformative/junk features (§5.4).

**Next.** L025 (Inductive bias: smoothness — Grinsztajn 2022 §5.2) published this session; see
[[0062-lesson-025-published.md]].
