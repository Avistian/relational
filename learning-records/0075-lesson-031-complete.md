# 0075 — Lesson 031 complete: Embeddings for Categoricals (Q4 opener)

**Date:** 2026-07-19
**Status:** Complete (self-reported by the user: "lesson 31 done")
**Curriculum:** Year 1 · Q4 · lecture 031 — the Q4 opener (L031–L040 bridge to neural tabular).

## How it was marked complete
The user said "lesson 31 done" (alongside the request to create Lesson 032), with no lab EXIT-ticket text
pasted — so, per the L017–L030 precedent, there is no rubric score for the lab; treated as self-reported
complete. Full teaching content, the three viz, the lab, and the verified numbers are in the published
record [[0074-lesson-031-published.md]].

## Retained (assumed, from the lesson design)
- **Four categorical encodings and their trade-offs:** one-hot (safe-but-wide, no similarity), ordinal
  (compact but a **false order** — hurts linear/neural, tolerated by trees), target/mean encoding
  (compact, ordered by risk, scales to any cardinality — but built from the label), and **entity
  embeddings** (a learned d-dim dense vector per level; target encoding is the 1-D case).
- **Target encoding leaks unless out-of-fold** (M28): a signal-free near-unique id scored 0.891 AUC naive
  vs 0.504 out-of-fold — the L002/L022 leakage discipline applied to a *feature*.
- **Entity embeddings tie one-hot on a small flat table** (M29): 0.774 vs a *fair* one-hot MLP 0.798; the
  undertrained 0.728 baseline would have faked a +0.07 win (the L028 trap). The payoff is
  representational, at scale and on high-cardinality foreign keys — the atom of RDL.

## Next
Lesson 032 — **TabTransformer preview** (Huang et al. 2020; read the architecture figure). Published this
same session; see [[0076-lesson-032-published.md]]. Continues the Q4 bridge: the per-column entity
embeddings from L031 feed a self-attention Transformer that turns them into *contextual* embeddings.
