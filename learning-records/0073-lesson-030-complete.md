# 0073 — Lesson 030 complete: Q3 Checkpoint — Write a Benchmark Report

**Date:** 2026-07-18
**Status:** Complete (self-reported by the user: "lesson 30 done")
**Curriculum:** Year 1 · Q3 · lecture 030 — the **Q3 checkpoint**, closing the Q3 arc (L021–L030).

## How it was marked complete
The user said "lesson 30 done" (alongside the request to create Lesson 031), with no EXIT-ticket text
pasted — so, per the L017–L029 precedent, there is no rubric score for the checkpoint lab; treated as
self-reported complete. Full teaching content, the reference benchmark report, and the verified numbers
are in the published record [[0072-lesson-030-published.md]].

## Retained (assumed, from the checkpoint design)
- **The benchmark-report contract:** a defensible report is Q2's five fair-comparison rules (same
  data / split / metric / budget / preprocessing) *plus* Q3's three: a **deployment-matched split**
  (temporal if timestamped, L021), an **audited feature set** / model info sheet (leakage, L022), and a
  **proven-then-explained gap** — a corrected significance test (L023) reported as a **budget curve**
  (L024) with an **inductive-bias explanation** (L025–L027).
- **A tie is a valid verdict.** On credit_g the stable 5×5 CV gave GBDT 0.780 vs MLP 0.772, mean gap
  **+0.0081**; naive paired t p=0.218, **corrected resampled t p=0.643** — **no significant winner**.
  Reporting "GBDT wins" from +0.008 is exactly the L023 mistake. The honest tie *is* the deliverable.
- **The budget curve can dip** on small data (best-by-validation ≠ monotone in test): the MLP's tuning
  dipped (0.819 → 0.805) because a 15-config search overfit a ~200-row validation slice — disclosed, not
  hidden.

## Q3 closed
L030 closes **Year 1 · Q3** (L021–L030): honest splits → leakage → significance → benchmark protocol →
three inductive biases → honest neural baseline → AutoML → the assembled benchmark report. The thesis
dossier verdict was updated "after L030 / Q3": single-table *search and architecture* returns are nearly
exhausted (L029 tie, L030 tie), so the remaining upside is **representational** — the bridge Q4 begins.

## Next
Lesson 031 — **Q4 opener**: *Embeddings for categoricals* (entity embeddings, Guo & Berkhahn 2016; target
encoding pitfalls). Published this same session; see [[0074-lesson-031-published.md]]. It opens Q4
(Consolidation & bridge to neural tabular, L031–L040) — the first taste of a *learned representation*.
