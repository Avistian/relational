# 0079 — Lesson 033 complete: When to Stop Feature Engineering

**Date:** 2026-07-21
**Status:** Complete (self-reported by the user: "lesson 33 done")
**Curriculum:** Year 1 · Q4 · lecture 033 — third lesson of the Q4 bridge (L031–L040).

## How it was marked complete
The user said "lesson 33 done" (alongside the request to create Lesson 034), with no lab EXIT-ticket text
pasted — so, per the L017–L032 precedent, there is no rubric score for the lab; treated as self-reported
complete. Full teaching content, the diminishing-returns viz, the Tier-A `credit_g` FE-budget lab, and the
verified numbers are in the published record [[0078-lesson-033-published.md]].

## Retained (assumed, from the lesson design)
- **The feature-engineering trade-off (Domingos 2012):** "feature engineering is the key" (most early
  gains + effort live in features) pulls one way; "overfitting has many faces" (each feature raises
  variance) and "more data beats a cleverer algorithm" pull the other. Returns diminish, then go negative.
- **The stopping rule:** measure CV score *with* its σ after each feature; **stop when the marginal gain
  Δ < σ** (it has entered the noise band), not when you run out of ideas. Reallocate the budget to more
  data, then tuning, then an honest significance check (L023).
- **Verified evidence (credit_g, model held fixed):** GBDT peaks at k=3 (0.7911, +0.0046 over the 0.7865
  baseline — inside the ±0.032 band, *not* significant) then **declines to 0.7659 by k=8**, below baseline
  (the overfitting tax); the linear model drifts only +0.006, also within noise. No measurable single-table
  FE gain on credit_g.
- **FE value is conditional (M32/M33):** huge for a weak model missing the hypothesis (L009: Ridge
  0.64→1.00), marginal-then-negative for a strong model already fed the informative columns — always ask
  "diminishing returns *for which model?*"
- **Thesis hook:** the single-table FE ceiling is real and cheap to hit; the features that would still pay
  are aggregates *across related tables* (DFS-style, L009). RDL aims to *learn* those relational aggregates
  end-to-end — "the returns moved across the join."

## Next
Lesson 034 — **Relational data without RDL** (Kimball star schema & joins; sketch a 3-table join).
Published this same session; see [[0080-lesson-034-published.md]]. Opens the L034–L035 "relational data
without RDL" thread that leads into L035 "What joins destroy" (Fey ICML 2024 §2).
