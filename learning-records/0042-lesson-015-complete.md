# Lesson 015 complete — LightGBM

User: **"lesson/lab 15 done"** (2026-07-04).

Sixth Q2 unit closed (curriculum lec 015). Lab `labs/0015-lightgbm.ipynb` ran clean end-to-end (all CHECK + EXIT); solution executed and gitignored during publish.

**Rubric (lab-authoring, 0–2 × 5 axes):**
- Correctness — 2 (GOSS estimate unbiased; num_leaves sweep + LGBM/XGB CV all pass).
- Leakage discipline — 2 (relkit 5-fold PR-AUC; no full-data fit).
- Conceptual takeaway — 2 (leaf-wise capacity = `num_leaves`; GOSS keeps large-|g|, samples small-g, amplifies (1−a)/b; honest "edge, not magic" on small data).
- Mid-zone effort — 2 (crucial fragment = `goss_weights` by hand; caught the `b·|rest|` vs `b·n` bias bug via CHECK).
- Reproduction — 1 (credit_g table matched within tolerance; speed table is stretch/ungraded).
- **Total ≈ 9/10.**

**Retention signal:** GOSS unbiasedness + the (1−a)/b amplification, and leaf-wise-overfits-small-data, both landed. L014 histogram/candidate-threshold callback closed into L015 subtraction trick.

**Carry-forward for L016 (CatBoost):** the *ordering/permutation* discipline in CatBoost (ordered target statistics + ordered boosting) is the same target-/temporal-leakage instinct trained in L002 (PIT) and L004 (grouped/nested CV) — call it back explicitly. High-cardinality categorical foreign keys are the single-table stand-in for relational identity, so CatBoost is a direct thesis baseline.
