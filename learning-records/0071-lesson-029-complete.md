# 0071 — Lesson 029 complete: Manual Feature Engineering vs AutoML

**Date:** 2026-07-18
**Status:** Complete (self-reported by the user: "lesson 29 done")
**Curriculum:** Year 1 · Q3 · lecture 029 — the last modelling lecture before the L030 Q3 checkpoint.

## How it was marked complete
The user said "lesson 29 done" (alongside the request to create Lesson 030), with no EXIT-ticket text
pasted — so, per the L017–L028 precedent, there is no rubric score for the lab; treated as
self-reported complete. Full teaching content and the verified numbers are in the published record
[[0070-lesson-029-published.md]].

## Retained (assumed, from the lesson design)
- **CASH:** AutoML jointly searches *which algorithm* (a top-level categorical hyperparameter) and its
  hyperparameters, selecting the config with the best **validation** score (Auto-sklearn uses Bayesian
  optimization / SMAC; the lab used random search as an installable stand-in).
- **Two extensions over Auto-WEKA:** meta-learning **warm-start** (start the search from configs that won
  on similar past datasets, found via meta-features) and automated **ensemble construction** (Caruana
  greedy selection over the pool already trained — free diversity, +0.007 on credit_g).
- **What it does NOT do:** no domain feature engineering, no change to the representation — it tunes a
  model on top of an already-flattened table.
- **The honest tie:** on credit_g the big jump was tuning *at all* (default XGB 0.775 → tuned 0.806,
  +0.031); a 4-algorithm AutoML with ensembling then only **tied** the tuned XGB (0.803, bands overlap)
  at far higher compute. AutoML buys automation/robustness, not new accuracy.

## Next
Lesson 030 — **Q3 checkpoint** (Grinsztajn 2022 full; write a 1-page benchmark report). Published this
same session; see [[0072-lesson-030-published.md]]. It consolidates the whole Q3 arc (L021–L029) into one
defensible benchmark report.
