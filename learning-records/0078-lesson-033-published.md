# 0078 — Lesson 033 published: When to Stop Feature Engineering

**Date:** 2026-07-21
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 033 — matches the CURRICULUM row (033: "When to stop feature
engineering · Domingos 2012 (few useful things) · FE time budget exercise"). Third lesson of Q4
(Consolidation & bridge to neural tabular, L031–L040). Plan entry: `plan/year-1.md` §033.
**Primary reading:** Pedro Domingos 2012, *A Few Useful Things to Know about Machine Learning*, CACM
55(10):78–87 — the sections "Feature engineering is the key", "More data beats a cleverer algorithm", and
"Overfitting has many faces" ([PDF](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)).

## Single skill
Allocate a fixed modeling budget between feature engineering and model tuning, and justify the split with
the feature-engineering trade-off — most of the early gains live in features, but their returns
**diminish** and each added feature raises the overfitting/variance tax, so you **stop when the marginal
held-out gain sinks into the CV noise band** (Δ < σ), not when you run out of ideas.

## Why this was the ZPD
L031/L032 taught *how* to build ever-richer single-table representations (encodings, embeddings, contextual
attention); L033 is the curriculum's designated next step and the natural counterweight: *when to stop*. It
consolidates the Q4 bridge by naming the ceiling on single-table effort with a live experiment, and it is
the cleanest possible setup for the "relational data without RDL" thread (L034–L035): once the learner has
*measured* that manual single-table FE returns die (then go negative) against a competent model, "the value
moved across the join" lands as a demonstrated fact, not a slogan. It also reactivates the L009 (DFS),
L023 (significance), and L028–L032 (tie pattern) threads.

## What shipped
- **Lesson** `lessons/0033-when-to-stop-feature-engineering.html` (~40 min, standard #17 thoroughness):
  full vocabulary (FE, marginal/diminishing returns, bias–variance/overfitting tax, CV noise band ±σ,
  modeling budget/opportunity cost, "more data beats a cleverer algorithm" — each from first principles) →
  Domingos' three "useful things" as a table → the L009 "why FE matters for a weak model" beat (reused
  `feature-viz.js`) → predict → the diminishing-returns experiment (new viz) → honest verdict table (FE
  value is conditional on model & data) → the budget rule (stop when Δ < σ, then reallocate) → teach-back →
  thesis bridge (returns moved across the join; RDL learns DFS aggregates) → 5 subtleties → 3 quizzes →
  primary reading → lab. Forward-references to L034/L035 and Y4 are plain text.
- **One new reusable viz (standard #9)** + one reuse:
  - `assets/fe-returns-viz.js` — the diminishing-returns curve: two **real** curves (GBDT + linear) each
    with a shaded ±1σ CV noise band, a "stop here" marker at the GBDT peak (k=3), model toggles, a
    band toggle, and a k-slider reading out Δ-vs-baseline and within-noise / exceeds-1σ. Numbers baked from
    `_verify_l033.py`.
  - Reused `assets/feature-viz.js` (L009) for the "engineered ratio turns a cloud into a line for a linear
    model" beat — the case *for* spending FE effort early.
  - Headless check `labs/_viz_check_l033.js` — **11/11 pass**. **Browser MCP unavailable** (headless env;
    no chrome-devtools server) → node verification only, consistent with L021–L032.
- **Lab** `labs/0033-when-to-stop-feature-engineering.ipynb` — **Tier A** (real `credit_g`), sklearn.
  Domingos is an *essay*, so the standard #18 scope (stated in the intro) is **operationalise the essay's
  three claims as one controlled experiment**, not implement an architecture. Three crucial fragments:
  Task 1 = the **feature-budget loop** (add hand features one at a time, record mean ± σ CV AUC); Task 2 =
  the **stopping rule** `first_within_noise` (first k whose marginal gain < σ); Task 3 = **model-dependence**
  (rerun with a linear model; the L009 asymmetry). Stretch points at the L023 corrected resampled t-test to
  prove the peak is not significant. Student blank (6 `____`, 0 outputs); solution executed clean &
  gitignored. Built via `_build_l033.py`; rendered to `labs/html/0033-*.html`.
- **Verify** `labs/_verify_l033.py` — ran clean; numbers below.

## Verified live (`_verify_l033.py` + executed solution; credit_g Tier A, seed 0, 5-fold, model held fixed)
- **GBDT (HistGradientBoosting)** CV ROC-AUC vs k hand features:
  `[0.7865, 0.7828, 0.7863, 0.7911, 0.7856, 0.7740, 0.7748, 0.7748, 0.7659, 0.7756, 0.7756]`
  (std ≈ 0.025–0.037). **Peaks at k=3 (0.7911, +0.0046 over the 0.7865 baseline — inside the ±0.032 band,
  not significant), then declines to 0.7659 by k=8 — below baseline** (the overfitting tax).
- **Linear (scaled logistic regression):** baseline 0.7913, total drift over 0→10 features **+0.006**, also
  within its ~0.02 band. Both models stay inside their noise envelopes — for opposite reasons (a tree
  already synthesises interactions; the dataset is fairly linear).
- The lab's stopping rule returns **k=1** on this curve (the very first feature's marginal change is already
  within noise), and the EXIT ticket reproduces the peak/decline exactly.

## Honest framings kept
- **"Feature engineering is the key" is conditional.** Huge for a weak model missing the hypothesis (L009:
  Ridge 0.64→1.00), marginal-then-negative for a strong model already fed the informative columns. Always
  ask "diminishing returns *for which model?*" (M33).
- **A +0.005 bump inside a ±0.03 band is not a win.** The correct report is "no measurable single-table FE
  gain on credit_g" — the L023/L030 discipline applied to features, not models (M32).
- **"Within noise" ≠ "useless."** A feature can still earn its keep for interpretability, robustness, or a
  simpler leak-safe pipeline; the budget rule is about *accuracy* returns.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l033-fe-returns` [misconception], `l033-thesis`).
- `assets/paper-deck.js` +1 (`domingos2012` — the three "useful things" + the honest stopping rule).
- `misconceptions.md` **M32** (FE always helps → diminishing then negative; stop at Δ<σ) and **M33**
  (diminishing returns is a dataset property → it is model-relative).
- `thesis-dossier.md` +1 ledger line (L033, FOR + BAR, C1/C4/C3 — quantifies the single-table ceiling) and
  **Current verdict updated to "after L033 / when to stop feature engineering."**
- `reference/glossary.html` — Q4 section +4 terms (diminishing returns, marginal return, CV noise band,
  FE budget rule).
- `RESOURCES.md` — Year 1 +1 (Domingos 2012 CACM PDF, with the sections to read).
- `lessons/manifest.json` regenerated → **33 entries** (L033 Q4, published); `labs/html/0033-*.html`
  rendered. `NOTES.md` Session 33 logged.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l033.js` 11/11 clean.

## Next
Lesson 034 — **Relational data without RDL** (Kimball star schema & joins summary; sketch a 3-table join),
opening the L034–L035 "relational data without RDL" thread that leads into L035 "What joins destroy"
(Fey ICML 2024 §2) — the bridge from "single-table returns are exhausted" (this lesson) to "here is the
structure flattening throws away."
