# 0070 — Lesson 029 published: Manual Feature Engineering vs AutoML

**Date:** 2026-07-17
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 029 — the last modelling lecture before the L030 Q3 checkpoint
(1-page benchmark report). Matches the CURRICULUM Q3 row (029: Manual FE vs AutoML, Feurer et al. 2015
Auto-sklearn skim, "compare to tuned XGB").
**Core paper:** Feurer, Klein, Eggensperger, Springenberg, Blum & Hutter 2015, *Efficient and Robust
Automated Machine Learning* (NeurIPS 2015, **Auto-sklearn**), §2 (CASH) + §3 (the two extensions).
Background: Thornton et al. 2013 (Auto-WEKA — the CASH framing) and Caruana et al. 2004 (ensemble
selection).

## Single skill
Understand what AutoML automates — the **CASH** problem (jointly choosing the algorithm *and* its
hyperparameters, selected by validation) solved by Bayesian optimization (SMAC), plus **meta-learning
warm-start** and **automated ensemble construction** (Caruana) — and run a fair AutoML-vs-tuned-XGBoost
comparison, knowing that on typical tabular data AutoML usually **ties** a competently tuned GBDT at far
higher compute (it buys automation/robustness, not a win), and does **not** invent domain features or
escape the single-table paradigm.

## Why this was the ZPD
L024–L028 built the single-table contestants by hand and established the honest bars (tuned GBDT +
tuned ResNet). The obvious skeptic's question is "why not just automate all that?" — so before the Q3
checkpoint we confront AutoML directly. It is the strongest form of "automate the single-table
pipeline," so pinning down *exactly where it stops* (search, not representation; models, not features)
is what makes the relational thesis' remaining claim precise. Also closes the L009 (manual FE) and L017
(HPO) / L018 (ensembling) threads by showing what a machine does and doesn't subsume.

## What shipped
- **Lesson** `lessons/0029-manual-fe-vs-automl.html` (~50 min, standard #17 thoroughness): full
  vocabulary section (AutoML, HPO recall, algorithm selection, CASH + its argmin, Bayesian
  optimization/surrogate/SMAC, meta-learning/meta-features/warm-start, ensemble selection, manual FE) →
  CASH search + viz → meta-learning warm-start → ensemble selection + viz → predict → honest bake-off +
  viz + table → teach-back → thesis bridge → 5 subtleties → 3 quizzes → primary reading → lab.
- **Three reusable viz** (standard #9, one per mechanistic beat), all driven by REAL verify numbers:
  - `assets/cash-search-viz.js` — the real 40-iteration CASH trace (credit_g, seed 0): dots coloured by
    algorithm, a best-so-far step line (0.796 → 0.817), a ★ on the winning HistGB; filter by algorithm.
  - `assets/ensemble-select-viz.js` — Caruana greedy ensemble vs single best; toggle shows the 3-algo
    blend composition (LogReg ×5, HistGB ×4, ExtraTrees ×1) and the +0.007 test gain (0.824 → 0.831).
  - `assets/automl-bakeoff-viz.js` — default XGB / tuned XGB / tiny AutoML with ±sd whiskers; readout:
    tuning is the jump (+0.031), AutoML ties the tuned XGB (−0.002, bands overlap).
  - Headless check `labs/_viz_check_l029.js` — 15/15 pass. **Browser MCP unavailable** (headless env) →
    node verification only, consistent with L021–L028.
- **Lab** `labs/0029-manual-fe-vs-automl.ipynb` — Tier A (real credit_g). Crucial fragment (Task 1) =
  the **greedy ensemble-selection pick** (`cand = (ens_sum + val_probs[j]) / (n_added + 1)`); Task 2 =
  the **CASH selection** (`best_idx = int(np.argmax(val_aucs))`) + single-vs-ensemble on test; Task 3 =
  the bake-off, blank = the tuned-XGB "keep best-validation pipe" (`best_va, best_pipe = va, pipe`).
  Student blank (4 `____`, 0 outputs); solution executed clean & gitignored. Built via `_build_l029.py`;
  random search stands in for SMAC (installable; Bergstra & Bengio 2012 justify it). Rendered to
  `labs/html/0029-*.html`.
- **Verify** `labs/_verify_l029.py` (+ `labs/_dump_l029_trace.py` for the cash-search-viz trace) — ran
  clean.

## Verified live (credit_g, real Tier-A, via _verify_l029.py)
- **CASH search (seed 0, budget 40):** tried all 4 algorithms (logreg 11, histgb 11, extratrees 11,
  rf 7); best-validation AUC climbed 0.796 → 0.817 (peak at iter 15, a HistGB). Single best-val config
  TEST AUC **0.824**.
- **Ensemble selection:** greedy blend of 10 distinct members across 3 algorithms (LogReg ×5, HistGB ×4,
  ExtraTrees ×1) → TEST AUC **0.831**, **+0.007** over the single best (free — models already trained).
- **Honest bake-off (5 seeds), ROC-AUC:** default XGB **0.775 ± 0.025** → tuned XGB **0.806 ± 0.016**
  (**+0.031**, the big jump is tuning at all) ≈ tiny AutoML **0.803 ± 0.020** (**−0.002** vs tuned XGB,
  bands overlap → a tie). AutoML did not beat a competently tuned GBDT; it reached the same ceiling with
  less human effort, at far higher compute.
- **Lab reproduction (3 seeds, budget ~20–30):** consistent — default 0.788 < tuned 0.804 ≈ AutoML
  0.810; greedy ensemble 0.835 vs single best 0.824 (+0.011). All CHECK + EXIT clean.

## Honest framings kept
- **AutoML ≠ automated feature engineering:** it searches models + generic preprocessing, not domain
  features; Featuretools/DFS (L009) are the different, complementary tool.
- **Random search ≠ SMAC:** the demo/lab use random search as an installable stand-in; the mechanism
  taught is CASH + select-by-validation + ensemble, not the exact optimizer.
- **The tie is deflationary and honest:** most of the gain is tuning at all; a 4-algorithm AutoML only
  matches a tuned GBDT. AutoML's value is convenience/robustness, not new accuracy.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l029-cash`, `l029-ties` [misconception]).
- `assets/paper-deck.js` +1 (`feurer2015`).
- `misconceptions.md` **M26** ("AutoML replaces FE and beats a hand-tuned model" → it automates search,
  not features/representation, and ties a tuned GBDT).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1/C4 — an honest single-table baseline can be
  "whatever AutoML finds", but AutoML never touches the representation, and its tie shows single-table
  search is nearly exhausted).
- `reference/glossary.html` +5 Q3/AutoML terms (AutoML, CASH, Bayesian optimization/SMAC,
  meta-learning/warm-start, ensemble selection).
- `lessons/manifest.json` → 29 entries; `labs/html/0029-*.html` rendered.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l029.js` 15/15 clean.

## Env note
`.venv` already present with torch/xgboost/sklearn 1.9.0 from Session 28 — no bootstrap needed this
session (the recurring "preinstall the lab venv incl. CPU torch" note from Sessions 22–28 held up).
Auto-sklearn itself was **not** installed (Linux/version-fragile); the lab reproduces its mechanisms on
sklearn + xgboost instead, which is the honest, portable choice.

## Next
Lesson 030 — **Q3 checkpoint** (Grinsztajn 2022 full; write a 1-page benchmark report). It consolidates
the whole Q3 arc: honest splits (L021), leakage (L022), significance (L023), the benchmark protocol +
three inductive biases (L024–L027), honest neural baselines (L028), and AutoML vs manual (L029) — the
deliverable is a reproducible, skeptic-proof 1-page benchmark report.
