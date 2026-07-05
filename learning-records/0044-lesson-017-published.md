# Lesson 017 published — Hyperparameter search (random vs grid)

Eighth Q2 unit (curriculum lec 017). Bergstra & Bengio 2012, *Random Search for Hyper-Parameter Optimization* (JMLR 13:281–305 — **no arXiv**; JMLR PDF). No new dependency: scipy/xgboost already in `.venv`.

**Concept / single skill:** explain why random search beats grid **at an equal budget when few knobs matter** (low effective dimensionality), run `RandomizedSearchCV` correctly, and avoid the optimism trap. Three pillars:
- **Grid's curse** — cost is `k^d` (exponential in #knobs); a grid of `n` points is a `√n×√n` lattice, so it tries only **√n distinct values on each axis**. Random draws `n` independent points → **≈n distinct values on every axis**.
- **Log-uniform sampling** for rates spanning orders of magnitude (`loguniform(0.01, 0.3)`), `randint` for integer knobs, `uniform(0.6, 0.4)` for fractions. Budget = `n_iter` (you control cost directly).
- **Optimism / selection bias (L004 callback)** — `best_score_` is the max over searched candidates → biased. Report **nested CV** or held-out test.

**New reusable asset:** `assets/search-viz.js` — the iconic B&B Fig 1. SVG 2-D box, peaked green marginal on the important axis (top), near-flat orange marginal on the useless axis (left); Grid/Random toggle + budget selector (9/16/25) + Resample; dashed projections drop each trial onto the important axis; readout shows distinct-important-axis values (grid √n vs random n) and best peak value found. CSS `.sv-*` in the lesson `<style>`. Headless Node mount check clean (4 top-level children; math: grid distinct = round(√budget), random = budget). **Browser MCP unavailable again** (empty tools folder; only user-arxiv authed) → headless verification only.

**Verified live (`_verify_l017.py` + executed `solutions/0017-hyperparameter-search.ipynb`):**
- **Synthetic mechanism** (1 important dim, sharp peak; optimum = 1.000; random mean/200 seeds): the **honest crossover** — in low dim a lucky grid *wins*: 2-D budget 9 grid **1.019** vs random 0.771; budget 25 grid 1.019 vs random 0.947. As knobs grow the budget forces grid to **2 values/axis** and it collapses: 5-D budget 32 grid **0.076** vs random **0.967**; 8-D budget 64 grid 0.134 vs random 1.012. Fixed-budget(27) sweep shows the crossover at d≈3–4.
- **credit_g tuned XGBoost** (relkit 5-fold PR-AUC, equal budget 27): default **0.883** · GridSearchCV (3×3×3) **0.891** · RandomizedSearchCV (5 knobs, log-uniform rate, 27 draws) **0.901** — matches RF 0.901 (L012) and clears tuned-XGB 0.896 (L014).
- **Nested CV honesty:** `best_score_` 0.901 (optimistic) vs nested CV **0.895** (honest); gap +0.006 (small but always ≥0, grows with #candidates).

**Honest framing (the L017 myth-buster):** random search is **not** a universal law — a well-placed grid can beat it in 1–2 dims (our 2-D rows). The advantage is about **many knobs, most useless**; it appears and dominates as dimensionality grows. Random search is dimension-agnostic (keeps ~n distinct tries per axis regardless of d).

**Thesis bridge:** the undervaluation bet lives or dies on **fair baselines** — every model (GBDT baseline *and* relational model) gets the **same tuning budget** under the **same CV protocol**, reporting a nested/held-out number. Random search is the equalizer (identical budget regardless of knob count). Directly sets up Y1 Q3 benchmark-literacy and the Y3 lec-135 "fair tuning budget on the REG."

**Lab:** `labs/0017-hyperparameter-search.ipynb` — paper-repro structure, 4 TODO cells + stretch. Task 1 crucial fragment = `random_search` by hand + reproduce the crossover; Task 2 = GridSearchCV vs RandomizedSearchCV on XGB/credit_g at equal budget (student writes `dists` w/ loguniform + `n_iter=grid_n`); Task 3 = nested CV honest estimate (L004 callback). Stretch = `HalvingRandomSearchCV`. Student blank (0 outputs); solution executed clean (all CHECK + EXIT) and gitignored. numpy-2.5 fix: `float(response(...)[0])` (can't `float()` a 1-elem array). Committed `_verify_l017.py` + `_build_l017.py`. Manifest regenerated (17 entries); all labs re-rendered to `labs/html/`.

Next: Lesson 018 (Ensembling & stacking — Wolpert 1992, stacked generalization; simple blend), then L019 (when trees win — Grinsztajn preview), L020 = Q2 checkpoint.
