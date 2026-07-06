# Lesson 018 published — Ensembling & stacking

Ninth Q2 unit (curriculum lec 018). Wolpert 1992, *Stacked Generalization*, Neural Networks 5(2):241–259
(**no arXiv**; ScienceDirect). No new dependency — sklearn/xgboost/lightgbm/catboost already in `.venv`.

**Concept / single skill:** build a stacked ensemble whose meta-learner is trained on **out-of-fold (OOF)**
base predictions; know *when* combining beats the best single model (**diversity** — uncorrelated errors
cancel); and never leak (in-sample base predictions crown the memorizer). Three pillars:
- **Three combiners** — averaging/blending (`VotingClassifier`), weighted blend, and **stacking** (a level-1
  meta-learner over level-0 base predictions, Wolpert 1992).
- **Diversity is the fuel** — measured via the OOF-prediction correlation matrix: the 3 GBDTs correlate ≈0.89
  (CatBoost↔RF 0.95, nearly redundant); the logistic model is the odd one out at ≈0.68. An ensemble of 3 GBDTs
  is "one model in a trench coat."
- **The OOF mechanism + the leak** — `cross_val_predict` gives each row a prediction from a model that never
  saw it. Skip the folds → the meta-learner over-trusts whichever base overfits (L004/L005 leakage, one level up).

**New reusable asset:** `assets/stacking-viz.js` — level-0/level-1 SVG: 12 rows × 4 folds, a **META-FEATURE**
column, and a mode toggle. *Out-of-fold*: "next fold ▶" holds out each fold in turn and fills its rows green
(honest) — after 4 folds all 12 are honest. *In-sample (leak)*: every cell fills red at once (memorized copy of
the label). CSS `.stk-*` in the lesson `<style>`. Headless Node mount check clean (`labs/_viz_check_l018.js`:
6/6 — 3/12 at fold 0, 6/12 after one step, 12/12 honest, 12 red on leak, reset to 3/12). **Browser MCP
unavailable again** (empty tools folder; only user-arxiv authed) → headless verification only.

**Verified live (`labs/_verify_l018.py` + executed `solutions/0018-ensembling-stacking.ipynb`, relkit 5-fold
PR-AUC, credit_g label-encoded):**
- **Base learners:** logistic 0.874 · XGBoost 0.883 · LightGBM 0.889 · CatBoost 0.900 · **RandomForest 0.901
  (best single)**.
- **Diversity:** GBDT↔GBDT corr ≈0.89; logistic↔GBDT ≈0.68.
- **Blend (OOF average):** all-5 **0.899**, 3-GBDT-only 0.895 — the naïve blend actually *trails* the best
  single (drags 4 models toward the weaker logistic). Honest: on 1000 rows ensembling barely moves.
- **Stacking (logistic meta, `StackingClassifier` cv=5):** diverse-5 **0.902** (edges best single by +0.001),
  3-GBDT-only 0.899.
- **Leakage trap (70/30 split + a 1-NN memorizer):** 1-NN in-sample PR-AUC 1.000, OOF 0.704. Naïve meta gives
  1-NN the **largest** weight (coef **+3.00**), train PR-AUC **1.000** (mirage) → test **0.895**. OOF meta gives
  1-NN **−0.11**, train 0.885 → test **0.930**. **Held-out gap +0.035** purely from not leaking.

**Honest framing (the L018 myth-buster):** ensembling is **not** free lunch on a tiny single table — the stack
edges the best single model by a hair (+0.001) and the naïve blend loses. The durable lessons are the
**mechanism (OOF), the diversity requirement, and the leak**, not the headline number. Ensembling pays off with
(a) diverse base families and (b) enough data for stable meta-weights — the leaderboard regime.

**Thesis bridge:** the real single-table baseline is a **leak-free stacked ensemble** of tuned models
(TabArena / Kaggle), not a single default. "Beats XGBoost" is the wrong bar; "beats a tuned, stacked ensemble
under the same CV protocol" is the bar. Ensembling is where the single-table paradigm is *strongest* — so a
relational model that clears it is a legible win. Continuity with L017: score the whole stack on data its
fitting never saw. Sets up Y1 Q3 benchmark-literacy and Y2 Q2 lec-057 "ensembling across model families."

**Lab:** `labs/0018-ensembling-stacking.ipynb` — paper-repro structure, 3 TODO + stretch. Task 1 crucial
fragment = OOF meta-features by hand (`cross_val_predict`) + blend; Task 2 = the leak contrast (student builds
`oof_tr`, compares naïve vs OOF meta-learners with the 1-NN memorizer); Task 3 = `StackingClassifier`
(final_estimator + cv) reproduction (0.902). Student blank (4 `____` cells, 0 outputs); solution executed clean
(all CHECK + EXIT) and gitignored. **Setup-cell fix:** solutions live in `labs/solutions/` so the path insert
now adds both `Path(".")` and its parent → `relkit` imports whether run from `labs/` or `labs/solutions/`.
Committed `_verify_l018.py` + `_build_l018.py` + `_viz_check_l018.js`. Manifest → 18 entries; all labs
re-rendered to `labs/html/`.

Next: Lesson 019 (When trees win — Grinsztajn 2022 preview), then L020 = Q2 checkpoint.
