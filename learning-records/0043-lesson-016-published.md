# Lesson 016 published — CatBoost

Seventh Q2 unit (curriculum lec 016). Prokhorenkova et al. 2018 (NeurIPS 31, arXiv `1706.09516`). New dependency: `catboost` 1.2.10 added to `.venv` + `requirements-labs.txt`.

**Concept:** CatBoost is the GBDT family member that attacks **leakage**, not speed. Two leaks, one cure ("only use the past" via a random permutation):
- **Ordered target statistics** — encode a categorical for row *i* from only the rows *before* it in a random permutation (its prefix) plus a smoothing prior `a·p`; the greedy whole-column mean leaks row *i*'s own label (worst on rare/high-cardinality categories). Callback to L009 (target-encoding leak), L002 (PIT), L004.
- **Ordered boosting** (§4) — gradient for row *i* comes from a model trained only on earlier rows → removes *prediction shift*. Default `boosting_type="Ordered"` on small data, `"Plain"` (fast) on large.
- **Oblivious (symmetric) trees** — same split per level = a depth-6 decision table (64 leaves); weaker per tree, strongly regularized, branch-free fast inference.

**New reusable asset:** `assets/ordered-viz.js` — permutation table with a "current row" pointer; greedy encoding circles the current row's own y (the leak), ordered encoding shades only the prefix; reshuffle shows ordered TS depends on the permutation (why CatBoost averages several). Headless Node mount check clean (browser MCP unavailable again; only user-arxiv authed). CSS lives in the lesson `<style>` (`.ov-*`).

**Verified live (`_verify_l016.py` + executed `solutions/0016-catboost.ipynb`, `relkit` 5-fold PR-AUC):**
- **TS leakage:** on a *pure-noise* high-cardinality category (K=700, y ⟂ cat), AUC(encoding, y): greedy (own-row incl.) **0.845** = spurious leak; ordered (prefix, mean of 20 perms) **0.493** ≈ 0.5 = clean. First occurrence of each category = prior exactly (CHECK).
- **credit_g** (n=1000, 13 cat cols): XGB (label-enc) 0.883 · LGBM (native) 0.880 · **CatBoost (native) 0.896** — best single-table default on this categorical-rich set.
- **adult** (n≈49k, 8 cat cols): XGB 0.829 · LGBM 0.831 · CatBoost 0.831 — near-tie; leakage diluted when categories are common.
- **Ordered vs Plain boosting** (credit_g, 400 trees): Ordered 0.889 vs Plain 0.896 — a near-wash. Honest framing kept: most of CatBoost's edge here is the *ordered TS* (native categoricals), not ordered boosting; ordered boosting is a robustness device and is slower.

**Honest framing (the L016 myth-buster):** CatBoost holds no secret math — careful manual target encoding *inside a proper CV pipeline* (L009) recovers much of the categorical advantage; oblivious trees can underfit numeric-interaction-heavy data (the adult tie). CatBoost automates the leakage discipline.

**Thesis bridge:** relational DBs are built from high-cardinality categorical foreign keys (`user_id`, `product_id`) — the entities. Ordered TS is the leakage-free single-table way to encode identity, a direct baseline RDL must beat, and the same "use only the past" rule RDL enforces with timestamps on the entity graph (Y3–Y4).

**Lab:** `labs/0016-catboost.ipynb` — paper-repro 4-block. Crucial fragment = implement `ordered_ts` by hand (encode-before-update) and show greedy leaks / ordered is clean on pure noise; Task 2 = CatBoost native vs XGB label-enc vs LGBM native on credit_g (reproduction target = lesson table). Manual clone-free CV helper PROVIDED (CatBoost's `cat_features` breaks `sklearn.clone`, so `relkit.cv_pr_auc`/`cross_val_score` can't be used directly — documented in `_verify_l016.py`). Student blank (3 TODO cells, no outputs); solution executed clean (all CHECK + EXIT) and gitignored. Committed `_verify_l016.py` + `_build_l016.py`. Manifest regenerated (16 entries); all labs re-rendered to `labs/html/`.

Next: Lesson 017 (Hyperparameter search — Bergstra & Bengio 2012; random vs grid search; then L018 ensembling/stacking, L019 when trees win, L020 = Q2 checkpoint).
