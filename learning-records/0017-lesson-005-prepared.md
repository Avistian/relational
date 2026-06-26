# Lesson 005 prepared — pipelines & preprocessing

Published `lessons/0005-pipelines-preprocessing.html`, reference `reference/pipelines-preprocessing.html`, and reusable `assets/pipeline-viz.js` (preprocessing-leak grid: "fit on all data" rings every test cell vs "Pipeline" rings none). Single skill: wrap every transform with a `fit` (impute/scale/encode/select) inside a `Pipeline`/`ColumnTransformer` so it refits per fold on train only. Builds straight on L004's leakage-discipline; tuning via `step__param` composes with nested CV.

**Verified against sklearn 1.9 (ran live):** SelectKBest-before-CV on pure noise inflates accuracy to 0.78 vs 0.44 honest inside pipeline (28-pt fiction). `OneHotEncoder` with `set_output(transform="pandas")` requires `sparse_output=False` (otherwise "Pandas output does not support sparse data"). ColumnTransformer + `make_column_selector` + pandas output yields `num__a` / `cat__cat_x` names. StandardScaler leak is near-zero on stable data — used as a teaching point that the rule must be unconditional.

**Thesis bridge:** GBDT baseline numbers are only fair if preprocessing refits per fold; and hand-built PIT aggregates (L002) *are* preprocessing — RDL's pitch is to learn them end-to-end, so flawless manual pipelines are the baseline the thesis must beat.

**Exit:** lab prints leaked mean ≥ pipeline mean + one-sentence cause, or "lab done." Lesson 006 (missingness taxonomy MCAR/MAR/MNAR) unlocks after.
