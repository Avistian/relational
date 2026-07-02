# Lesson 012 prepared — bagging & random forest

Second Q2 unit (ensemble arc). Builds directly on L011: a single deep tree is **high-variance**; bagging averages decorrelated trees to cut that variance. Random Forest = bootstrap aggregating + per-split feature subsampling.

**New reusable asset:** `assets/ensemble-viz.js` — 1-D slider viz: as the number of averaged trees `B` grows, the jagged single-tree step functions collapse toward a smooth ensemble mean (variance reduction made visible). Reusable for any averaging/ensemble lesson.

**Verified live (sklearn 1.9, credit_g, prevalence 0.700):**
- Single deep `DecisionTreeClassifier` CV PR-AUC = **0.757** (barely above prevalence).
- `RandomForestClassifier(n_estimators=300)` CV PR-AUC = **0.901**; OOB accuracy = **0.754**.
- Variance proxy: per-row std of single-tree positive-prob = **0.270**; a 10-tree ensemble = **0.068** (~4× drop).

**Lab (incremental rule active):** `labs/0012-bagging-random-forest.ipynb` imports `relkit` (data + `cv_pr_auc`), compares single tree vs RF, reads OOB, and measures the variance drop. Follows new intro standard (concept recap + per-task goal/why).

**Thesis bridge:** RF is a second baseline, but on tabular the boosted GBDT (Q1 HistGBDT, then XGBoost L014) usually beats RF — bagging cuts variance, boosting cuts bias. Sets up Lesson 013 (boosting intuition).

**Primary reading:** Breiman 2001, *Random Forests* (skim §1–2, OOB).

**Also this session:** web notebook viewing added — `scripts/render_notebooks.sh` (nbconvert → `labs/html/`), `notebooks.html` gallery (View HTML / source / Run on Binder), `binder/requirements.txt` for the runnable path.
