# Lesson 015 published — LightGBM

Sixth Q2 unit (curriculum lec 015). Ke et al. 2017 (NeurIPS). New dependency: `lightgbm` 4.6.0 added to `.venv` + `requirements-labs.txt`.

**Concept:** LightGBM = XGBoost's L014 split-gain math, re-engineered for speed/scale via four ideas:
- **Histograms** (~255 bins; child = parent − sibling histogram, the subtraction trick — a callback to L014's candidate-threshold preview).
- **Leaf-wise (best-first) growth** — split the frontier leaf with max delta loss. Lower loss per #leaves than level-wise, but overfits small data → **`num_leaves` is the primary complexity knob** (not `max_depth`). Real shift from XGBoost.
- **GOSS** — keep top `a·n` rows by |gradient|, sample `b·n` from the small-gradient rest, amplify by `(1−a)/b` to keep sums unbiased (Algorithm 2).
- **EFB** — bundle mutually-exclusive sparse features; NP-hard, greedy approx.

**New reusable asset:** `assets/growth-viz.js` — leaf-wise vs level-wise growth toggle + split-budget slider; readout always shows both remaining-loss values so leaf-wise ≤ level-wise is explicit. Headless Node mount check clean (browser MCP unavailable again; only user-arxiv authed).

**Verified live (sklearn 1.9 + lightgbm 4.6.0 + xgboost 3.3.0, `relkit` 5-fold PR-AUC), from `_verify_l015.py` / executed `solutions/0015-lightgbm.ipynb`:**
- **GOSS unbiased:** uses ~30% of rows (a+b=0.2+0.1), amplify 8×; mean estimate error over 200 seeds ≈ +0.06 vs true left-sum −33.2. (First implementation sampled `b·|rest|` and was biased by (1−a); fixed to `b·n` per the paper — this bug was caught by the notebook CHECK.)
- **credit_g** (n=1000): GBDT 0.879 · XGB default 0.883 · **LGBM default 0.889** · LGBM tuned 0.893 (search picked `num_leaves=7`, η=0.02, 600 trees).
- **num_leaves sweep** (credit_g): 7→0.889, 15→0.887, 31→0.888, 63→0.884, 127→0.884 — bigger trees overfit, as leaf-wise theory predicts.
- **adult** (n≈49k): XGB 0.829 vs LGBM 0.831 — accuracy near-tie.
- **Honest speed** (50k×50, 100 trees): sklearn GBDT 64.3s (1×) · XGBoost-`hist` 0.58s (≈110×) · LightGBM-`gbdt` 0.60s (≈107×) · LightGBM-`goss` 0.42s (≈152×).

**Honest framing (the L015 myth-buster):** the "20× faster" is vs *conventional pre-histogram* GBDT; modern XGBoost-`hist` adopted the same trick and is on par — the same shape as L014's "untuned XGB ≈ 2016 sklearn." GOSS is **opt-in** (`boosting_type='goss'`, default is `'gbdt'`). LightGBM's durable edges: native categoricals, low memory, sparse/high-cardinality scaling (EFB).

**Currency fix:** CURRICULUM verified index had LightGBM as arXiv `1711.08251` — that ID is an unrelated hep-ph paper. LightGBM has **no arXiv**; corrected to cite the NeurIPS proceedings in CURRICULUM.md + RESOURCES.md.

**Lab:** `labs/0015-lightgbm.ipynb` — paper-repro 4-block. Crucial fragment = implement `goss_weights` and show unbiasedness; Task 2 = num_leaves sweep (see overfitting); Task 3 = LGBM vs XGB on credit_g (reproduction target = lesson table); stretch = hist boosters vs conventional GBDT speed. Student blank (3 TODO cells, no outputs); solution executed clean (all CHECK + EXIT) and gitignored. Committed harness `_verify_l015.py` + builder `_build_l015.py`. Manifest regenerated (15 entries); all labs re-rendered to `labs/html/`.

Next: Lesson 016 (CatBoost — Prokhorenkova et al. 2018; ordered boosting + native categorical handling; target-leakage-free encoding).
