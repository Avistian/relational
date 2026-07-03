# Lesson 014 published — XGBoost

Fifth Q2 unit (curriculum lec 014). First lab where a **new dependency** enters the stack: `xgboost` 3.3.0 added to `.venv` and `requirements-labs.txt`. Chen & Guestrin 2016 §2 (Eq 5–7) is the paper.

**Concept:** XGBoost = L013 stagewise boosting + (1) second-order (gradient *and* hessian) loss approximation, (2) an explicit tree-complexity regularizer Ω = γT + ½λ‖w‖². The two formulas that fall out:
- regularized leaf weight  **w\* = −G/(H+λ)**
- structure-score split gain  **½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ**

**Callbacks wired in:** L011 (gain = children − parent, ΔG sign — now closed by re-deriving it in regularized form); L013 (shrinkage/learning rate); L012 (column/row subsampling from RF); L006 (missingness → sparsity-aware default direction, no imputation); L010/Grinsztajn (the *tuned* baseline the thesis must beat).

**New reusable asset:** `assets/gain-viz.js` — a candidate split with fixed G/H sends stats into L/R leaves; λ slider shrinks leaf weights + scores, γ slider lifts a red toll line and flips the verdict KEEP→PRUNE once γ > raw gain. Headless Node mount+math check clean (browser MCP unavailable this session, as in S14).

**Verified live (sklearn 1.9 + xgboost 3.3.0, `relkit` 5-fold PR-AUC), from executed `solutions/0014-xgboost.ipynb`:**
- **By-hand leaf weight = XGBoost's own leaf output, exactly:** λ=3 → −0.4798 / +3.0947 (matches `trees_to_dataframe`). This is the crucial-fragment payoff — the formula *is* the implementation.
- λ shrinks left-leaf w*: 0→−0.591, 1→−0.548, 3→−0.480, 10→−0.334.
- γ prunes: raw gain 67.58; γ=1.5×raw → net −33.79 → PRUNE.
- **credit_g** (n=1000, prev 0.700): sklearn GBDT 0.879 · XGB default **0.883** (≈ tie) · XGB tuned (40-iter RandomizedSearch) **0.896**. Best params lean on regularization (depth 4, η 0.02, λ 10, subsample 0.7) — as the small/noisy regime predicts. (RF from L012 was 0.901 — still competitive; the honest point.)
- **adult** (stretch, n=48,842): GBDT 0.824 · XGB default 0.829.

**Honest framing kept:** untuned XGBoost barely edges 2016-era sklearn; the win is *tuning*. "Beat a default XGBoost" is explicitly called out as a non-result — the relational thesis must beat a **tuned** booster on structure it sees across joins.

**Lab (incremental rule active):** `labs/0014-xgboost.ipynb` — paper-reproduction 4-block. Crucial fragment = implement `node_score` / `reg_leaf_weight` / `split_gain` and match XGBoost's leaf outputs; Task 2 = λ/γ sweep; Task 3 = default vs tuned on credit_g (reproduction target = lesson table). Solution executed end-to-end (all CHECK + EXIT clean); solution gitignored in `labs/solutions/`.

**Primary reading:** Chen & Guestrin 2016 §2 (Eq 5–7), skim §3 sparsity-aware + §4 systems.

Next: Lesson 015 (LightGBM — Ke et al. 2017; speed vs XGB, histogram + leaf-wise growth).
