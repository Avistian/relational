# Lesson 014 complete — XGBoost

User: "lesson 14 done." Fifth Q2 unit closed; the regularized-booster machinery is now explicit.

**Lab score ≈ 9/10** (lab-authoring rubric, 0–2 × 5):
- Correctness 2 — all CHECK pass; by-hand leaf weight reproduces XGBoost's own leaf output exactly (λ=3 → −0.480/+3.095).
- Leakage discipline 2 — scoring via `relkit.cv_pr_auc` (StratifiedKFold); tuning via `RandomizedSearchCV` on CV folds only.
- Conceptual takeaway 2 — the two-formula story landed: `w* = −G/(H+λ)` and the structure-score gain `½[L+R−parent]−γ`.
- Mid-zone effort 2 — implemented `node_score` / `reg_leaf_weight` / `split_gain` (crucial fragment) and the λ/γ sweep.
- Reproduction 1 — credit_g table matched (GBDT 0.879 · XGB default 0.883 · tuned 0.896); slight variance expected across environments.

**Retrieval demonstrated in the follow-up Q&A (this session, before the "done" call):**
- g/h are derivatives of the loss w.r.t. the current score F, recomputed every round and discarded at inference. (Squared error: g = F−y, h = 1. Logistic: g = p−y, h = p(1−p).)
- Thresholds are chosen from a hessian-weighted histogram (~256 candidate cuts), not every unique value.
- A split with gain ≤ 0 after the γ toll is **pruned**; η enters only the **outer** update F ← F + η·f_t, never the gain or leaf-weight formulas.
- At inference: walk each tree, accumulate η·w_leaf onto the base score, then sigmoid for classification.

**Retained skills / open threads:**
- L011 ΔG (children − parent) callback is now fully **closed** — re-derived in regularized form in L014.
- Honest-baseline discipline reinforced: untuned XGB ≈ 2016-era sklearn; the win is *tuning*. The relational thesis must beat a **tuned** booster.
- Histogram split finding was previewed in L014 → becomes the backbone of L015 (LightGBM).

Next: Lesson 015 (LightGBM — leaf-wise growth, GOSS, EFB; speed vs XGBoost).
