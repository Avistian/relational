# Lesson 010 prepared — Q1 checkpoint: reproducible sklearn baseline

Published `lessons/0010-baseline-checkpoint.html`, reference `reference/baseline-recipe.html`, reusable `assets/checklist.js` (interactive leakage-spine checklist; ticks all → "ready" feedback, reusable for Q2/Q3/Q4 checkpoints), and capstone notebook `labs/0010-baseline-checkpoint.ipynb` (blank fill-in cells, runs end-to-end verified).

**Nature:** this is a **checkpoint/capstone**, not a new concept — it synthesizes Lessons 001–009 into one reproducible, leakage-safe sklearn baseline and evaluates it honestly. The single "skill" is integration: wire the whole spine together unaided.

**Verified live (venv sklearn 1.9.0; n=8000, 15.4% positive, MAR missingness 18%, mixed numeric+categorical, irregular threshold-interaction signal):**
- Dummy(prior): PR-AUC **0.154** (= prevalence floor), ROC **0.500**.
- Logistic (scaled, impute+indicator, OHE, ratio feature): CV PR-AUC **0.381**, test PR **0.389**, ROC **0.733**, Brier **0.114**.
- HistGBDT (native NaN, ratio feature): CV PR-AUC **0.470**, test PR **0.443**, ROC **0.789**, Brier **0.108** — wins on every metric (Grinsztajn 2022: trees win on irregular tabular).
- Calibration of the winner: HistGBDT Brier **0.1078 → 0.1055** (sigmoid), ROC ~unchanged.

**Leakage spine assembled (the checkpoint rubric):** stratified split (L003) + grouped/temporal awareness (L004); everything fit-bearing inside one Pipeline/ColumnTransformer (L005); median impute + missing-indicator (L006); PR-AUC + class handling under imbalance (L007); PR-AUC-vs-prevalence + calibration/Brier (L008); ratio + categorical engineering inside the pipeline (L009); fixed seeds + pinned `.venv` for reproducibility.

**Grounding:** Grinsztajn et al. 2022 (NeurIPS, why trees beat DL on tabular — primary; the reason the GBDT baseline is what the thesis must beat); sklearn Pipeline / ColumnTransformer / common-pitfalls. No new deps (HistGradientBoosting is sklearn-native; XGBoost/LightGBM wait for Q2).

**Thesis bridge:** this baseline *is* the yardstick. Every RDL result later in the thesis is "better/worse than a competently-built, leak-free, calibrated GBDT baseline." Building it reproducibly now means later comparisons are credible. Q2 (011–020) swaps HistGBDT for tuned XGBoost/LightGBM/CatBoost on top of this same harness — first incremental-build candidate (see NOTES reproduction-lab rule).

**Exit:** capstone notebook prints the model comparison table (PR-AUC vs prevalence, ROC, Brier) + calibration delta, and the user ticks the leakage-spine checklist; or "lab done." Lesson 011 (Q2, gradient boosting — XGBoost) unlocks after.
