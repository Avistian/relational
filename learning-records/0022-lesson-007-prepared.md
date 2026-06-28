# Lesson 007 prepared — class imbalance

Published `lessons/0007-class-imbalance.html`, reference `reference/class-imbalance.html`, reusable `assets/imbalance-viz.js` (threshold slider over a fixed score distribution → live confusion matrix + accuracy/recall/precision, showing the accuracy paradox and threshold tuning in one widget), and notebook lab `labs/0007-class-imbalance.ipynb` (blank fill-in cells, uses imbalanced-learn).

**Single skill:** handle class imbalance *without leaking and without trusting accuracy* — judge with imbalance-aware metrics (recall, PR-AUC, balanced accuracy), and apply the handling menu (class weights, threshold tuning, resampling) with resampling done **inside the `imblearn` Pipeline** so it only touches training folds. Directly interleaves L005 pipeline-leakage discipline with a new resampler that also rewrites `y`.

**Verified live (venv: sklearn 1.9.0, imbalanced-learn 0.14.2, 5% positive synthetic):**
- Accuracy paradox: `DummyClassifier(most_frequent)` → accuracy **0.945**, recall **0.0**, PR-AUC **0.055**.
- SMOTE leakage: resample-all-then-CV F1 **0.887** vs SMOTE-inside-imblearn-pipeline F1 **0.479** (≈0.41 of fictional F1 — parallels L005's 0.78/0.44 noise demo). Balanced-accuracy version 0.889 vs 0.864 (smaller — F1 exposes it best).
- `class_weight="balanced"`: recall **0.55 → 0.83**, but PR-AUC barely moves (0.756 → 0.734) — reweighting trades precision for recall; it does not improve ranking.
- Threshold tuning: F1@0.5 **0.711** vs best F1 **0.725** at threshold 0.3.
- ROC-AUC **0.914** vs PR-AUC **0.747** on the same model — ROC looks rosier under imbalance.

**Grounding:** He & Garcia 2009 (IEEE TKDE, DOI 10.1109/TKDE.2008.239) primary; imbalanced-learn "Common pitfalls and recommended practices" for the resample-inside-CV rule and the `imblearn.pipeline` fix; SMOTE = Chawla et al. 2002. New dep `imbalanced-learn>=0.12` added to `requirements-labs.txt` (installed 0.14.2).

**Thesis bridge:** the canonical RDL/RelBench tasks (churn, fraud, conversion, link prediction) are heavily imbalanced; a fair GBDT/RDL comparison must use PR-AUC/AUROC not accuracy, and must never resample before the temporal split. `scale_pos_weight` (XGBoost) / `class_weight` is the leak-free default; SMOTE on an entity graph is fraught (synthesizing rows breaks relational structure) — another reason the structural RDL view is cleaner. Reinforces L002/L005/L006 leakage spine.

**Exit:** notebook prints accuracy-vs-recall paradox numbers, the SMOTE leaked-vs-honest F1, and a class_weight recall gain, plus one-sentence cause; or "lab done." Lesson 008 (metrics — ROC vs PR, calibration) unlocks after and was previewed here.
