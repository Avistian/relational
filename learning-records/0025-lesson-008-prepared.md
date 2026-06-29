# Lesson 008 prepared — metrics: ROC vs PR curves & calibration

Published `lessons/0008-metrics-calibration.html`, reference `reference/metrics-calibration.html`, reusable `assets/reliability-viz.js` (reliability-diagram widget: a fixed over/under-confident score set plotted against the y=x diagonal; toggle "raw" vs "calibrated" to see points snap onto the diagonal), and notebook lab `labs/0008-metrics-calibration.ipynb` (blank fill-in cells).

**Single skill:** pick and read the *right* evaluation curve for the question — ROC for ranking, PR for rare-positive operating performance — and separately check/fix **probability calibration** (reliability curve + Brier score; Platt/sigmoid & isotonic via `CalibratedClassifierCV`). Splits the two things people conflate: *ranking quality* (AUROC/AUPRC) vs *probability trustworthiness* (calibration). Builds straight on L007 (PR-AUC default under imbalance).

**Verified live (venv sklearn 1.9.0; one dataset, 17.7% positive, RandomForest):**
- ROC-AUC **0.863** vs PR-AUC **0.720**, PR no-skill baseline = prevalence **0.177** — ROC's floor is 0.5 regardless of imbalance, PR's floor is the positive rate, so ROC always looks rosier.
- Calibration: RF Brier **0.0898 raw → 0.0856 sigmoid → 0.0853 isotonic** (lower better).
- Reliability curve (raw RF) is the classic distortion: predicted 0.55 → actual 0.72, predicted 0.65 → actual 0.88 (under-confident mid/high) — i.e. RF probabilities need calibration before they can be read as probabilities.

**Grounding:** Saito & Rehmsmeier 2015 (PLOS ONE, PR>ROC under imbalance) primary; Davis & Goadrich 2006 (ICML, PR/ROC relationship); Niculescu-Mizil & Caruana 2005 (ICML, calibration + reliability diagrams + Platt/isotonic); sklearn §1.16 Probability calibration + Brier score. No new deps.

**Thesis bridge:** RelBench leaderboards rank on AUROC/MAP/AUPRC; to compare a GBDT baseline against an RDL model *fairly* you must (a) report the curve that matches the task (PR/MAP for rare-positive recommendation & churn), and (b) not confuse a better AUROC with trustworthy risk scores — a deployed churn/fraud score that feeds a downstream threshold or expected-value calc must be *calibrated*. GBDTs and neural nets are both often mis-calibrated; calibration is part of an honest baseline. Reinforces L007 metric choice.

**Exit:** notebook prints ROC-AUC vs PR-AUC with the prevalence baseline, and Brier raw-vs-calibrated + one reliability reading, plus one-sentence cause; or "lab done." Lesson 009 (feature engineering / curated FE patterns, curriculum lec 009) unlocks after; lec 010 is the Q1 reproducible-baseline checkpoint.
