# Lesson 008 started — metrics: ROC vs PR & calibration

User began Lesson 008 after completing Lesson 007 (class imbalance; lab done). Prior signal: L006/L007 takeaway pattern of naming one effect when several apply — watch for same on calibration vs ranking. **Retrieval check passed:** warm-up 3/3 (A: PR baseline = prevalence; A: high AUROC + off-diagonal reliability = good ranker, bad probabilities; A: `CalibratedClassifierCV(cv=...)` fits calibrator on held-out folds). L007 PR-AUC default retained.

**Exit:** notebook `labs/0008-metrics-calibration.ipynb` prints ROC-AUC vs PR-AUC (with prevalence), Brier raw-vs-calibrated + AUROC unchanged, plus one-sentence cause; or "lab done." Lesson 009 (feature engineering, curriculum lec 009) unlocks after; lec 010 = Q1 reproducible-baseline checkpoint.
