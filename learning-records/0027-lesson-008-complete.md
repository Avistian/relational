# Lesson 008 complete — metrics: ROC vs PR & calibration

User completed Lesson 008 lab ("lab done"). Warm-up retrieval before reading was **3/3** (PR baseline = prevalence A; high-AUROC + off-diagonal reliability = good ranker/bad probabilities A; `CalibratedClassifierCV(cv=...)` fits on held-out folds A).

**Skill secured:** separate *ranking quality* (ROC-AUC vs PR-AUC, read against the prevalence baseline) from *probability trustworthiness* (reliability curve + Brier), and fix the latter with `CalibratedClassifierCV` (sigmoid/isotonic) without leaking. Extends the L005–L007 leakage spine to the calibration step.

**Evidence:** warm-up 3/3 in chat; lab self-reported "lab done." During the lab the user hit two real `CalibratedClassifierCV` snags and worked through them with guidance: (1) `p_cal = cal.fit(...)` returns the estimator not probabilities → corrected to `cal.predict_proba(Xte)[:, 1]`; (2) the calibrator needs a base estimator passed in. Both are exactly the misconceptions the lab is meant to surface — good sign the feedback loop worked.

**Implications:** Lesson 009 (feature engineering) prepared and published. The `fit()`-returns-self confusion is worth a quick callback if it recurs. Continue nudging multi-factor takeaways (L006/L007/L008 pattern).
