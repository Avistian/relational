# Lesson 007 complete — class imbalance

User completed Lesson 007 lab ("lab done"). Warm-up retrieval before reading was **3/3** (accuracy paradox A; SMOTE-before-CV leak A; missing-indicator-as-signal A — good carry-over from L006).

**Skill secured:** distrust accuracy under imbalance; judge with recall/PR-AUC; handling menu (class weights → threshold → resampling) with resampling **inside `imblearn.pipeline`** so it touches train folds only. Extended the L005/L006 leakage spine to a step that also rewrites `y`.

**Evidence:** warm-up 3/3 in chat; lab self-reported "lab done." (Lab verified-runnable: accuracy paradox 0.945/0.0, SMOTE leak F1 0.887 vs 0.479, class_weight recall 0.55→0.83.)

**Implications:** Lesson 008 (metrics: ROC vs PR curves, calibration) unlocked and prepared. Watch the L006/L007 pattern of naming one effect when several apply — keep nudging multi-factor takeaways.
