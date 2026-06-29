# Lesson 007 started — class imbalance

User began Lesson 007 after completing Lesson 006 (missingness taxonomy; EXIT TICKET submitted). Prior signal: L006 takeaway named MNAR/indicator but omitted MAR bias — watch for naming only one mechanism when several apply. **Retrieval check passed:** warm-up 3/3 (A: accuracy paradox = majority-class predictor; A: SMOTE-before-CV leaks synthetic points into validation; A: indicator preserves was-missing signal under MNAR). L005/L006 leakage spine solid; this lesson extends it to resampling via `imblearn.pipeline`.

**Exit:** notebook `labs/0007-class-imbalance.ipynb` prints paradox numbers, leaked-vs-honest SMOTE F1, class_weight recall gain + one-sentence cause; or "lab done." Lesson 008 (metrics: ROC vs PR, calibration) unlocks after.
