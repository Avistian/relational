# Lesson 005 started — pipelines & preprocessing

User began Lesson 005 after completing Lesson 004 (grouped & nested CV; lab done). Prior signal: strong on grouped leakage; nested-CV selection-bias question was corrected once — **retrieval check passed:** warm-up 3/3 (A: `best_score_` = inner-CV validation mean; B: scaler-on-all-X leaks test stats; C: PIT aggregates = preprocessing RDL learns end-to-end). Homework already uses grouped CV — do not re-teach that groups matter; focus on preprocessing fit scope and Pipeline/ColumnTransformer wiring.

**Focus this session:** preprocessing-leak viz (`pipeline-viz.js`), SelectKBest-before-CV demo (0.78 vs 0.44 on noise), mixed-type `ColumnTransformer`, `step__param` grid syntax, thesis bridge (manual aggregates = preprocessing). Exit when lab prints leaked mean ≥ pipeline mean + one-sentence cause, or user says "lab done."

**Implications:** Lesson 006 (missingness taxonomy MCAR/MAR/MNAR) unlocks after exit. Optional: audit `homework/report.md` for transforms fit outside CV.
