# Lesson 004 complete — grouped & nested CV

User completed Lesson 004 (lab done) plus warm-up retrieval. Warm-up: grouped leakage identified correctly (same entity in train+test, answer A); on the nested-CV selection-bias question chose D (`best_score_` is train-set) — corrected to B (same folds chose *and* scored → selection bias). The `best_score_`-is-validation-not-train misconception is now addressed; watch it doesn't resurface.

**Evidence:** warm-up answers in chat; lab self-reported "lab done."

**Implications:** Lesson 005 (pipelines & preprocessing) unlocked. Reinforce that nested-CV outer mean is the honest number and `best_score_` is a *validation* score (not training) chosen by the same data. Pipelines lesson should connect preprocessing-leakage to the CV discipline just built.
