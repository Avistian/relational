# Lesson 004 started — grouped & nested CV

User began Lesson 004 after completing Lesson 003 (stratified split + CV on train only; warm-up on country/static features and design matrix correct). Prior signal: already uses grouped CV in homework pipelines — do not re-teach that groups matter; focus on `StratifiedGroupKFold`, nested CV (outer mean vs `best_score_`), and `groups` plumbing through both loops.

**Focus this session:** group-leakage viz, Cawley & Talbot selection bias, nested CV two-loop mental model, sklearn metadata-routing pitfall vs manual loop. Exit when lab prints four numbers (KFold mean, GroupKFold mean, grid `best_score_`, nested mean) or user says "lab done."

**Implications:** Lesson 005 (pipelines & preprocessing) unlocks after exit. Optional: audit `report.md` for nested vs reported CV scores during or after lab.
