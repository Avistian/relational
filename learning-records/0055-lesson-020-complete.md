# Lesson 020 complete — Q2 checkpoint passed

User signalled completion: **"lesson/lab 20 done"** (in the same message that asked to build L021). No
EXIT-ticket text was pasted, so — following the L017/L018 precedent — **no numeric rubric score** is
recorded; the unit is marked done on the user's word.

**What L020 covered.** The Q2 checkpoint (`lessons/0020-q2-checkpoint.html` + `labs/0020-q2-checkpoint.ipynb`):
reproduce a published-style tree baseline under the **fair-comparison contract** (fixed data / split /
metric / tuning budget / preprocessing scope), then match or beat it with a tuned XGBoost / LightGBM and a
leak-free OOF stack, and report the verdict honestly. Verified numbers (adult, ROC-AUC): reference default
XGB 0.9282; tuned XGB 0.9294 / LGBM 0.9296 (tuning barely moves a strong default); stack 0.9297 with OOF
correlation 0.997 (two GBDTs are near-redundant → stacking adds nothing). See
[[0051-lesson-020-published.md]].

**Q2 arc closed.** Lessons 011–020 (decision trees → RF → boosting → XGBoost/LightGBM/CatBoost → tuning →
stacking → when-trees-win → checkpoint) are all complete. The dossier's honest bar is now assembled: beat a
**tuned** GBDT *and* a leak-free stacked ensemble, under honest splits, reporting the gap honestly.

**Next.** Q3 opens with L021 (data splits in the wild) — the temporal-vs-random split, the first pillar of
the Q3 "evaluation rigor" quarter. Published this session; see [[0056-lesson-021-published.md]].
