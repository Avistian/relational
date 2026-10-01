# L170 complete saved-evidence replay

| Context K | F1 gain ± SD | Positive draws | Trial gain ± SD | Positive draws |
|---|---:|---:|---:|---:|
| 64 | -0.02098 ± 0.03744 | 2/10 | +0.01707 ± 0.02199 | 9/10 |
| 128 | -0.00863 ± 0.05041 | 3/10 | -0.00612 ± 0.02829 | 4/10 |
| 256 | +0.00343 ± 0.01228 | 6/10 | +0.01752 ± 0.03414 | 6/10 |
| 512 | +0.00437 ± 0.01981 | 6/10 | +0.00574 ± 0.01765 | 6/10 |
| 1024 | +0.00142 ± 0.00915 | 6/10 | +0.00719 ± 0.02538 | 6/10 |

Gain = RDB-PFN minus TabICL AUROC. SD describes support-draw variation within each fixed task. All numbers are authenticated author-reference evidence, not learner results.

All 300 evaluations / 229,050 predictions are reused in L170. Exact TabICL repeatability FAIL; Griffin INCOMPLETE_BUDGET_GATE; RDBLearn pipeline and fresh pretraining NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.
