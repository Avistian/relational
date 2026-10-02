# L178 evidence summary

**Compare contracts before scores.** Match the raw database, complete query keys, target window and positive class, permitted task labels, feature/temporal policies and selection procedure. Equal support counts and frozen weights do not establish equal label access.

**Positive-class conversion:** for complemented release labels, use `y_DNF = 1 − y_release` and `p_DNF = 1 − p_release`. Cast saved probabilities to float64 first. Change both or you reverse the interpretation of the score. The real F1 target is 30 days; 60 days changes 82 test labels.

**Tuned GNN:** record every configuration × seed × epoch, choose on validation, then freeze test evaluation. Reject missing/duplicate grid cells. Earliest epoch and declared configuration order break ties.

**Numerical health:** a finite output is insufficient. An affine operation on NaN followed by output masking may leave NaN parameter gradients. The actual local probe has 256. A repaired primitive is not a repaired benchmark model.

**Stop rules:** unmatched information or uncompleted audits block admission. The observed gradient failure stops training. An affordable plan does not override either. READY_FOR_PILOT admits only the next bounded diagnostic, never a scientific conclusion.

**Observed:** published-result replay **COMPLETE**; fresh three-model comparison **INCOMPLETE_TRAINING_HEALTH_GATE**. The local encoder probe has **256 nonfinite gradient entries**. No fresh benchmark model fits or inference and **$0 cloud/API spend**. Whole-paper reproduction and repaired-model training: **NOT_RUN**.

| Saved published arm | Replayed mean AUROC | Sample SD, 10 supports | Paper reference |
|---|---:|---:|---:|
| RDBPFN | 0.721938 | 0.019983 | 0.7219 |
| RDBPFN_single | 0.663990 | 0.034177 | 0.6640 |
| TabICLv1.1 | 0.717568 | 0.009770 | 0.7176 |


The table replays the original RDB-PFN experiment; it is not a RDBLearn-versus-GNN leaderboard. 30 runs, 21,060 predictions, all 702 queries. No new model inference. Historical availability and historical identity remain unestablished.

[Lesson](../lessons/0178-fair-model-comparison.html) · [Student notebook](../labs/0178-fair-model-comparison.ipynb) · [Full report](../labs/evidence/l178/report.json) · [Protocol](../labs/l178-reproduction.md) · [RDBLearn paper](https://arxiv.org/html/2602.18495v1) · [RelGNN paper](https://arxiv.org/html/2502.06784v2) · [RDB-PFN paper](https://arxiv.org/html/2603.03805v5).

