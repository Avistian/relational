# RelBench portfolio synthesis

Named experiment: **L154 RelBench portfolio evidence replay**. This is saved-prediction replay; no new fits.

| Task / metric | Local test mean ± sample SD | Completed seeds | Published comparator (context only) | Fresh FE / tree |
|---|---:|---:|---|---|
| rel-trial/study-outcome / AUROC | 69.2588 ± 0.7489% | 5/5 | Raw-table LightGBM: 70.090% | NOT_RUN / NOT_RUN |
| rel-f1/driver-position / MAE | 3.9708 ± 0.1626 positions | 5/5 | Raw-table LightGBM: 4.170 positions | NOT_RUN / NOT_RUN |
| rel-trial/site-sponsor-run / MAP@10 | NOT_RUN — INCOMPLETE | 0/5 | Past Visit (ranking heuristic): 17.310% | NOT_RUN / NOT_RUN |

**Coverage: 2/3 tasks; 0 fresh matched comparator tasks.** Local superiority: **NOT_ESTABLISHED**.

The recommendation pilot evaluated validation only: 1.050815% MAP@10 on 37,003 queries. This is not a third completed test result.

## Within-task descriptive context
- rel-trial/study-outcome: oriented benefit relative to the published baseline -0.8312 percentage points. PUBLISHED_CONTEXT_ONLY; local win NOT_ESTABLISHED.
- rel-f1/driver-position: oriented benefit relative to the published baseline +0.1992 positions. PUBLISHED_CONTEXT_ONLY; local win NOT_ESTABLISHED.

Separate course-selected classification lane: 68.4570 ± 1.3502% AUROC, seeds10–14. Never pooled with reference seeds0–4.

## Evidence and limits
Independently rescored 61,148 prediction rows; checked 15 validation-selected checkpoints and 55 frozen input files.

No arithmetic mean across AUROC, MAE and MAP; no win rate with missing matched baselines. Sample SD describes training-seed variability on the same fixed test population; it is not uncertainty across databases.

L151 pinned task table; L152 targets from seed0 packet previously checked against archive SQL; L153 pinned relevance sets. No new source-SQL execution.

Reference classification selected by design, not test score; course-selected lane remains separate. No retuning.

Source nonfinite-gradient observations in L151-L153 persist; rescoring cannot certify optimization health.

Full selected recommendation reproduction INCOMPLETE. Whole paper / fresh FE / fresh trees NOT_RUN. Historical identity and feature-arrival legality NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE. Live Colab / deployment NOT_CHECKED.

Additional cloud spend: USD0. Replay does not count historical training costs again.

## Sources
[RelBench v1 Tables 6–8](https://arxiv.org/html/2407.20060v1#A2.SS1); upstream reproduction ledgers: [L151](../../l151-reproduction.md), [L152](../../l152-reproduction.md), [L153](../../l153-reproduction.md).

## Learner defense
PENDING_WRITTEN_DEFENSE — explain coverage, baseline identity, a within-task difference, seed uncertainty and the missing experiment before claiming completion.
