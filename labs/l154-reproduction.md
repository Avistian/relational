# L154 RelBench portfolio evidence replay

Approved 2026-10-01. This selected experiment reproduces the portfolio report from existing author predictions. It performs no training, dataset download, SQL rerun, graph rebuild, checkpoint inference or paid dispatch. Additional cloud spend USD0. Whole-paper reproduction NOT_RUN.

## Frozen inputs and lanes

`evidence/l154/input-manifest.json` pins 55 consumed files with SHA256. `_freeze_l154.py` creates the manifest once and refuses to overwrite it. Ordinary replay never regenerates the freeze. The captured paper-v1 tables and extracted published values are separately included. These hashes establish current-byte integrity, not historical identity. A notebook embeds the same packet, verifies every hash and decodes only inside a fresh temporary directory.

| Lane | Task | Seeds | Training represented in saved evidence | Evaluation replayed |
|---|---|---|---|---|
| Classification reference | rel-trial/study-outcome | 0–4 | Five complete 20-epoch fits, first strict validation maximum | 960 validation + 825 test rows per seed |
| Classification course-selected | rel-trial/study-outcome | 10–14 | Five complete 20-epoch fits at previously validation-frozen rate | 960 validation + 825 test rows per seed |
| Regression reference | rel-f1/driver-position | 0–4 | Five complete 10-epoch fits, first strict validation minimum | 499 validation + 760 test rows per seed |
| Recommendation pilot | rel-trial/site-sponsor-run | pilot0 | 32 training batches; not a complete fit | 37,003 validation rankings;53,241 catalog; top 10;test NOT_RUN |

Primary task coverage uses the predeclared classification reference, regression reference, and incomplete recommendation entries. Course-selected classification is supplementary and never pooled or chosen because of test performance. Notebook validation runs from prior lessons and search-candidate fits are excluded from the reporting seed sets. The saved validation selection record is checked; L154 does not repeat HPO.15 completed-fit checkpoint selections are checked against their validation histories.

## Executable contract

`relkit/portfolio_l154.py` exposes three learner functions: `summarize_runs`, `compare_entries`, `portfolio_verdict`. They reject missing/duplicate/mixed seeds, incomplete histories supplied by the adapter, changed populations/evaluation contracts and invalid metric scales. Missing test scores remain null, never zero. Summaries use sample SD. Published comparisons cannot produce local-win claims. Coverage counts tasks once and does not equate author evidence with learner mastery.

`_replay_l154.py` is a read-only adapter. It reverses prediction order and realigns full(entity,cutoff)keys before independently computing AUC with tied ranks, MAE and MAP. L151 uses the pinned task table; L152 targets are aligned across seeds and checked against previously audited query-key hashes; L153 uses pinned relevance sets. This is not a fresh source-SQL audit. Scalar target alignment and ranking candidate validity are checked. Actual local test population hashes include query keys, while matching evaluation rules are a separate contract field.

55 frozen files and 61,148 prediction rows are checked. An independent audit uses sklearn AUROC/MAE and vectorized MAP, plus corrupt-byte, missing-key, repeated-key, tied-score, missing-seed, unit/direction and pilot-promotion rejection checks. Publication and notebook tests do not increase benchmark completeness.

## Published context and interpretation

Primary source: https://arxiv.org/html/2407.20060v1#A2.SS1. Source implementation pin inherited from L151–153:9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Published test values are historical context, not local comparator executions. The classification/regression tree is raw-table LightGBM; the displayed recommendation comparator is Past Visit, a ranking heuristic. None is represented as locally executed manual FE.

AUROC and MAP fractions are displayed as percentages, differences as percentage points. MAE uses finishing-position units. No raw cross-metric average or cross-task significance claim is calculated. Relative changes are instructional and do not authorize a portfolio headline average. Training-seed SD is not database-level uncertainty or a paired baseline comparison. Previous course test exposure remains; no pristine confirmatory claim.

Current conclusion: two of three test tasks complete, zero fresh matched baseline tasks; local superiority NOT_ESTABLISHED. The broad curriculum's five-task aspiration and minimum three-completed-task coverage are not satisfied. Source nonfinite-gradient findings, unknown historical bytes and unobserved feature-arrival times persist. Learner PENDING_WRITTEN_DEFENSE.

## Full training and budget boundaries

The original visible full training lanes remain in [L151](l151-reproduction.md), [L152](l152-reproduction.md) and [L153](l153-reproduction.md). L154 does not silently substitute report replay for fresh model reproduction. L153 remains INCOMPLETE with test NOT_RUN. Its frozen pilot projection is USD41.23255 five-run compute before overhead and USD51.85417 safety-adjusted runs/checks, above the earlier USD10 aggregate cap. It is a scenario estimate, not a bill. No L154 cloud allowance is transferred to that experiment. New paid work requires a separately approved experiment; no retries or cloud calls occur here.

## Commands and delivery

```bash
# From relational/; bounded CPU work. Do not rerun the one-time freeze.
.venv/bin/python labs/_check_l154.py
.venv/bin/python labs/_replay_l154.py
.venv/bin/python labs/_audit_l154.py
.venv/bin/python labs/_build_l154.py
.venv/bin/python labs/_execute_l154.py
.venv/bin/python labs/_delivery_l154.py
# After staging intended inputs:
.venv/bin/python labs/_check_pages_checkout.py
```

Portable default solution requires Python 3.10+, numpy and IPython; exact tested versions are recorded in `_execution_l154_results.json`. It runs in an empty directory and generates `l154-portfolio-report.json` and `.md`. Learner edits are passed into the real report path. All datasets are bundled as the frozen evidence packet; no live Colab execution is inferred. Browser, offline notebook, clean Git-index Pages build and deployment statuses are separate. No deployment requested.
