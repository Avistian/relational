# L181 reproduction contract

Approved: RelBench v2 v1 Tables5/14/15, results-position and qualifying-position, five deterministic baselines and five fresh GNN seeds per task. Archive code at publication-date commit `0d47fe0c8a1a51aaf97ab485f4a028e290f97c67`; this is not asserted to be the experiment commit. Original model, trainer, task construction and feature transformations are visible under sources/l181/upstream and in the notebook appendix.

Observed: COMPLETE_SELECTED_BASELINE_REPRODUCTION; 25,010 labels and 68,925 validation/test predictions independently checked. All40 MAE/R² comparisons CLOSE_ROUNDED within .001. GNN NOT_RUN_TRAINING_HEALTH_GATE; overall selected experiment INCOMPLETE. Whole benchmark, RelGT-AC numerical reproduction and historical training identity remain unestablished/unrun. No paper GNN metric is relabeled a local result.

## Frozen protocol

- Complete nine-table F1 database archive SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`, matching pinned v2 registry. All extracted tables authenticated against L171 input hashes.
- Source timestamp intervals `(dbmin,val-1s]`, `(val,test-1s]`, `(test,dbmax]`; val2005-01-01/test2010-01-01. Missing targets removed; retain complete `(primary key,time)` identities. Bounded interval rewrite replaces only the memory-intensive date_range with proven identical extrema and independent SQL predicates; no shorter population.
- Results task global target removal plus statusId,positionOrder,points,laps,milliseconds,fastestLap,rank. Qualifying task global target removal alone. Label source remains raw target; target table identities differ from driver identity.
- Baseline validation fitted on train; test refitted on train+validation, exactly as released. Unseen entity keys fall back to zero. Baselines deterministic, no artificial seed SD.
- GNN intended: source Model/HeteroEncoder/GraphSAGE, ResNet row encoder four layers128; two sum-aggregating graph layers128, fanouts128/64, batch512, 10epochs, AdamLR.005, L1Loss, validationMAE selection, last tie retained, clipping train label percentiles2/98. Seeds0–4 course-declared, original identities unresolved. Planned full10fits; zero executed.
- Descriptive tolerances fixed before measurement: baseline .001MAE/.001R²; GNN .2MAE/.03R² mean. Tolerances are proximity criteria, not statistical equivalence or historical identity.

## Stop evidence and remaining gaps

Pinned numerical encoder evaluated on every pre2005 numerical row for each table (untimed rows retained). Finite forward output but bad gradients in circuits/results; current CPU Torch2.13.0+cpu and Frame0.3.0. Independent LinearEncoder circuits test confirms128nonfinite gradient elements. Early input-imputation controls finite; no repaired full model fitted. This is not a full sampled GNN update and does not prove authors' historical CUDA behavior. Full GNN operator after admission NOT_VALIDATED.

Original graph materialization includes all dates in feature statistics. Full preprocessing horizon and per-query same-day availability remain unresolved. Complete sampler audit NOT_RUN after prerequisite stop. Authentic data identity is not complete pipeline legality.

RelGT-AC: paper-described architecture only. Bounded source search found no authenticated model release. Seed-only versus global masking, source/claim consistency, historical dependency identity and validation-versus-test interpretation remain explicit gaps. No trained surrogate replaces it.

## Budget and execution

USD10 aggregate cap, planned stop8 with2 reserved, including setup, pilots, retries and verification. One L4 plus **two physical CPU cores** and16GiB costs .000222+2*.0000131+16*.00000222 = .00028372USD/s (~1.021392USD/h), checked2026-10-02. This is a price formula, not a runtime estimate. No cost pilot admitted after scientific failure; actual cloud/API spend0. Local numerical ceiling3600aggregate seconds with all attempts logged in evidence/l181/local-budget.json. Failed preparation attempts are retained.

```bash
.venv/bin/python labs/_budget_l181.py .venv/bin/python labs/_audit_l181.py
.venv/bin/python labs/_budget_l181.py .venv/bin/python labs/_verify_l181.py
.venv/bin/python labs/_budget_l181.py .venv/bin/python labs/_run_l181.py
# Last command exits2; validates evidence and blocks dispatch.
.venv/bin/modal run modal/l181_paper_repro.py
# Local admission only: no remote function; same expected block.
```

The notebook's optional RUN_PAPER_REPRO cell is OFF. It never dispatches a paid run; it reports the observed prerequisite failure from the authenticated packet. Reopening training requires explicit source/environment resolution and a new named protocol if repaired. No shortened training, hidden imputation or budget expansion is permitted.

## Evidence boundaries

Source/data audit and full baseline reproduction: COMPLETE. Fresh GNN training: NOT_RUN. Whole approved model experiment: INCOMPLETE. Historical experiment identity: NOT_ESTABLISHED. Learner: PENDING_WRITTEN_DEFENSE. L180 practical exit stays INCOMPLETE. Browser/notebook/Pages checks have separate receipts. Live Colab and deployment NOT_CHECKED; no push requested.
