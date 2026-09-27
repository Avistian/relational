# Lesson 137 — full selected reproductions and REG error analysis

## Claim boundaries

Fresh selected training: RelBench arXiv2407.20060v1 Table7, basic RDL, rel-f1/driver-position. Seeds0–4, ten full epochs,7453 train queries per epoch,499 validation,760 test. Validation/test targets3.193/4.022 MAE; inherited predeclared descriptive tolerance±.2. Fresh means3.192021/4.089695, sample SD.045983/.143919: CLOSE/CLOSE. This is not statistical equivalence or historical run identity.

FE: complete released user-study SQL and LightGBM pipeline replay, five ten-trial searches plus selected refits. Fresh validation/test means2.777330/3.948917. Original Figure3 task scalar/search randomness unavailable; historical score identity NOT_ESTABLISHED. Original human feature-ideation effort is not reproduced. Other tasks and whole-paper training NOT_RUN.

Error analysis is an original course extension: all12590 held-out model predictions form6295 paired query/run rows. Positive paired absolute-loss difference means GNN loses. The primary target precision is the archive's float64 for both methods; inherited RDL trainer casts some internal labels to float32. Checks verify the small target difference and use aligned archive keys.

## Provenance and deviations

GNN source9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; visible Model in relkit/rdl_l117.py, trainer in relkit/tuning_train_l135.py. Two128-channel sum-aggregation GraphSAGE layers, typed Frame encoders, relative-time encoding, Adam.005, fanouts[128,64],batch512,L1 loss. First minimum-validation checkpoint; evaluation clips to training-label2nd/98th percentiles. Original-model forward comparisons use identical batches and selected weights. The paper lists128 neighbors while the released code divides by hop; preserve[128,64].

FE source445bb7a3b1230f49f8e5890ae81754d3e365680f; PyTorch Frame0.2.2 commit56f687ddf4bf1c4a7d7b72ab0ef4117493256199. Original SQL/schema under sources/l129; canonical visible pipeline relkit/fe_experiment_l129.py. Full50 engineered features plus numeric driverId; timestamp ignored. Train-fitted mappings; LightGBM4.3.0 L1 objective, ten seeded TPE trials/run,2000 rounds,50-round early stopping, train-only selected refit. Freeze SQL rows to archive order since released SQL omits ORDER BY. Seeded TPE and4threads are declared adapters. Original-source pilot checks exact predictions.

Both pipelines use the released database capped through2010-01-01. GNN preprocessing statistics use this full snapshot; FE mappings fit training rows only. GNN historical sampling is inclusive timestamp≤query cutoff. FE standing/results use strict past; schedule slots may refer to future scheduled dates with unknown publication times. Released final static attributes lack mutation/arrival histories. Actual ingestion legality is NOT_ESTABLISHED. Matching inputs at the archive level does not make this an architecture-only controlled experiment. Original FE staging endpoint returned404; v1 archives replace it, with disclosed historical identity gap.

DatabaseSHA256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; taskSHA256775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e. Text model pin in sources/l117/text_model.json. Source hashes in _budget_l137.json, _sources_l137.json and _upstream_l137.json; FE training artifacts record live source hashes and exact matrix identity.

## Frozen error-analysis protocol

History counts result rows strictly before each query in the capped snapshot; low≤22, the training-query median. Stale recency>180days; no history gives infinity. Recent-slot missingness uses24SQL past-slot columns, threshold≥.5. All six complementary slices are reported; they overlap across dimensions. Min support30queries and10drivers. Average per-query losses across five runs, then nominate the largest positive supported validation slice, lexical tie break. No candidate means no nomination. frozen.json saves validation-only decision and hashes before test slice analysis. Primary test scores may already exist in trainer artifacts; this boundary applies to slice selection, not physical nonexistence of test outputs.

Selected high_history: validation365queries/35drivers,gap+.574672; test359/19,gap+.088247, driver-cluster percentile95%[-.370139,.641112]. Test observed_recent_slots has0rows; recent_history has16rows and is also unsupported. Global test gap+.140779, interval[-.110617,.402166]. No superiority, equivalence or causal mechanism inferred.

Intervals use2000whole-driver replacement draws,seed137, retaining query weights. They condition on the fitted models and split; shared race/time dependencies remain. They do not cover training or cross-database uncertainty. Matched integer seeds across families are bookkeeping, not common random numbers. This test split was used in prior lessons; no new pristine confirmatory holdout is claimed. Proposed feature-injection intervention NOT_RUN.

## Exact author commands

From repository root; fresh output directories required. Existing author operators refuse overwrites/duplicate dispatch.

```bash
uv venv /tmp/l137-fe-runtime --python 3.11
uv pip sync --python /tmp/l137-fe-runtime/bin/python labs/requirements-l129-runtime.txt
uv pip install --python /tmp/l137-fe-runtime/bin/python ipython==8.31.0
.venv/bin/python labs/_check_l137.py
.venv/bin/python labs/_prepare_l137.py
.venv/bin/modal run --detach modal/l137_repro.py --mode pilot
.venv/bin/python labs/_collect_l137.py --mode pilot
.venv/bin/python labs/_pilot_check_l137.py
.venv/bin/modal run --detach modal/l137_repro.py --mode final
.venv/bin/python labs/_collect_l137.py --mode final
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_prepare_fe_l137.py
for seed in 0 1 2 3 4; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 /tmp/l137-fe-runtime/bin/python labs/_run_fe_l137.py --seed "$seed" --output "labs/evidence/l137/fe/paper/seed-$seed"
done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_order_check_fe_l137.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_audit_fe_l137.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/l137-fe-runtime/bin/python labs/_slices_l137.py
.venv/bin/python labs/_freeze_l137.py
.venv/bin/python labs/_analyze_l137.py
.venv/bin/python labs/_errors_l137.py
.venv/bin/python labs/_audit_l137.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_run_fe_l137.py --seed 129 --trials 2 --output labs/evidence/l137/fe/pilot
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_source_check_fe_l137.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l137-fe-runtime/bin/python labs/_analyze_fe_l137.py
.venv/bin/python labs/_figures_l137.py
.venv/bin/python labs/_build_l137.py
.venv/bin/python labs/_execute_l137.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 /tmp/l137-fe-runtime/bin/python labs/_execute_fe_l137.py
.venv/bin/modal run --detach modal/l137_notebook_check.py
.venv/bin/python labs/_collect_notebook_l137.py
.venv/bin/python labs/_verify_l137.py
.venv/bin/python labs/_delivery_l137.py
```

Wait for each remote phase before collection. Source-only fresh GNN rerun helper: `_new_run_l137.py --directory /tmp/l137-repeat-001`; creates isolated operators/volume and new budget, launches nothing. Full independent FE rerun: use the standalone notebook in a fresh directory, pinned CPU environment, RUN_FULL_FE_REPRODUCTION=True. Its visible code includes SQL inputs, mappings, all50search trials and5refits. Full GNN standalone lane uses RUN_FULL_GNN_REPRODUCTION=True in the pinned GPU environment. These runtime stacks differ; do not merge them. Notebook flags alone do not enforce dollars/timeouts. Subsequent independent paid reruns need their own authorization; the completed author budget is not a reusable allowance.

## Budget, validation and failures

Authorized USD10 total. Modal rate checked2026-09-27 at https://modal.com/pricing: T4.000164/s +2physical cores*.0000131/s +16GiB*.00000222/s =.00022572USD/s. Workers900s,retries0; reserve worst-case runtime before dispatch and reject commitments+USD3overhead>USD10. Six primary slots reserveUSD1.218888. One full-notebook check slot reservesUSD.203148; total commitments+overheadUSD4.422036. Primary measured worker estimateUSD.056856; GPU notebook validationUSD.046103; combined measured worker estimateUSD.102959. Full GPU notebook time204.25s is separate in _notebook_gpu_l137_results.json. Startup/storage/invoice NOT_ITEMIZED; available credits do not enlarge budget.

FE runs locally, no paid cloud. Author search loop capped3600seconds aggregate; each validation gate600seconds. First local full-notebook attempt stopped before training because the pinned minimal uv runtime had no IPython or pip. Failure log retained; installing pinned IPython8.31.0 fixed the display dependency without changing model packages. Full-gate validation models are kept separate from primary result aggregation.

Default solution execution, separate full FE/GNN gate execution, independent source/SQL/tree/metric checks and browser/copied-Pages results are recorded in dedicated reports. Live Colab and deployment NOT_CHECKED. No leaderboard submission or publication performed. Learner mastery PENDING_WRITTEN_DEFENSE.
