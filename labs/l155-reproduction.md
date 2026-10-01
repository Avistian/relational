# Lesson155 · Compare manual FE and RDL

Approved2026-10-01. Named experiment: **L155 rel-f1/driver-position — released manual-FE pipeline versus basic RDL**. Author preparation does not imply learner mastery.

## Frozen protocol

| Contract | Manual FE | Basic RDL |
|---|---|---|
| Source | User-study445bb7a3b1230f49f8e5890ae81754d3e365680f | RelBench9aa346267c2e1c560bd92da07d6f4ad1ca2f0639 |
| Data | Full v1 F1 archives;7453train/499val/760test | Same complete query populations |
| Inputs | Released50 SQL features plus numeric driverId; date ignored | Typed row encoders, relative time, two128-channel sum-GraphSAGE layers |
| Preprocessing | Train-fitted Frame0.2.2; LightGBM4.3.0 | Frame0.2.3; database statistics through test cutoff |
| Fit | Five seeded TPE searches0–4;10trials each;2000round cap;50round patience; selected train-only refit | Five seeds0–4;10complete epochs;Adam.005;batch512;L1;uniform128/64fanouts |
| Selection | First minimum validation MAE trial | First strict minimum validation MAE epoch |
| Output | Raw tree sum | Train-target2nd/98thpercentile clipping |
| Target | Released Section6 FE computation; historical scalar not recoverable | Table7 val3.193/test4.022, descriptive mean tolerance±.20 |

Same evaluation target and split do not imply identical information or tuning budgets. This is a comparison of complete released pipelines, not a causal architecture ablation. Figure3 regression uses boostedRDL and is NOT reproduced by this basicGNN comparison. Source sampling uses128/64 versus the paper's summarized128 setting. FE archive row order is fixed and TPE seeds explicit; original search randomness is unspecified. Historical staging endpoint unavailable; current archive bytes are hash-pinned. Prior F1 test exposure makes this a replay, not pristine confirmation.

FE history uses strict past; GNN sampling permits timestamps equal to query cutoff. Schedule publication times and mutable-attribute arrival histories are unavailable. Feature-arrival legality and historical identity NOT_ESTABLISHED. Real first-backward nonfinite-gradient observations must be retained; source-output parity does not establish gradient health.

## Independent evidence

Before primary fitting: reconstruct all8712labels; regenerate all SQL feature rows and independently reconstruct their values. All saved predictions use complete(entity,cutoff)keys. After fitting: independently sum saved tree leaves, compare original GNN outputs, rescore all12590held-out predictions, verify full seed/trial/epoch coverage and validation-only selection. A frozen input manifest makes report replay reject changed bytes.

Quality difference is FE absolute error minus RDL absolute error. Positive favors RDL. The2000-draw driver-cluster bootstrap(seed155) averages per-query errors over five fits before resampling entire drivers; it is conditional on fitted models, ignores shared race/time dependence and does not estimate transfer uncertainty. Integer seed labels do not create paired random initializations. No average across incompatible task metrics.

Human effort remains NOT_OBSERVED. Original human trial and full-paper reproduction NOT_RUN. Logs require participant, task, assistance policy, method, phase, scope, kind, timezone-aware start/end and coverage attestation. Overlapping human intervals are rejected. Machine time is separate. Shared setup is reported separately; the primary ratio measures marginal active human time. Reusing author SQL does not remeasure feature ideation. A learner's second-method practice is order/experience confounded.

## Execution

From repository root:

```bash
.venv/bin/python labs/_check_l155.py
.venv/bin/python labs/_labels_l155.py
.venv/bin/modal run --detach modal/l155_repro.py --phase pilot
.venv/bin/python labs/_collect_l155.py --seeds 0
.venv/bin/modal run --detach modal/l155_repro.py --phase remaining
.venv/bin/python labs/_collect_l155.py --seeds 1,2,3,4
# Separate requirements-l129-runtime.txt environment:
# _prepare_fe_l155.py -> _audit_fe_l155.py -> _fit_fe_l155.py
.venv/bin/python labs/_analyze_gnn_l155.py
# Pinned FE: _analyze_fe_l155.py and original-trainer parity
.venv/bin/python labs/_audit_l155.py
.venv/bin/python labs/_report_l155.py
.venv/bin/python labs/_build_l155.py
.venv/bin/python labs/_execute_l155.py
```

Author dispatch is single-use and refuses existing outputs/reservations. Independent repetitions use the portable notebook in an empty directory. Its complete FE and GNN gates use separate pinned runtimes, remain off by default, and do not enforce dollar limits. Model/trainer/SQL source is visible. The author Modal operator reserves cost before dispatch and checks frozen sources.

## Budget and delivery

USD10 aggregate hard cap. Eight1800s slots at T4+2physicalCPU+16GiB=.00022572USD/s reserveUSD3.250368; USD3overhead allowance; remainingUSD3.749632. Rates checked2026-10-01:https://modal.com/pricing. Pilot seed0 is included; require1.25*seconds+120<1800before further fitting. No automatic retries; failures consume slots. Local FE uses4threads, primary aggregate3600s timeout,USD0cloud. Worker-body estimates exclude unitemized startup/build/storage and are not invoices. `_budget_l155.json` records actual reservations; `_verify_l155_results.json` records delivered evidence and limitations.

No deployment requested. LiveColab/deploymentNOT_CHECKED. LearnerPENDING_WRITTEN_DEFENSE. The recommendation result remains INCOMPLETE/testNOT_RUN; classification manualFE remains NOT_RUN. One matched task does not satisfy the three-task Year4 exit criterion.

## Recorded outcome

Five primary GNN fits: validation3.168988±.030376/test4.013141±.208250MAE. Five primary FE searches: validation2.777330±.027905/test3.948917±.070469. FE−RDL test−.064225, driver-bootstrap95%[−.312329,+.172961]. Full selected computational pipelines COMPLETE; Table7 descriptive CLOSE. No decisive superiority/equivalence and no measured human-effort ratio.

Portable48-code-cell offline notebook and full pinned FE/GNN gates passed; each full gate independently checked6295predictions. Notebook validation fits are excluded from primary metrics. FE repeated validation is deterministic; GNN GPU reruns have stochastic variation and are independently rescored. First GPU gate failed at source hashing before fitting; its cost/failure are preserved under evidence/l155/notebook, and repaired gate under notebook-final. Notebook transport now checks inclusion of every hashed model source. Seven1800s GPU reservations plusUSD3overhead totalUSD5.844072; measured worker-body estimateUSD.108702,not an invoice. All recorded local compute138.663seconds,cloudUSD0.

The GNN-only summary retains the upstream helper field fresh_fe_comparison=NOT_RUN for its own lane. It is not the combined L155 verdict: evidence/l155/fe/summary.json and report.json contain the completed fresh comparison.
