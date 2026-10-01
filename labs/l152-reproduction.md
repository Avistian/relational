# Lesson152 · Regression portfolio reproduction

Approved2026-10-01. Named target: Robinson et al., RelBench v1 arXiv2407.20060v1 Table7, basic RDL rel-f1/driver-position. Published validation3.193±.024/test4.022±.119MAE. Frozen descriptive mean tolerance±.20; closeness is not equivalence.

## Frozen protocol

| Axis | Contract |
|---|---|
| Source | RelBench9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; historical training commit unknown |
| Database | Full nine-table F1 graph,74063rows,338842directed edges |
| DB SHA256 | ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task SHA256 | 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Queries | 7453train/499validation/760test; complete(driverId,nanosecond cutoff)keys |
| Target | Mean positionOrder in(cutoff,cutoff+60days]; future participation conditions eligibility |
| Model | Four-block Frame ResNet row encoders,128channels,relative-time addition,two typed sum-GraphSAGE layers,scalar MLP |
| Text | GloVe model/revision pinned in sources/l117/text_model.json |
| Training | Seeds0–4; ten complete epochs; Adam.005; batch512; L1 |
| Sampling | Uniform128/64fanouts; owner cutoff at every hop; paper table reports128 |
| Preprocessing | Type proposal seed42; full database through test cutoff, not train-only statistics |
| Selection | First strict validation-MAE minimum; test never selects |
| Predictions | Released train-target2nd/98thpercentile clipping; final validation resamples |
| Diagnostics | MAE primary; RMSE and bias secondary; unique validation-prediction quintile edges per seed frozen before test bins; strict median inequalities with ties |
| Runtime | Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame.2.3,RelBench1.1.0; full requirements-l117-runtime.txt and Modal image |

## Execution and audit

Seed0 is the included full ten-epoch feasibility pilot. Each primary worker constructs a fresh graph and verifies archive hashes. Every yielded batch verifies original edge/node identity, disjoint query ownership and timestamps. No fitted calibration correction, tuning, or test-based changes. Prior F1 test exposure is disclosed.

`evidence/l152/summary.json` contains independently rescored6295primary predictions, all five results and diagnostics. `label-audit.json` reconstructs all8712labels from raw result rows. `_task_audit_l152_results.json` additionally regenerates the complete timestamp grid and compares all queries with the pinned original SQL and archive, including empty cutoff windows. This second independent check completed after primary training; raw-label/archive checks preceded training. `_audit_l152_results.json` records independent SQL graph counts, all held-out archive identities, and upstream byte checks. `_source_check_l152_results.json` records exact source AST and small local neural output/gradient/Adam parity. Real first-backward nonfinite gradients are retained and counted per seed; source parity does not prove gradient health.

The portable notebook's default path runs the live learner functions on embedded author evidence. Its full gate uses the same visible functions and complete model/trainer to train five fresh fits; own evidence is labeled separately. Notebook execution does not grade the written defense.

## Commands

From the repository root:

```bash
.venv/bin/python labs/_check_l152.py
.venv/bin/python labs/_labels_l152.py
.venv/bin/python labs/_source_check_l152.py
.venv/bin/modal run --detach modal/l152_repro.py --phase pilot
.venv/bin/python labs/_collect_l152.py --seeds 0
.venv/bin/modal run --detach modal/l152_repro.py --phase remaining
.venv/bin/python labs/_collect_l152.py --seeds 1,2,3,4
.venv/bin/python labs/_labels_l152.py
.venv/bin/python labs/_audit_l152.py
.venv/bin/python labs/_analyze_l152.py
.venv/bin/python labs/_figures_l152.py
.venv/bin/python labs/_build_l152.py
.venv/bin/python labs/_execute_l152.py
.venv/bin/modal run --detach modal/l152_notebook_check.py
.venv/bin/python labs/_collect_notebook_l152.py
.venv/bin/python labs/_delivery_l152.py
.venv/bin/python labs/_verify_l152.py
```

Author paid commands are single-use: reservations reject repeats, source changes reject dispatch, and remote output directories cannot overwrite earlier evidence. For an independent replay use the portable notebook in a fresh directory with the pinned GPU runtime and RUN_FULL_REPRODUCTION=True. It downloads/validates archives and rebuilds graphs. Local/notebook execution has no monetary guard; inspect the runtime and full procedure first. Never delete the completed author ledger to bypass its guard.

## Cost and evidence boundaries

USD10aggregate hard cap. T4+2physicalCPU+16GiB=.00022572USD/sec at https://modal.com/pricing checked2026-10-01. Eight proposed1800-second reservations=USD3.250368; USD3overhead reserve; marginUSD3.749632. The actual reservation ledger is authoritative. Pilot must finish with1.25*seconds+120<1800before remaining fits. No automatic retries. Failed attempts consume reservations. Worker-body estimates exclude startup/other unitemized resources; they are not invoices.

Historical identity and feature-arrival legality NOT_ESTABLISHED. Whole-paper reproduction and fresh manual-FE NOT_RUN. Author five-seed completion establishes the selected released-protocol experiment only. Learner PENDING_WRITTEN_DEFENSE. LiveColab and deployment NOT_CHECKED. No publication requested.


## Completed primary result and portable execution

Primary five-seed validation MAE3.180178±.049613; test MAE3.970840±.162612; secondary test RMSE4.840815±.205145. Both validation and test mean deviations lie within the descriptive±.20band. Primary original-model held-out output error≤3.8147e-6. All403895query occurrences passed ownership/time/edge checks. All five first backwards contain640nonfinite gradient entries; retained source behavior, not a claim of healthy optimization.

The isolated portable notebook executed all23code cells including five additional ten-epoch fits. Its validation MAE3.168762±.020882/test MAE4.047167±.175830;6295additional predictions independently rescored. These do not enter the primary mean. Exact bitwise repeatability is NOT_ESTABLISHED. Code hashes in `_execution_l152_results.json` and `_notebook_l152_results.json` identify the checked definitions, independently of later prose-only refreshes.

Five primary1800-second worker reservations plus one1800-second full-notebook reservation totalUSD2.437776; addingUSD3overhead reserve givesUSD5.437776. No failed paid attempts or training retries. Actual invoice NOT_ITEMIZED. Full notebook runtime196.06seconds; primary worker-body estimateUSD.059556. Final combined measured resource estimate is recorded in `_verify_l152_results.json`.
