# L178 — Matched-Information F1 Comparison

Approved 2026-10-02. **Fresh comparison: INCOMPLETE_TRAINING_HEALTH_GATE.** Complete selected published-result replay: **30 runs / 21,060 predictions / 702 queries**. No fresh benchmark model fit or inference, cloud dispatch, repaired-model comparison or whole-paper reproduction was performed.

## Two different experiment lanes

Published replay: RDB-PFN arXiv2603.03805v5 Table9, full rel-f1/driver-dnf, support512, seeds0–9, RDBPFN/RDBPFN_single/TabICLv1.1. Authenticate every NPZ against original per-run receipts; regenerate support indices; rescore complete keyed predictions; compare every seed with the saved original report. These are reused author predictions, not 30 newly executed model runs. Fixed references .7219/.6640/.7176 and original descriptive tolerance .02 are inherited from L166, not newly selected.

Fresh course protocol: same1024 support keys for each arm and seed0–2, all566 validation and702 test queries, canonical DNF-positive30-day labels. RDB-PFN fixed6-layer checkpoint model_eval00528.pt; RelGNN128channels,4heads,1composite layer, BCEWithLogits, learning rates .001/.005,10 complete epochs each; RDBLearn0.1.2 actual FastDFS0.2.1 depth2 SQL pipeline with TabICL0.1.3/v1.1/32estimators, target-history augmentation disabled. Select RelGNN configuration by mean best validation AUROC across all3seeds; earliest epoch/listed configuration breaks ties. Do not score test until selection is frozen. This limited, matched-label search differs from the papers' full searches and populations. It is a course experiment, not historical reproduction.

## Data and information audit

Original RDB-PFN source a95378225478daa262b85f180d482da7516b0af6, data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL checkpoint eaf789a9b25ee8486d6f48997ba076f850bbc30b. RDBLearn annotated tagv0.1.2 resolves to commit b5b03ebf8091547285a6e06cba53d2d1a40cb171 (the tag object33f0949b23e732b59625114b3295ddd195122a3c is not the commit). RelGNN cffdb8b54627e92c7dd112c1243dde739c90d35b. FastDFS wheel SHA256 e651a3b5db092a11deab0c9f01fe5797425caacb5e312833106e508994b2b0e3. Complete source and input bytes are authenticated by input-manifest.json and audit-input-manifest.json.

All12679train/validation/test released labels are exact complements of independently reconstructed raw30-day MAX(statusId!=1), using26080results rows. Both label and probability must be complemented to interpret saved scores as DNF risk. Convert saved float32 probabilities to float64 before 1-p so new rounding ties do not change the AUROC audit. Changing only the positive-class name is wrong. Extending the label window to60days changes627train/56validation/82test labels. Earlier60-day support-availability checks were conservative admission bounds, not the task definition. Task membership conditions on future participation; this is not an all-driver prospective cohort.

The three1024-support schedules are regenerated from the original SHA256 seed convention; all support outcome windows finish before validation starts. Equal support sizes alone do not prove equal label access: RDBLearn's fit stores full target history before downsampling when augmentation is enabled. Disable it, pass only the approved support, and keep raw historical database facts common to all arms.

## Executed preflights and stop

Actual FastDFS depth2 produced65columns for seed0support query driver34 at1999-08-01. Replacing all numeric non-key cells at or after this cutoff in six timestamped tables changed zero feature values. This one-query intervention is a diagnostic only; complete all-query/all-feature regeneration and temporal audit remain NOT_RUN.

The original RelGNN HeteroEncoder numerical branch ran on all58earlier results belonging to the same real support query. Eight numeric non-key columns include30missing milliseconds and21missing position cells. The output and squared-readout loss are finite, but256numerical weight-gradient entries are not. PyTorch Frame0.2.3 LinearEncoder performs the affine operation before its fallback nan_to_num; a zero upstream gradient multiplied by NaN remains NaN. Independent NumPy arithmetic and a separate original LinearEncoder execution reproduce256nonfinite entries. An early-imputation diagnostic control produces zero. That control is not a repaired full RelGNN fit or proof of unchanged model behavior.

Runtime boundary: this prerequisite used local CPU Torch2.13.0+cpu and source-pinned PyTorch Frame0.2.3, not the historical Torch2.5.1CUDA stack. It is a real local training-health failure, not a claim that every RelGNN deployment fails. The approved fail-closed protocol stopped before any model pilot; all6planned GNN fits and all6planned ICL evaluations remain NOT_RUN. No ranking or accuracy-cost frontier exists. A repaired encoder, full temporal audit and pinned GPU validation would require a newly frozen continuation protocol.

Failed setup attempts are preserved: inherited Pydantic1 did not honor FastDFS's Pydantic2model_config; an isolated2.11.7environment fixed it without changing the course environment. The first raw-task metadata path was wrong. Gradient probe import and duplicate-key-column export failures are retained. The final probe excludes PK/FKcolumns, as the graph materializer does. No failed attempt is erased from the local budget ledger.

## Execute the complete saved-evidence and raw-label audit

From repository root:

```bash
.venv/bin/python labs/_budget_l178.py .venv/bin/python labs/_check_l178.py
.venv/bin/python labs/_budget_l178.py .venv/bin/python labs/_audit_l178.py
.venv/bin/python labs/_budget_l178.py .venv/bin/python labs/_verify_l178.py
.venv/bin/python labs/_figures_l178.py
.venv/bin/python labs/_build_l178.py
.venv/bin/python labs/_budget_l178.py .venv/bin/python labs/_execute_l178.py
.venv/bin/python labs/_delivery_l178.py
.venv/bin/python labs/_checkout_l178.py
```

The portable solution embeds every input byte and the live scoring, comparison-gate and selection functions. It reconstructs every published prediction score and every raw label, and independently checks the numerical failure witness. It does not retrain a model, rerun FastDFS or perform a full backward pass by default. Exact original model/featurizer code and the actually executed diagnostic scripts are visible inline. No established runnable full L178 cloud operator is claimed after the early stop.

## Cost and delivery boundaries

USD10aggregate cap; USD8planned stop; USD1.50per arm and USD3shared overhead ceiling. Actual new cloud/API spendUSD0; dispatches0. No paid pilot was admitted, so no full-comparison cost estimate is established. Local numerical cap3600seconds includes dependency preparation, failed attempts, checks and notebook executions. A reservation is not an invoice. Author preparation is not learner mastery: PENDING_WRITTEN_DEFENSE. Live Colab and deployment NOT_CHECKED; no push requested.
