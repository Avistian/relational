# L070 v2 reproduction contract

## What the default experiment establishes

The primary comparison has seven arms on five fixed numeric binary tasks and three downstream seeds: **six archived procedures plus 15 fresh corrected TabM-mini-v2 fits**. The default additionally fits 15 XGBoost replay controls and makes 30 frozen-predictor feature-erasure measurements. All 60 validation candidates and 720 neural epoch losses are retained. The original `_verify_l070_results.json`, historical/current subpanels and operators remain unchanged. An eight-arm diagnostic retains the legacy incorrect mini variant separately.

The core skill is evaluation and selection, so this is a key-parts procedure mirror with the full corrected TabM implementation and training loop visibly provided. It does not implement original pretrained networks again. Default source checking uses official TabM only as a copied-weight comparator. The main notebook runs no pretrained checkpoint downloads or paid/cloud jobs.

## Exact bounded protocol

- Data: diabetes OpenML 37, blood_transfusion 1464, kc1 1067, phoneme 1489, sklearn WDBC. At most 600 rows sampled without labels using seed 70; 60/20/20 stratified partitions use 70 then 71. Saved original IDs and source hashes bind all arms. WDBC positive label 1 means benign.
- Numeric imputation: training median. TabM additionally uses training StandardScaler; pretrained wrappers retain their own released preprocessing. No test-derived preparation or selection.
- Fresh XGBoost: 100 trees, depth 3/6, learning rate .05, subsample .8, colsample .8, hist, 1 job, random seeds 0/1/2. Two candidates selected on validation log loss.
- Corrected mini: `relkit/tabm_v2.py`, SHA256 `0fbd840771d45973f4519bac409630ad000946bdc0f01323da08275608e5694e`; k=8, width=64, depth=3, dropout=.1; AdamW learning rates .001/.003, weight decay 1e-4; batch 128, 24 epochs, one torch thread. Last minimum epoch and first minimum candidate. No categorical embeddings, paper search or k32 benchmark reproduction.
- Test metric: binary log loss in nats, P(y=1), float64 epsilon clipping. Fresh records retain all candidate validation probabilities and selected weights/runtime identities. Original archive has test probabilities and validation scalar errors; a separate original-source replay supplies stronger validation reconstruction.
- Intervention: select the feature with greatest absolute **training** Pearson target correlation; replace only that query feature with its training median; no refit. Same selected models, test IDs and targets. This is a stress test, not causal importance or retraining after deletion.
- Aggregation: average three seed losses within dataset, rank dataset means with average ties, average ranks over five datasets. Paired bootstrap resamples whole dataset-mean gaps, 2000 replicates seed 70. Report per-dataset sample SD and conditional paired seed intervals separately. Five datasets remain five units.
- Cost: sum candidate fitting/validation plus one selected prediction. Common loading/imputation excluded; archived wrapper boundaries and shared CPU load prevent controlled efficiency or cache claims.

## Run and rebuild locally

From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-labs.txt
.venv/bin/python -m pip install rtdl_num_embeddings==0.0.12
.venv/bin/python labs/_check_l070_v2.py
.venv/bin/python labs/_source_check_l070_v2.py
.venv/bin/python labs/_identity_l070_v2.py
.venv/bin/python labs/_verify_l070_v2.py --preset lab --output /tmp/l070-fresh.json
```

The notebook and versioned runner call `_prepare_l070_v2.py` before the experiment. It verifies and stages the four exact public Parquet snapshots shipped in `data/l070-v2/`; the manifest attributes their OpenML sources. Existing matching files are reused; a conflicting cache is rejected and preserved. This keeps archived file hashes reproducible without relying on a fresh Parquet serialization. Breast cancer data comes from the pinned sklearn dataset.

`smoke` performs one dataset/seed with 4 epochs/30 trees; it is an execution test. `lab` is the measured five-dataset protocol. `closer` retains those tasks and increases to 128 epochs/800 trees; this changes local optimization budgets while archived pretrained inference stays fixed. A complete original-paper preset is not implemented. Choose a **new output path**; no resume or overwrite is promised. The runner writes after each selected arm/seed so interruption retains partial evidence, which must not be promoted to a complete panel.

The canonical committed author evidence is `_verify_l070_v2_results.json`; a rerun at `/tmp` does not replace it automatically. When intentionally refreshing author evidence, archive the old result/operator, use a fresh run and regenerate:

```bash
.venv/bin/python labs/_analyze_l070_v2.py
.venv/bin/python labs/_figures_l070_v2.py
.venv/bin/python labs/_build_l070.py
.venv/bin/python labs/_execute_l070.py
```

`_build_l070.py` owns the manuscript rendering, reference, student notebook, ignored teacher notebook and prepared HTML. Teacher execution is recorded in `_execution_l070_v2_results.json` with notebook input/output/code, executor and actual EXIT hashes. It executes the visible live definitions, not imported solutions. Student notebooks retain blanks. Changing executable definitions requires new execution evidence; no hash relabeling of old results is allowed.

## Replay the original pretrained predictions

The independent parent-owned operators and reports in `reviews/lesson-quality-audit-047-070/` validate original package inference separately from the teaching runner. Historical inference uses tabpfn 2.0.9 / tabicl 0.1.4; current archived inference uses tabpfn 8.5.0 / tabicl 2.2.0. Checkpoints, immutable revisions and exact hashes are in `_sources_l070_v2.json` and `_sources_foundation.json`. This lesson's TabICL-v1.1 May checkpoint is not lesson 066's earlier February checkpoint. Synthetic TabPFN-2.5 is not the Real-TabPFN variant.

Original scientific operators are retained as `relkit/foundation_benchmark.py`, `_run_current_foundation.py` and `_assemble_foundation_checkpoint.py`. To repeat the historical experiment, use the explicit legacy route and a fresh output path. The default lesson-70 route now runs the corrected experiment:

```bash
# Isolated original package environments, exact immutable checkpoints required.
# --legacy selects the preserved original historical operator.
.venv/bin/python labs/_run_foundation.py --lesson 70 --legacy --preset lab --output /tmp/l070-historical-original.json
.venv/bin/python labs/_run_foundation.py --lesson 70 --current --output /tmp/l070-current-original.json
.venv/bin/python labs/_assemble_foundation_checkpoint.py --historical /tmp/l070-historical-original.json --current /tmp/l070-current-original.json --output /tmp/l070-original-complete.json
```

The historical replay reproduces the legacy mini operator under its legacy name. It must not replace the corrected primary panel. The historical scripts' original resume checks bind fewer dependencies than the current lesson's live graph; use fresh output paths and independently verify all package/data/checkpoint identities. These commands are for original inference/measurement, not pretraining. Download/source access and any package compatibility fixes belong in the new run's ledger.

## Original-paper results: precise remaining work

- **TabPFN-3:** [report 2605.13986v1](https://arxiv.org/html/2605.13986v1), §§2.1–2.4, Appendix C/E.2/E.3. Its TabArena protocol has 816 tasks/51 datasets, 8-fold bagging fit-time procedure and full-training refit; binary AUROC, multiclass log loss, regression RMSE. Reported models use multiple estimators and GPU workloads; current local one-estimator CPU runs do not reproduce those costs. Main TALENT comparisons use classification accuracy/RMSE and exclude 26 development datasets. Full benchmark/subset/tuning, bagging, score fallback and hardware reconstruction remains NOT_RUN.
- **TabICLv2:** [report 2602.11139v1](https://arxiv.org/html/2602.11139v1), §3 / App A for repeated grouping, target-aware embeddings, QASSMax and many-class encoding; Apps J/K for benchmark metrics/protocol. Its three-stage synthetic pretraining, Muon optimizer and full benchmark results are NOT_RUN locally.
- **TabPFN-2.5:** [report 2511.08667v1](https://arxiv.org/html/2511.08667v1), §3/4 and App C. Synthetic and real-data-fine-tuned identities differ. Our synthetic one-estimator checkpoint does not reproduce Real-TabPFN-2.5, tuned/ensembled TabArena-Lite or internal datasets.
- **TabM:** [paper 2410.24210v3](https://arxiv.org/html/2410.24210v3), §3.3/5.1; [official source](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4). Corrected mini forward/input/all-parameter gradient parity is checked. Full 46-dataset experiment, paper hyperparameter search, member budgets and embedding variants remain NOT_RUN.

No single local command implements these heterogeneous full protocols. The next research step is to freeze one specific paper table's released dataset/split/metric/configuration manifest, run its official evaluation framework, then reconcile every prediction and aggregation decision. Full original pretraining additionally needs the corresponding prior/training code, data generator and compute. A larger CPU preset or a close rank is not a fidelity certificate.

## Evidence and access

Verified locally: saved/raw-row audit, corrected model source parity, complete fresh bounded fits and intervention, actual current-kernel execution and independent replay where its report records PASS. Browser rendering, copied Pages staging, public deployment and live Colab are separate checks. No live Colab session, cloud run, full pretraining or user mastery is claimed by this package.
