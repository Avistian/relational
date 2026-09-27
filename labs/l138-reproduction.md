# L138 reproduction contract

Approved scope: **rel-amazon/user-churn**, RelBench v1 Table 6 basic RDL, five complete released-protocol runs. Published AUROC percentage points: validation **70.45 ± 0.06**, test **70.42 ± 0.05**. Predeclared descriptive closeness tolerance: **1.0 percentage point**, not a statistical equivalence test. Source commit: `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`. Source hashes and URLs: [manifest](sources/l138/manifest.json).

## Protocol

- Full historical database and task archives, SHA256 matches confirmed in [probe](evidence/l138/probe.json). Raw data includes future events for label reconstruction; the model's `get_db()` filters at 2016-01-01. No row caps or feature deletion.
- Training query dates: 2008-01-10 through 2015-07-02; validation: 2015-10-01; test: 2016-01-01. Eligibility is at least one review in `(t−91 days, t]`; churn means no review in `(t, t+91 days]`.
- Released stype inference and full-snapshot feature statistics. GloVe 300-dimensional mean-word text features, checkpoint revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`. Untimestamped metadata has no historical availability guarantee. Preparation uses seed 42; all five fits reuse the materialized features. This fixes preprocessing but does not recover historical RNG states.
- Per-table ResNet row encoders, two 128-channel GraphSAGE layers, sum neighbor and relation aggregation, node LayerNorm/ReLU, relative-day encoding, one-output linear head. BCEWithLogitsLoss; Adam learning rate 0.005; batch size 512; uniform temporal fanout 128/64.
- Ten source-defined epochs. The released condition `steps > 2000` permits **2,001 batches**, so an epoch is a capped pass. First strict maximum validation AUROC selects state. Re-evaluate validation/test with sigmoid outputs; no test selection.
- Seeds 0–4. Runtime: [requirements](requirements-l117-runtime.txt), Torch 2.5.1/CUDA 12.4 and pyg-lib 0.4.0+pt25cu124. Historical seeds and exact training environment identity remain **NOT_ESTABLISHED**.

## Commands

From repository root, ordinary checks and rebuild:

```bash
.venv/bin/python labs/_check_l138.py
.venv/bin/python labs/_verify_l138.py
.venv/bin/python labs/_source_sql_l138.py
.venv/bin/python labs/_figures_l138.py
.venv/bin/python labs/_build_l138.py
.venv/bin/python labs/_execute_l138.py
.venv/bin/python labs/_delivery_l138.py
```

Author cloud operators, in order (the ledger refuses repeat dispatch into an already reserved phase):

```bash
.venv/bin/modal run --detach modal/l138_repro.py
.venv/bin/modal run --detach modal/l138_audit.py
.venv/bin/modal run --detach modal/l138_pilot.py
.venv/bin/modal run --detach modal/l138_full.py
.venv/bin/python labs/_collect_full_l138.py
```

The operators preserve raw archives, complete graph/text features and checkpoints on volume `l138-amazon-evidence`. Before full dispatch, collect the completed `training_pilot.json` into `labs/evidence/l138/`; stdout completion may precede the volume commit. The full operator requires a five-run forecast within reserved 3,000-second workers, including 25% timing margin and 120 seconds startup per worker. It enforces the shared USD 10 cap. Do not erase reservations to bypass that guard. A guard refusal is not a score reproduction failure.

The collector waits for all five committed runs, validates query identities, epoch counts, first-maximum selection and archive hashes, and independently re-scores every prediction. Selected checkpoint files are also collected into ignored `labs/results/l138/checkpoints/`; their cloud paths, hashes and sizes are recorded in the training summary. Checkpoints and the large graph are not published to Pages. Rebuild and execute the notebook after collection. The final pinned-runtime check was dispatched as:

```bash
.venv/bin/modal run --detach modal/l138_notebook_check.py --attempt 4
```

This validates the final notebook and original-model logits/gradients in the pinned stack. It is a paid author check, subject to the same ledger. The recorded result is [_notebook_pinned_l138_results.json](_notebook_pinned_l138_results.json).

Local single-seed training in that pinned runtime:

```bash
python labs/_full_l138.py --seed 0 --epochs 10 --output /path/to/new/seed-0 --cache /path/to/materialized
```

Repeat seeds 0–4 for the complete named lane. The portable notebook visibly includes the model, graph constructor and trainer, with five-seed execution behind `RUN_FULL_REPRODUCTION=True`. It checks runtime versions but does not enforce billing. Full preparation reserves 128 GiB host RAM; cached training workers reserve 64 GiB. Live Colab remains **NOT_CHECKED**.

Cached workers load the same graph and task parquet tables directly, validate every task entity against graph rows, and mask test labels for the loader. This avoids keeping redundant raw strings in memory. Labels, order, model features and update protocol are preserved. Fresh preparation can consume RNG differently from cached preparation; neither path establishes historical bitwise identity.

## Independent evidence

- [Task audit](evidence/l138/task_audit.json): all **5,470,060** labels and eligible rows reconstructed from **20,862,040** raw reviews; zero mismatches. Future events are used only for label verification.
- Original published task SQL executed on 36 real histories; all targets agree. See [_sql_source_l138_results.json](_sql_source_l138_results.json).
- Fixed course recency baseline: validation AUROC **0.5986550615**, AP **0.7072266245**; test AUROC **0.5822758772**, AP **0.6628364115**. No fit or tuning. [All keyed held-out predictions](evidence/l138/recency_predictions.npz). This is separate from Table 6 RDL and LightGBM.
- Source AST identity for executed model/graph primitives, archive hashes, independently computed metrics, unique query keys, and rejected endpoint/tie/leakage mutants. The model omits only the unused recommendation method `forward_dst_readout`; classification methods match.
- The default notebook reconstructs 36 real examples, re-scores all **761,677 recency** and **3,808,385 neural** held-out predictions, and executes a synthetic full neural forward/backward pass. Re-scoring archived outputs does not retrain a model.
- Full model outcomes: [training summary](evidence/l138/training.json), [execution ledger](evidence/l138/reproduction.json). Every sampled timestamp is checked against its owning cutoff, and every training target against its query. Each seed audits 15,104,717 query occurrences across training and evaluation.
- Desktop, mobile, keyboard/reset, no-JavaScript, print and copied Pages checks: [_delivery_l138_results.json](_delivery_l138_results.json). Author reference execution does not establish learner mastery.

## Budget

**USD 10 aggregate**, including all seeds, attempts and validations; **USD 3 overhead reserve**. [Prices](https://modal.com/pricing) checked 2026-09-27: T4 USD 0.000164/s; physical CPU USD 0.0000131/s; RAM USD 0.00000222/GiB/s. No automatic retries; credits do not expand the cap.

| Phase | Reservation | Maximum compute cost, USD |
|---|---|---:|
| Archive/text probe | T4, 2 CPU, 32 GiB, 1,800 s | 0.470232 |
| Full task audit | 2 CPU, 16 GiB, 1,200 s | 0.074064 |
| Full graph pilot | T4, 2 CPU, 128 GiB, 2,700 s | 1.280772 |
| Five full fits | T4, 2 CPU, 64 GiB, 3,000 s each | 4.984200 |
| Four notebook validations | 2 CPU, 16 GiB, 600 s each | 0.148128 |
| Overhead reserve | Startup, commit and other overhead | 3.000000 |
| **Conservative total** | | **9.957396** |

See [_budget_l138.json](_budget_l138.json) for dispatch reservations and timing-based estimates. Worker-body timing excludes startup and volume commit. Invoice charges are **NOT_ITEMIZED**; the reservation is not a measured invoice.

## Publication-table alignment gap

[RelBench v1](https://arxiv.org/html/2407.20060v1) Table 2 reports **4,732,555** training queries; Table 13 reports **2,956,658** positives. The checksum-matched archive has **4,708,383** queries and **2,937,827** positives: **24,172 fewer rows** and **18,831 fewer positives**. Held-out row and positive counts match. See [count audit](evidence/l138/paper_alignment.json).

The final NeurIPS paper retains that training count and the 70.45/70.42 AUROC values. Earlier registry commits record other task archive hashes, but their bytes and the exact paper training artifact have not been recovered. The cause of the discrepancy is unknown. Full released-pipeline execution and numerically close scores do not establish exact historical paper reproduction.

**Historical reproduction: NOT_ESTABLISHED. Whole-paper/all-task reproduction: NOT_RUN. Deployment and live Colab: NOT_CHECKED. Learner: PENDING_WRITTEN_DEFENSE.**
