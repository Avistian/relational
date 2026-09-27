# L118 reproduction contract: Cvitkovic Home Credit GCN

## Current result

**Selected paper experiment: NOT_RUN. Full reproduction is not complete.** Data access succeeded using the user's Kaggle credentials, stored outside the repository. A full-width T4 timing pilot on 1,024 real application graphs completed. Its current-port estimate exceeds the approved USD10 aggregate ceiling. No five-fold job was launched.

Paper target: expanded arXiv `2002.02046v1`, Table 4 Home Credit GCN, AUROC **0.780 +/- 0.004**, five cross-validation folds. Predeclared descriptive mean tolerance: **0.01 AUROC**. This tolerance is not a statistical equivalence test. Full three-dataset/all-model reproduction and historical identity remain **NOT_ESTABLISHED**.

## Source identity and protocol audit

Author source: https://github.com/mwcvitkovic/Supervised-Learning-on-Relational-Databases-with-GNNs at `57195ccab62d23dcbcac1a317f8a9811a9fd6cb5`. The vendored source subset and per-file hashes are in `_sources_l118.json`. Author code is MIT licensed; DGL v0.3.1 primitive source is Apache-2.0 licensed. Paper history: 2019 workshop version versus expanded 2020 arXiv experiment; the latter supplies this target.

| Component | Released choice | Port/evidence |
|---|---|---|
| Target population | 307,511 labeled applications; 48,744 extra unlabeled applications in prepared data | Full released fold identities independently reproduced; raw labeled population scanned |
| General algorithm | Incoming closure, then outgoing closure, induced directed multigraph | Independent transitive-closure oracle, 792 cases |
| Home Credit extraction | Neo4j undirected 0..2-hop paths around target application | Original builder preserved; bounded pilot reconstructed from CSVs, NOT checked against Neo4j |
| Feature values | Source Cypher casts, categorical vocabularies and scalar statistics from metadata; remove Application.TARGET | Pilot interprets original property expressions; full trainer consumes author-format graphs |
| Numeric encoding | Float32, subtract center, divide scale, add 1e-7, missing flag, clip [-5,5] | Visible implementation and controlled missing/clipping checks |
| Category encoding | Index 0 missing/unknown; learned min(cardinality,32) embedding; initialization clipped [-2,2] | Visible implementation; original initializer parity |
| Row model | Per-table d→4d→256; SELU; hidden and embedding dropout .5 | Pinned original TabMLP compared after copied weights |
| Graph model | One 256→256 GCN, symmetric degree normalization, SELU/dropout .5; reverse and self edges supplied by collator | Original GraphConv Python executed with independent dense adapter; controlled output/gradient agreement |
| Readout | Two-layer gate and two-layer value MLP; graph-local softmax; weighted sum; linear 256→2 | Original readout Python parity and graph-batch separation |
| Optimizer | AdamW 1e-4, weight decay 0; StepLR gamma1 has no effect | Visible trainer; modern runtime |
| Splits | Sorted IDs; shuffled KFold(5, random_state14); train_test_split(test_size .15, random_state14) inside trainval | All original train/val/test IDs match exactly |
| Model randomness | Seed1234 reset for each fold | Preserved; these are folds, not independent seed replications |
| Selection | Validate before training; first strictly largest validation AUROC; patience50; max300epochs | Preserved; tiny complete save/reload trainer exercised |
| Final validation | Source validates after final training but does not update best_auroc | Port omits unused final validation, replays selected best state |
| Metrics | Held-out per-fold AUROC, mean and sample SD across five folds | Runnable independent prediction rescore; full predictions NOT_RUN |
| Original manual search | Validation-based choices and earlier learning-rate sweep | Not reconstructed; released final settings fixed |

The original `populate_db_info` reads full tables. Its global scaling/vocabulary is preserved by loading released metadata; this is not a claim of fold-local preprocessing. Original timestamp availability and historical ingestion are not established. The CSV pilot found 5,148 references to previous applications absent from the selected per-app reconstruction. They may include genuinely missing records or context not included by that reconstruction; equivalence to the Neo4j two-hop graph has NOT been established. The pilot therefore supports resource calibration only, not source-faithful data or paper results.

## Budget and measured pilot

Current Modal standard rates checked at https://modal.com/pricing on 2026-09-27: T4 $0.000164/second, CPU $0.0000131/core/second, memory $0.00000222/GiB/second. T4 + 2 cores + 16 GiB totals **$0.00022572/second**. Pilot worker hard timeout: 1,800 seconds; maximum worker reservation **$0.406296**. No retries. Reserve **$2** for overhead/retries inside the standing **$10 total** cap. Build/startup/storage overhead is not itemized in the measured function-body estimate.

Measured pilot: 1,024 graphs, 130,886 nodes; 2,009,071 parameters; peak GPU allocation 2.655 GiB. CPU batch construction 1.3821 seconds; warm training-step median about 0.1579 seconds; evaluation step 0.0496 seconds. Six repeated training steps on the SAME batch are timing measurements, not six epochs or held-out evaluations. Function-body worker estimate approximately **$0.00129**, excluding unitemized overhead.

The estimate uses 205 training and 37 validation batches per epoch. Five folds × 300 epochs project **$124.83**, before full dataset preparation, final tests, and reserve. A 50-epoch scenario projects **$20.80**. Neither is a guaranteed invoice or stopping time. Repeated graph sizes, caching, dataloader workers, I/O, and GPU utilization may change cost materially. The current port uses zero dataloader workers and rebuilds encodings; its timing does not estimate an optimized historical pipeline. GPU-only timing suggests batching/encoding is the primary optimization opportunity, but an unmeasured optimization is not permission to exceed the cap.

**Stop decision:** retain runnable full-data path; do not launch five folds within an insufficient reservation. A future run needs either a measured optimized pipeline whose aggregate projection fits USD10, or explicit larger-budget approval. Credits do not increase the budget. The CLI preflight enforces the recorded projection plus reserve and the prior USD0.406296 pilot worker reservation. See `_budget_l118.json`.

## Exact commands already executed

Run from the repository root with the project virtual environment. Download tooling used `kaggle==2.2.4` (`.venv/bin/pip install kaggle==2.2.4` if missing); the neural runtime is pinned separately. Local source/notebook package versions are recorded in `_environment_l118.json`:

```bash
.venv/bin/python labs/_check_l118.py
.venv/bin/python labs/_source_check_l118.py
.venv/bin/python labs/_verify_l118.py
.venv/bin/kaggle competitions files -c home-credit-default-risk
.venv/bin/kaggle competitions download -c home-credit-default-risk -p labs/data/l118/raw --quiet
.venv/bin/python labs/_prepare_l118_pilot.py
.venv/bin/modal run modal/l118_pilot.py
.venv/bin/python labs/_run_l118.py
.venv/bin/python labs/_figures_l118.py
.venv/bin/python labs/_build_l118.py
.venv/bin/python labs/_execute_l118.py
.venv/bin/python labs/_delivery_l118.py
```

Actual check results live in matching JSON files; this command list is updated with completed checks during authoring. The default runner prints a plan; it does not spend money or claim completion. Pilot cannot relaunch if its existing reservation/evidence file is present. Do not remove that ledger to bypass the aggregate budget.

Raw download SHA256: `4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de`. Raw files and prepared graphs remain ignored; no credential or licensed row data is embedded in lesson/notebook artifacts. The public scalar/vocabulary metadata comes from the author repository.

## Full preparation path (NOT_RUN here)

The modern port accepts the author's `preprocessed_datapoints` format. Each file is one trusted local pickle `(edge_list, node_types, edge_types, features, label)`, named by application ID. This format is produced by the original Neo4j builder, not by the bounded pilot. Do not point the full runner at the 1,024-graph pilot and call it full data.

Use an isolated checkout and the author's historical environment instructions. The declared historical versions are Python3.6, PyTorch1.3.0, DGL-CUDA10.0 0.3.1, and Neo4j3.5.4. Other historical dependencies were not fully pinned. The environment has not been reconstructed or certified on this host.

```bash
git clone https://github.com/mwcvitkovic/Supervised-Learning-on-Relational-Databases-with-GNNs.git /tmp/cvitkovic-release
git -C /tmp/cvitkovic-release checkout 57195ccab62d23dcbcac1a317f8a9811a9fd6cb5
cd /tmp/cvitkovic-release
conda env create -f docker/whole_project/environment.yml
conda activate RDB
docker build -t rdb-neo4j docker/neo4j
mkdir -p "$HOME/RDB_data/raw_data/homecreditdefaultrisk"
```

Unzip the downloaded archive into `$HOME/RDB_data/raw_data/homecreditdefaultrisk`. In the original checkout, `data.utils.get_db_container('homecreditdefaultrisk')` starts the named Neo4j container with that directory mounted at `/data`. Obtain its container ID, then run the source Cypher explicitly:

```bash
python -c "from data.utils import get_db_container; print(get_db_container('homecreditdefaultrisk'))"
# Replace CONTAINER_ID with the returned ID.
docker exec -i CONTAINER_ID cypher-shell < data/homecreditdefaultrisk/homecreditdefaultrisk_neo4j_loader.cypher
python -m data.homecreditdefaultrisk.build_dataset_from_database
```

The explicit Cypher path above avoids a path inconsistency in the author's `build_database_from_kaggle_files.py`. The original container/Neo4j dependency chain may need compatibility repair on modern hosts; this path is preserved and documented, not claimed executed. Keep the released metadata file rather than silently regenerating different statistics.

Once all 356,255 graphs exist and budget is resolved, return to the teaching repository:

```bash
.venv/bin/python labs/_run_l118.py --prepared "$HOME/RDB_data/homecreditdefaultrisk/preprocessed_datapoints"
# Full execution is explicit. Default USD10 still fails the current cost gate.
.venv/bin/python labs/_run_l118.py --prepared "$HOME/RDB_data/homecreditdefaultrisk/preprocessed_datapoints" --execute --device cuda
```

`--budget-usd` is available for a separately authorized limit; do not change it merely to pass preflight. A real GPU/paid-host runtime must be provisioned separately. The runner itself launches no cloud services. It fingerprints every input, writes every fold prediction/ID, verifies coverage before aggregation, and uses an aggregate runtime deadline. No automatic resume is implemented; output directories must be new so an interrupted result cannot be mixed with a new implementation.

## Evidence boundaries

- Synthetic lesson fit: **COURSE_ONLY**.
- Full released fold identities: **EXACT** against original helper functions.
- Controlled source neural forward/gradient agreement: **PASS**, independent dense adapter on modern PyTorch.
- Real-data T4 timing pilot: **COMPLETE**, resource calibration only.
- Five full Home Credit GCN folds: **NOT_RUN**, budget and full prepared-data prerequisites unresolved.
- Historical DGL runtime/full-data model parity: **NOT_CHECKED**.
- All-model/all-dataset paper reproduction: **NOT_ESTABLISHED**.
- Live Colab and deployment: **NOT_CHECKED**.
- Learner mastery: **PENDING_WRITTEN_DEFENSE**.
