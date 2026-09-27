# Lesson 122: REG construction and full selected reproduction

## Actual result

COMPLETE selected released-protocol replay: five fresh full-data, ten-epoch RelBench v1 Table 7 F1 driver-position RDL runs. Validation MAE 3.18869880 ± 0.02246053; test 4.05511387 ± 0.15170905 (mean ± sample seed SD). Published means 3.193/4.022; both CLOSE under predeclared descriptive tolerance 0.2 MAE. This is not statistical equivalence, Fey beta-result reproduction, whole-paper parity, or learner mastery.

All 6,295 final predictions independently rescored; maximum original-model output discrepancy 3.8146973e-06. Fifty complete training epochs across five seeds; each epoch consumes all 7,453 training queries. Prior L117/L119/L120 checkpoints and predictions were not reused.

## Complete construction audit

Our visible `relkit/reg_l122.py` constructs all 74,063 released rows into nine node types and 26 directed relation types, 338,842 directed edges. Every forward edge agrees exactly with an independent SQL join and pinned upstream constructor; reverse edges agree after sorting. `evidence/l122/reg-topology.npz` holds all node counts and edge arrays. `evidence/l122/key-tables.json.gz` contains every key row extracted from the pinned archive; the standalone student/solution notebook reconstructs the full graph using its three live functions. Full row-feature payloads are excluded from this portable topology input. `results/l122/constructed-reg.pt` is a locally saved PyG HeteroData object; load only this trusted artifact with `torch.load(path, weights_only=False)`.

Local original-code topology comparison uses constant features and removes time attributes on a deep copy. The historical `to_unix_time` helper tries to mutate a read-only array under local pandas. This explicit topology-only comparison does not claim feature or timestamp parity. Full original feature/time execution occurs separately in the pinned GPU runs. Key permutation, isolated nodes, FK roles, null/dangling references and junction multiplicity are independently tested; six semantic mutants are rejected. Static correctness does not prove historical temporal validity.

## Frozen protocol and deviations

| Item | Executed contract |
|---|---|
| Paper | https://arxiv.org/html/2407.20060v1 Table 7 RDL F1; AppendixB/Table9 |
| Release | RelBench1.1.0 commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; original files and MIT license in `sources/l117/` |
| Canonical visible implementation | `relkit/rdl_l117.py`, SHA256 `8417d096b73354c0b08c4378e705bf4ba98be467b39a47c109513d13107029c2`; runner `_run_l117.py` |
| Database | Nine tables,74,063 release-censored rows at2010-01-01; no row cap |
| Database identity | `https://relbench.stanford.edu/download/rel-f1/db.zip`; SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482` |
| Task identity | `https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip`; SHA256 `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e` |
| Queries | All 7,453 train /499 validation /760 test; published time split |
| Encoders | Four-block table-specific Frame ResNet, width 128; numerical/categorical/timestamp encodings and frozen GloVe sentence vectors; relative-age encoding |
| Text | `sentence-transformers/average_word_embeddings_glove.6B.300d`, revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`; CPU text embedding |
| Graph | Row-to-node PK/FK mapping with separately typed reverse edges; every forward relation checked against key lookups; independent SQL census |
| GNN/head | Two typed sum-GraphSAGE layers, width 128, relation-specific root maps, summed relations, node-wise LayerNorm/ReLU; scalar seed-node head |
| Sampling | Native temporal NeighborLoader, disjoint query context, batch 512, uniform fanout[128,64], num_workers0 |
| Optimization | Adam .005, mean L1 loss, ten complete epochs; default released initialization/reset with each declared seed |
| Selection | First lowest validation-MAE epoch; restore full state; validation/test resample neighborhoods after selection |
| Evaluation | Clip to train-target2nd/98th percentiles; official MAE and independent scalar scorer; compare separate original Model on identical sampled inputs |
| Seeds |0–4; five fresh runs; seed100 one-epoch pilot is excluded from reported scores |
| GPU runtime | Python3.11, torch2.5.1+cu124, PyG2.6.1, Frame0.2.3, RelBench1.1.0, pyg-lib0.4.0+pt25cu124; remaining pins in `requirements-l117-runtime.txt` |
| Preprocessing deviation | Released full test-censored materialization; statistics are NOT train-only. Type inference fixed seed42 before per-run training reset |
| Paper/source deviation | Source fanout[128,64] versus paper Table9 neighbor count 128 |
| Unknowns | Historical seeds/random state/package identity; real ingestion/revision histories; all other paper tasks and fresh tabular baselines |

The course dual-clock graph walk uses explicitly known event/availability timestamps. The released benchmark only establishes its available event-time semantics. Do not transfer the course fixture's stronger availability guarantee to the benchmark.

## Exact commands

Run from the relational root:

```bash
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l122.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_mutation_l122.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l122.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_graph_audit_l122.py
.venv/bin/modal run --detach modal/l122_repro.py --mode pilot
.venv/bin/python labs/_collect_l122.py --mode pilot
.venv/bin/python labs/_pilot_check_l122.py
.venv/bin/modal run --detach modal/l122_repro.py --mode paper
.venv/bin/python labs/_collect_l122.py --mode paper
.venv/bin/python labs/_analyze_l122.py
.venv/bin/python labs/_audit_l122.py
.venv/bin/python labs/_figures_l122.py
.venv/bin/python labs/_build_l122.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l122.py
.venv/bin/python labs/_delivery_l122.py
.venv/bin/python labs/_verify_l122.py
```

The shipped budget ledger already contains these dispatches: attempting another pilot/paper dispatch refuses duplicate reservations. To reproduce independently, use a NEW Modal app/volume and a NEW ledger with empty reservations and `pilot_approved_for_full=false`; refresh source hashes and rates, preserve the USD10 total, then execute pilot→gate→five fits. Do not erase old evidence. Budget enforcement is bounded resource reservation, not live invoice monitoring. Two unused worker slots do not auto-authorize another five-seed batch.

The complete model/trainer is `relkit/rdl_l117.py`, with `_run_l117.py` preparing raw data and the pinned GPU environment in `requirements-l117-runtime.txt`. Both notebooks inline the model/trainer and have an OFF-by-default full-data five-seed gate. The gate checks runtime versions and archive hashes, but does not enforce cloud monetary reservations. Use the Modal route for the bounded author run. Default notebook execution requires no repository imports: it rebuilds full key topology and replays saved author predictions.

## Budget and evidence limits

Current-rate plan (2026-09-27, https://modal.com/pricing): T4 USD.000164/sec +2 physical cores at.0000131/core/sec +16GiB at.00000222/GiB/sec =.00022572/sec. Eight one-hour worker reservations cap requested worker resources at USD6.500736; USD3.499264 reserved for overhead. One pilot plus five full workers were consumed; no retries. Timed pilot conservative projection (ten times entire pilot per full seed) USD.283922. Actual completed-worker estimate, pilot included: USD0.05259377. This is not an invoice and excludes unitemized build/startup/storage overhead. No additional paid work is scheduled.

Source files were freshly compared against commit 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. `sources/l117/manifest.json`, licenses, text-model revision and wheel/runtime pins retain source provenance. Raw archive hashes and all seed records are preserved; `_audit_l122_results.json` compares task IDs, timestamps and labels against the downloaded release. `_graph_audit_l122_results.json` records full topology identity. `_source_check_l122_results.json` records CPU model-output, gradient and optimizer-step parity on a small fixture. These are separate evidence scopes.

Full-paper/historical identity NOT_ESTABLISHED; all other paper tasks and fresh baseline training NOT_RUN. Real ingestion/revision histories unavailable. Browser/notebook/copied Pages verification is reported only in its dedicated reports. Live Colab and deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.
