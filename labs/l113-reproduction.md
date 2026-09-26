# Lesson 113 — scaling OGB with an auditable ClusterGCN replay

## Frozen scientific question

Replay OGB v6 §4.1 Table4 **ClusterGCN on ogbn-products** using the released GraphSAGE aggregator. The complete selected experiment is ten independent full-data 50-epoch fits, not the entire OGB paper. Targets: validation92.12±0.09%, test78.97±0.33%; sample seed standard deviations. Predeclared descriptive tolerance: each ten-seed mean within0.5 percentage point. This is not a statistical equivalence test. No tolerance verdict with fewer than ten complete seeds.

[Current evidence](evidence/l113/summary.json) · [Budget](_budget_l113.json) · [Source identities](_sources_l113.json) · [Computation checks](_check_l113_results.json) · [Original-model checks](_source_check_l113_results.json).

## Protocol ledger

| Component | Frozen definition |
|---|---|
| Publication | https://arxiv.org/html/2005.00687v6#S4.SS1, Table4 |
| Source | snap-stanford/ogb commit `cf066f93311ab3099cad84d71085d1b0375dcc2e` (2020-05-28), latest change to cluster_gcn.py before paper v6; archived source and MIT license |
| Data | Complete release-v1 products.zip from Stanford HTTPS; archive and array hashes recorded by preparation;2,449,029 nodes;61,859,140 raw undirected edge records;123,718,280 directed entries after reversal |
| Features/labels |100 released numeric features;47 classes; no feature normalization |
| Splits | Official sales_ranking IDs:196,615 train;39,323 valid;2,213,091 test; disjoint and exhaustive |
| Visibility | Full graph/features eligible; train labels only in objective; validation selects; no test-based decisions |
| Graph semantics | Append reverse edges; retain original self-edges and duplicate multiplicity; add no self-loops; per-receiver neighbor mean |
| Partition |15,000 METIS groups; recursive=False; saved once and shared across seeds; hash recorded; Torch seed113 before partitioning does not establish deterministic METIS seed identity |
| Training batches | Shuffle groups per epoch;32 groups per induced union; retain cross-edges between selected groups; skip batch if it has no training labels |
| Model |3-layer SAGE:100→256→256→47; neighbor linear with bias plus root linear without bias; ReLU/dropout0.5 after layers1/2; log-softmax; no BN or hidden-vector L2 normalization |
| Initialization | PyTorch Linear uniform ±1/sqrt(fan_in), matching distribution of released SAGEConv; independently seeded processes0–9 rather than historical unseeded continuous resets |
| Optimization | Adam lr0.001, betas0.9/0.999, eps1e-8, weight_decay0; no scheduler; mean NLL on batch training nodes |
| Schedule |50 label passes; evaluate20,25,30,35,40,45,50; no early stopping |
| Selection | First validation maximum among these seven candidates; complete state_dict saved; test from selected predictor |
| Inference | Complete-neighbor, layerwise; chunk receivers; dropout disabled; original global node order |
| Metric | Fraction correct over exact released population, checked against OGB evaluator; mean and sample SD over completed ten-seed schedule |
| Runtime | Python3.12, torch2.8.0, PyG2.6.1, OGB1.3.6, numpy2.2.6, pandas2.3.2; compiled sampling dependencies recorded; T4,4 CPU cores,32GiB host RAM; preparation64GiB |
| Limits | Preparation1800s/call; pilot900s; each full worker2500s with trainer stopping after2350s; no automatic retries; aggregate budgetUSD10 |

## Deviations and unresolved historical details

1. **Historical partition and seeds unavailable.** A fresh fixed METIS partition is preserved, not represented as the original partition. Explicit seeds0–9 do not reproduce unspecified original RNG trajectories. Modern PyG/METIS can change partition assignments and loader ordering.
2. **Sparse computation.** Native PyTorch CSR replaces PyG scatter for visible neighbor means and gradients. Repeated edges coalesce into multiplicity weights; normalization divides by the total incoming multiplicity. This is algebraically the released mean, not binary deduplication. Original-class forward/input/parameter gradients are checked with copied weights and dropout disabled; these do not prove training-trajectory identity.
3. **Inference execution.** Exact receiver chunks65536 replace the source NeighborSampler chunks1024. All neighbors remain available. Dense and original-model replay establish computation within tolerance, not identical floating-point reductions or historical hardware timing.
4. **Initialization draw order.** We instantiate a freshly seeded model once per run. The original main constructs a model then resets it per run without an explicit seed list. Weight distributions agree; random draws are not identical.
5. **Loader workers.**0 workers replace the release's12; partition batching is semantically checked. Throughput claims are specific to this implementation and hardware.
6. **Released logging bug.** The source increments total_examples twice per training batch. It halves displayed mean loss/approximate training accuracy; loss.backward() uses the correct loss. We count training examples once. This is a logging correction; optimizer objective remains mean training-node NLL.
7. **Data wrapper.** Visible gzip-CSV reading replaces PygNodePropPredDataset. Original source files, shapes, split IDs, duplicate semantics and hashes are audited. An independent official loader check is recorded separately; absence of that check is never implied to be a pass.
8. **Checkpoints.** Selected state is saved for inference replay. Optimizer/RNG-exact mid-training resume is not implemented. Existing output directories are rejected, including incomplete ones. Start a fresh explicit namespace; never silently merge partial trajectories.
9. **Budget.** Stopping for the aggregate budget or per-worker runtime leaves the named experiment INCOMPLETE. A two-epoch pilot is PILOT_ONLY even though it uses every node. Fewer completed seeds cannot establish the ten-run mean/variance result.
10. **Other paper experiments.**NeighborSampling, GraphSAINT, full-batch products baselines, other OGB datasets, original hyperparameter tuning, historical bitwise identity and whole-paper parity are NOT_RUN/NOT_ESTABLISHED.

## Separate GCN bridge

`_bridge_l113.py` uses the exact L112 GCN on full arxiv, two epochs, seed113, full graph versus four random induced groups. Same initial model, architecture, labels, learning rate and label passes; different normalization, BN populations and number of steps. Both evaluate on the full graph after the last epoch. This is TEACHING_ONLY and cannot rank converged samplers. Times exclude loading, batch construction and normalization. The notebook inlines the GCN and bridge and calls the learner's live `induced_edges`.

## Exact commands

From repository root. Use a new environment; compiled wheel command below targets Linux x86_64 CUDA12.8 (the Modal runtime). CPU or other architectures require compatible PyG wheels or building the extensions; core small-graph notebook checks do not require them.

```bash
python3 -m venv /tmp/l113-runtime
/tmp/l113-runtime/bin/pip install -r labs/requirements-l113-runtime.txt
/tmp/l113-runtime/bin/pip install --only-binary=:all: pyg-lib==0.4.0 torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
OMP_NUM_THREADS=1 /tmp/l113-runtime/bin/python labs/_check_l113.py
OMP_NUM_THREADS=1 /tmp/l113-runtime/bin/python labs/_source_check_l113.py
# Full products download/processing; allocate64GiB CPU RAM.
/tmp/l113-runtime/bin/python labs/_prepare_l113.py --root labs/data/l113
# Pilot only: complete graph, two epochs.
/tmp/l113-runtime/bin/python labs/_run_l113.py --preset smoke --device cuda --output labs/results/l113/new-smoke
# One complete schedule (one seed is not the ten-run experiment).
/tmp/l113-runtime/bin/python labs/_run_l113.py --preset closer --device cuda --seed 0 --output labs/results/l113/new-one
# All ten complete schedules. Per-run timeout does not enforce aggregate billing.
/tmp/l113-runtime/bin/python labs/_run_l113.py --preset paper --device cuda --output labs/results/l113/new-paper
```

The notebook `RUN_FULL_REPRODUCTION=False` gate is off by default. It contains the same visible functions; turning it on needs sufficient host RAM, compiled sampling libraries, a GPU, and an external aggregate runtime/cost boundary. A standard free Colab machine may not have enough CPU RAM for products preprocessing. Live Colab is NOT_CHECKED.

Author cloud commands:

```bash
.venv/bin/modal run --detach modal/l113_repro.py --mode prepare
.venv/bin/modal run --detach modal/l113_repro.py --mode pilot
# Inspect pilot and update the source-fingerprinted budget gate before any full run.
.venv/bin/modal run --detach modal/l113_repro.py --mode paper_all
# Or --mode paper --seed 0 for one individually authorized seed. Do not run both.
.venv/bin/modal run --detach modal/l113_repro.py --mode replay
.venv/bin/python labs/_collect_l113.py
.venv/bin/python labs/_analyze_l113.py
.venv/bin/python labs/_figures_l113.py
.venv/bin/python labs/_build_l113.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l113.py
.venv/bin/python labs/_delivery_l113.py
```

The Modal volume is `l113-products`. Model evidence is namespaced by the SHA256 of the visible canonical implementation. Raw graph, partition and binary checkpoints remain reproducible local/cloud artifacts. Compact predictions, source identity, metrics and audit files are distributed. Re-running paid jobs is additional compute: read the budget before launching.

## Verification boundaries

- Dense independent operator/model/gradient oracle and original released-class checks.
- Induced edge relabeling, isolated nodes, held-out label mutation, first-validation-tie selection, layerwise inference equivalence.
- Exact recorded preparation/partition identity; named-run evaluation and checkpoint evidence separately.
- Executed notebook uses live task functions; mutation checks must reject wrong solutions.
- Browser desktop/mobile/keyboard/no-JS/print and realistic copied-Pages checks recorded separately from scientific reproduction.
- Learner PENDING_WRITTEN_DEFENSE; live Colab and deployment NOT_CHECKED.

## Recorded preparation and pilot

The full archive SHA256 is `5ea0a112edaec2141c0a2a612dd4aed58df97ff3e1ab1a0ca8238f43cbbb50a8`. The first single-stream download was stopped for poor throughput; its entire timeout allowance remains in the budget. A replacement uses16 nonoverlapping HTTP byte ranges, validates each Content-Range and length, then checks the assembled archive hash and ZIP extraction CRCs. The dataset and graph were not subsampled.

The completed preparation measured588.14s through partition save (including392.34s partitioning and serialization); subsequent final hashing/commit overhead is outside that timer. The first official-reader audit failed on an incorrect import name before loading data; it was corrected to `read_csv_graph_raw` and rerun. The successful full-data audit took97.06s and found exact feature and edge-multiset equality, including duplicates. All attempt allowances are counted.

The T4 pilot used two full label passes (17.21s and16.89s) and one complete-graph inference (12.81s). Peak PyTorch GPU allocation was2,935,920,640bytes across training/evaluation. Its selected test score is a pilot-only number, not a published-target comparison. The pilot supports an estimated970s per complete seed, including seven inference passes and initialization; hard outer cap2500s remains unchanged.

Historical operator review: archived PyG1.6.0 `sage_conv_1_6_0.py` confirms the same separate neighbor/root Linear maps, mean aggregation and normalize=False defaults. It uses torch.nn.Linear initialization. The runtime checks still execute the pinned OGB model class with the recorded modern PyG operator; no original library-build identity is claimed.

## Final selected-experiment result

All ten full-data50-epoch runs completed (500 total epochs). Independent prediction counting and OGB evaluator agree. Validation **91.691885%±0.090785pp**, gap−0.428115pp: **CLOSE**. Test **78.366032%±0.274946pp**, gap−0.603968pp: **OUTSIDE_TOLERANCE**. These are sample seed SDs, not confidence intervals. The predeclared0.5pp criterion is unchanged. This is completed execution of the selected released schedule, not successful reproduction of the published test mean.

Independent replay with the pinned OGB model class and PyG SAGEConv/torch_sparse reproduces all **24,490,290** archived class predictions exactly, with zero disagreements across ten checkpoints. CPU and GPU copied-weight operator checks, official feature/edge-multiset audit and exact checkpoint metric reconstruction pass. These checks do not identify the cause of the remaining published-score gap. Historical partition, seed sequence, sparse arithmetic and runtime differ; no test-based retuning or extra training was performed.

The separate pilot also replays all2,449,029 classes exactly. See `_replay_l113_results.json` and per-seed `replay.json`. Full-run training/evaluation times range871.5–1020.5s (data loading/setup excluded from those particular timers). Successful full-worker and pilot resource estimates are in `_budget_l113.json`; they include each worker's recorded setup time. All authorized calls have a conservative resource ceiling ofUSD8.895592, leavingUSD1.104408 of theUSD10 budget for other overhead. Actual provider charges are not fully itemized.
