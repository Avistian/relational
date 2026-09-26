# Lesson 115 — graph ML design patterns and full selected GCN reproduction

## Measured result

Ten fresh modular GCN fits on complete ogbn-arxiv, seeds 0–9, 500 epochs each. Validation 73.1471% ± 0.0878 pp; test 71.9713% ± 0.2476 pp (sample seed SD). Paper means 73.00% and 71.74%; both CLOSE under the predeclared absolute mean tolerance of 0.5 pp. SD is reported, not required to match. This is not a statistical equivalence test.

All 1,693,430 final node predictions were independently rescored and replayed through the original OGB GCN and GCNConv. Zero class mismatches; maximum current-CPU log-probability difference zero. Flat-versus-modular local outputs, gradients, BN state and one Adam update match exactly under controlled state and random masks. Original PyG comparisons separately pass forward/gradient tolerances in training and evaluation; the PyG Adam check uses eval-mode BN to avoid amplification of near-zero floating-point pre-BN bias gradients.

The selected executable experiment is COMPLETE. Historical environment/randomness and whole-paper parity remain NOT_ESTABLISHED. Architecture organization is a course synthesis, not a new published model. Link/graph examples train readout heads on tiny fixed embeddings and are SYNTHETIC_COURSE_EXAMPLES, not benchmark reproductions. Author execution does not establish learner mastery.

## Frozen protocol

| Component | Specification |
|---|---|
| Named experiment | Hu et al., OGB v6 §4.3 Table6, GCN on ogbn-arxiv |
| Paper | https://arxiv.org/html/2005.00687v6#S4.SS3 |
| Original code | snap-stanford/ogb commit 61e9784ca76edeaa6e259ba0f836099608ff0586, examples/nodeproppred/arxiv/gnn.py and logger.py; MIT license archived |
| Dataset | Full release-v1 arxiv:169343nodes,1166243directed edges,128features,40classes |
| Archive SHA256 |49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276|
| Splits | Released time IDs:90941train through2017,29799valid in2018,48603test in2019–2020 |
| Graph | Deduplicated binary-undirected adjacency plus exactly one loop per node; symmetric normalization;2484941nonzeros |
| Encoder | Identity; no additional learned feature projection |
| Processor | Two GCN layers128→256→256; each followed by BN, ReLU, dropout.5 |
| Head | Third GCN256→40, including neighbor aggregation, then log-softmax |
| Parameters |110120; same released operations and parameter shapes |
| Reset | All three convolutions reset in order, then both BN modules; explicit seeds0–9; constructor followed by reset |
| Visibility | All features and complete graph in training forward/BN; only train labels in loss. This is transductive, not a historical graph snapshot |
| Loss | Mean NLL over official training nodes |
| Optimizer | Adam lr.01, betas(.9,.999),eps1e-8,weight_decay0,amsgradFalse; no scheduler |
| Schedule |500epochs per seed, full-node forward; evaluate every epoch; no early stopping |
| Selection | First maximum validation accuracy; save and restore parameters and BN running buffers; no test-based selection |
| Metric | OGB accuracy, independently reconstructed by integer correct counts and official split denominators; mean and sample seed SD(ddof1) |
| Author runtime | Python3.12,torch2.8.0,PyG2.6.1,OGB1.3.6,NumPy2.2.6,pandas2.3.2,scikit-learn1.7.1; T4,2CPUcores,8GiB; per-run identities archived |
| Budget |USD10 aggregate; at most12workers×600seconds atUSD.00020796/s; no automatic retries; worker ceilingUSD1.497312; overhead reserveUSD8.502688 |

## Deviations and evidence boundaries

- Source-confirmed architecture and defaults do not recover historical seeds, hardware or original optimization trajectories. CUDA sparse accumulation can vary across runs; a changed score relative to L112 does not imply the refactor altered the mathematical operator.
- The course implements sparse GCN algebra visibly using native CSR. Original source uses PyG GCNConv. Independent replay gives original GCNConv a loop-free binary CSR input so it adds exactly one loop; supplying already-looped CSR would double-count loops. The dedicated regression detects that error.
- Paper reports aggregate benchmark cells; the released code is pinned as executable protocol. First-best selected state is saved explicitly for reproducibility. The original logger selects by validation even though its sample script does not persist every best checkpoint.
- The identity encoder, two-block processor and propagating head are our software decomposition. Neither tensor dimensions nor close scores alone prove preservation. Controlled forward, gradient and update checks support this refactor specifically.
- Node/link/graph task exercises reuse live student functions. The toy node model trains message passing; link and graph exercises train heads on fixed synthetic embeddings. They fit and score the same tiny examples and make no held-out-generalization claim.
- Temporal and heterogeneous designs are reasoned exercises. No temporal, heterogeneous, link or graph benchmark is claimed executed in L115.
- Default solution notebook executes only teaching examples. Full ten-seed author training is separate evidence. Live Colab and deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.

## Exact commands from repository root

```bash
# Local implementation and official-reader checks
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l115.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l115.py
.venv/bin/python labs/_upstream_l115.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_sparse_input_l115.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_audit_l115.py

# Full named experiment locally (new output directory; GPU recommended)
OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l115.py --preset paper --device cuda --data labs/data/l112 --output /tmp/l115-fresh-paper
# Teaching smoke on full data: 2 epochs, not paper reproduction
OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l115.py --preset smoke --device cpu --data labs/data/l112 --output /tmp/l115-fresh-smoke

# Bounded author cloud execution; do not rerun paid jobs merely to inspect results
.venv/bin/python -m modal run modal/l115_repro.py --mode pilot
# After inspecting pilot source parity, runtime and aggregate budget, mark
# pilot_approved_for_full true in labs/_budget_l115.json for this source hash.
.venv/bin/python -m modal run modal/l115_repro.py --mode paper
# The shipped evidence is already complete: volume output paths must not be reused.

# Download existing evidence without new training, then independently validate
.venv/bin/python labs/_collect_l115.py
.venv/bin/python labs/_analyze_l115.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_replay_l115.py

# Build and verify teaching artifacts
.venv/bin/python labs/_figures_l115.py
.venv/bin/python labs/_build_l115.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l115.py
.venv/bin/python labs/_mutation_l115.py
.venv/bin/python labs/_delivery_l115.py
.venv/bin/python labs/_record_l115.py
```

The pinned cloud environment is `labs/requirements-l115-runtime.txt`. To reproduce it locally, create a separate Python3.12 environment and install that file. The notebook installs only missing packages and prints its actual versions; it does not claim that an existing runtime is identical to the author runtime.

`_run_l115.py` defaults to the small smoke schedule; `--preset paper` means all ten seeds×500epochs unless an explicit `--seeds` subset is supplied. A subset is incomplete selected-experiment evidence, never the complete ten-run result. The standalone notebook full-run switch uses the same visible data reader, model and trainer and defaults OFF. Local/notebook execution has no automatic spending cap; the documented Modal author run is bounded.

## Evidence map and cost

- `evidence/l115/summary.json`: every seed, selected epoch, independent metrics and aggregate result.
- `evidence/l115/paper/seed-*/`: complete500epoch history, selected predictions, labels, split IDs, hashes, runtime identity and worker timing.
- `results/l115/paper/seed-*/checkpoint.pt`: local selected weights, excluded from site/git; downloadable from Modal volume `l115-ogb-evidence` using `_collect_l115.py` with account access. Public compact predictions suffice to rescore but do not substitute for weights in neural replay.
- `_sources_l115.json` and `sources/l115/`: pinned upstream source, license and hashes.
- `_check_l115_results.json`, `_source_check_l115_results.json`, `_sparse_input_l115_results.json`, `_audit_l115_results.json`, `_replay_l115_results.json`: distinct behavior, source, adapter, dataset and full replay checks.
- `_execution_l115_results.json`, `_mutation_l115_results.json`, `_delivery_l115_results.json`: notebook, live-task and local frontend evidence.
- `_budget_l115.json`: pilot gate and conservative resource allocation. Completed pilot+ten workers estimate **USD0.207338**, not an invoice; startup/build/storage overhead unitemized. No retries used. Local validation incurred no cloud worker charge.

Pilot app: https://modal.com/apps/pszar92/main/ap-Rvl9hVh6PMVEVjVPUWrwyI
Full app: https://modal.com/apps/pszar92/main/ap-l8N6hcD2nL4nHcxzJMVrqq

Delivery status is recorded by the final checker; deployment and live Colab remain separate NOT_CHECKED fields. No learner completion record is written for prepared material.

## Completed local delivery checks

PASS: 17 executed standalone solution code cells; 3 live tasks and 6 rejected mutants; 15 canonical definitions aligned; four portable figures; byte-identical rebuilds; desktop1200px/mobile375px with20widget states; keyboard/reset, cutoff controls, no-JS/print and47local links in the exact copied Pages workflow. Source model/logger/license bytes freshly match the pinned upstream commit. Full provenance: `_provenance_l115_results.json`. Browser checks do not establish live Colab or deployment.
