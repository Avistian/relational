# Lesson 116 — debugging GNN training and the full selected GCN experiment

## Measured result and boundary

Ten fresh complete ogbn-arxiv GCN fits, seeds 0–9, 500 epochs each. Validation accuracy **73.1283% ± 0.0860 pp**; test **72.0200% ± 0.3056 pp**, where ± is sample seed SD. Published means are 73.00% and 71.74%. Both are CLOSE under the **predeclared absolute mean tolerance of 0.5 percentage points**; this is not a statistical equivalence test.

All **1,693,430 selected node predictions** were independently rescored and replayed through the original OGB GCN and PyG GCNConv: zero class mismatches; maximum current-CPU log-probability difference zero. Source checks separately cover forward/gradient behavior and an Adam update. The repaired training step exactly matches an inline released-order step in a controlled three-update test, including BN buffers and matched dropout randomness.

**Selected executable experiment COMPLETE. Historical identity and whole-paper parity NOT_ESTABLISHED.** This debugging unit introduces no new architecture and claims no reproduction of Li et al.'s co-training/self-training experiments. The 80-epoch fault/repair pair and local scalar/sampling/label probes are course interventions, not paper experiments. Live Colab and deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE.

## Frozen protocol audit

| Component | Frozen specification and evidence |
|---|---|
| Named published target | Hu et al., OGB v6 §4.3 Table 6, GCN on ogbn-arxiv; https://arxiv.org/html/2005.00687v6#S4.SS3 |
| Executable source | snap-stanford/ogb commit `61e9784ca76edeaa6e259ba0f836099608ff0586`, `examples/nodeproppred/arxiv/gnn.py` and `logger.py`; source bytes freshly verified; MIT license archived |
| Data | Full release-v1 archive, 169343 nodes, 1166243 directed edges, 128 features, 40 classes |
| Archive SHA256 | `49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276` |
| Independent loader | Official OGB library-agnostic reader matches features, edges, labels and every split ID exactly; array hashes in `_audit_l116_results.json` |
| Split | Official publication-year IDs: 90941 train through 2017; 29799 validation in 2018; 48603 test since 2019 |
| Graph | Binary undirected deduplication; one loop per node; symmetric normalization; 2484941 nonzeros |
| Model | Identity encoder; two GCN128→256→256 blocks with BN/ReLU/dropout .5; third GCN256→40 and log-softmax; 110120 parameters |
| Algebra | S(HWᵀ)+b, where S=D⁻¹ᐟ²AD⁻¹ᐟ² after loop insertion. Bias is added after aggregation |
| Visibility | All graph/features participate in transductive forward and BN. Only train labels participate in loss; not a historical snapshot protocol |
| Initialization | Construct model, then reset all convolutions in release order, then BN. Explicit fresh seeds 0–9; historical author seeds were not recovered |
| Loss | Mean train-ID NLL; no held-out labels, no class weights |
| Optimizer | Adam lr .01, betas (.9,.999), eps 1e-8, weight decay 0, amsgrad False; no clipping or scheduler |
| Training | 500 epochs per seed, full-batch; zero gradients before backward; optimizer step afterward; no early stopping |
| Selection | First maximum validation accuracy; cloned weights and BN buffers restored. Test does not choose epochs, repairs or hyperparameters |
| Evaluation | Eval mode, no gradient recording; official OGB accuracy; independent integer-count scoring |
| Aggregation | Mean and sample seed SD, ddof=1, over ten complete fits; tolerance compares means only |
| Instrumentation | Gradient and parameter-change L2 norms recorded every epoch; cloned observations do not alter updates. Finite-loss/gradient guard aborts on invalid arithmetic |
| Author runtime | Python 3.12, torch 2.8.0, PyG 2.6.1, OGB 1.3.6, NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1; T4, two physical CPU cores, 8 GiB |
| Source identity | Canonical source SHA256 `1bac8f801a77766996049de2f18fcb2bb1f4a59a7ad2625007186140ffdb4534`; source-pinned cloud volume and per-run identities |

## Fault/repair experiment

Two fresh full-data 80-epoch runs use seed 101 and the same source, graph, architecture, labels, optimizer and dropout schedule. Broken arm omits only `optimizer.step()`. Initial loss **4.0509619713** and initial gradient norm **2.4799170494** match exactly. Every broken update norm is zero; every repaired update norm is positive. Epoch-80 losses are **4.0573987961** versus **0.9480451345**.

The train loss is recorded before that epoch's update; validation/test metrics are measured after it. Missing steps do not freeze BN buffers. Consequently a changing validation trace cannot establish weight movement. The repair is selected by the failing behavior and source contract, never by test accuracy. These paired curves are not an uncertainty estimate or a ten-seed ablation claim.

Small notebook probes separately execute a two-component toy GCN, a scalar derivative chain, fixed three-node propagation, seed/global label mapping and held-out-label perturbations. A real PyG NeighborLoader integration check confirms seed ordering. It uses the installed sampler backend; the notebook's hand-built fixture needs no PyG compiled extension. Seven deliberate mutations must be rejected by the three live notebook checks.

## Deviations and unrun scope

- Native PyTorch CSR implements the released GCN algebra visibly; the release uses PyG GCNConv. Original-model replay and controlled source checks establish current-runtime agreement, not historical numerical trajectories.
- Original-model replay receives a loop-free binary CSR; GCNConv adds its own loops. A regression probe rejects the double-loop adapter error.
- Adam's source comparison uses eval-mode BN to avoid amplifying analytically zero pre-BN bias gradients from float32 noise. Forward/gradient comparisons cover both train/eval modes. A separate canonical-versus-inline three-update check covers training-mode BN and Adam exactly.
- The pinned release supplies executable defaults where the publication alone is insufficient. It computes test scores at each epoch; this implementation retains them as reporting, without test-based selection. Best state persistence is added explicitly.
- Canonical source, initializations, current libraries and hardware are recorded. Original authors' exact seeds, runtime environment and trajectories remain unknown. CUDA sparse reductions can vary.
- Instrumentation records norms but changes no gradient values. Gradient clipping, altered depth and residual paths are discussed as hypotheses; they are not silently included in the benchmark.
- No Cora, products, other OGB dataset, novel model, or whole-paper suite is claimed executed here. Author checks do not establish student mastery.
- No silent resume: completed run directories are rejected. Local/cloud repeated launches require deliberate fresh output/reservation handling. Inspect existing evidence before running paid work again.

## Aggregate budget and actual resource estimate

Authorized cap **USD10 total**, covering all seeds, pilots, diagnostics, retries and validation. Verified Modal rate: T4 .000164/s + two CPU cores × .0000131/s + 8 GiB × .00000222/s = **USD.00020796/s**. Source: https://modal.com/pricing (checked 2026-09-26).

Maximum twenty 600-second workers = **USD2.49552** resource ceiling, leaving **USD7.50448** for overhead/storage. The local launcher locks and reserves workers before dispatch and rejects duplicate mode launches. Automatic retries are disabled. A pilot source/runtime gate precedes full dispatch; it projected 149.04 seconds per 500-epoch fit including extra margin. Any revised run requires checking remaining aggregate allocation. Never shorten a paper schedule to manufacture completion.

Used: one 10-epoch pilot, ten 500-epoch fits, two 80-epoch diagnostics, **13 workers**, no retries. Recorded successful worker resource estimate **USD0.2209044354**. This is not an invoice: startup/image-build/storage overhead is unitemized and reserved separately. Local CPU validation incurs no cloud-worker charge. See `_budget_l116.json` and every `completed.json`.

## Exact commands from repository root

```bash
# Red-capable probes followed by repaired behavior and source checks.
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l116.py --red
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l116.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_trainer_check_l116.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l116.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_sparse_input_l116.py
.venv/bin/python labs/_upstream_l116.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_audit_l116.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_mutation_l116.py

# Full named experiment, fresh directory, GPU recommended. No shortcut preset.
OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l116.py --preset paper --device cuda --data labs/data/l112 --output /tmp/l116-fresh-paper
# Smoke is two epochs; it is never paper-reproduction evidence.
OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l116.py --preset smoke --device cpu --data labs/data/l112 --output /tmp/l116-fresh-smoke

# Author cloud commands ALREADY EXECUTED; do not relaunch to inspect evidence.
.venv/bin/python -m modal run modal/l116_repro.py --mode pilot
# Inspect pilot parity/time; budget gate must match current canonical source hash.
.venv/bin/python -m modal run modal/l116_repro.py --mode paper
.venv/bin/python -m modal run modal/l116_repro.py --mode diagnostics

# Read existing cloud files without starting training; independent metrics/replay.
.venv/bin/python labs/_collect_l116.py
.venv/bin/python labs/_analyze_l116.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_replay_l116.py

# Rebuild and execute the standalone notebook, then verify real sampler/figures/UI.
.venv/bin/python labs/_figures_l116.py
.venv/bin/python labs/_build_l116.py
.venv/bin/python labs/_execute_l116.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_diagnostics_check_l116.py
.venv/bin/python labs/_delivery_l116.py
node labs/_check_pedagogy.js
.venv/bin/python labs/_record_l116.py
```

Standalone command `labs/_teaching_l116.py` also runs the local probes and writes `l116-task-report.json` in the working directory. The notebook executor instead runs in a temporary empty directory and archives the report as `_teaching_l116_results.json`.
