# L099 · R-GCN versus HGT: reproduction contract

This is an analysis unit. The complete declared ACM experiment is a new course comparison, not a published table. The separately executed AIFB lane and unrun HGT CS lane retain their own protocols. Full-paper parity: **NOT_ESTABLISHED**. Learner status: **PENDING_WRITTEN_DEFENSE**.

## Frozen ACM course experiment

Source graph: DGL `ACM.mat`, SHA256 `0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d`. Source and license archived in `sources/l099/`, immutable upstream revision and file hashes in `_sources_l099.json`.

- Follow the DGL ACM raw conference filter `[0,1,9,10,13]` with labels `[0,1,2,2,1]`. All 4,025 selected papers retained. Retain all connected authors (7,167) and subjects (60), compact their IDs, save original IDs. This is not ACM3025 or its split.
- Four directed edge stores: paper→author and reverse (13,407 each); paper→subject and reverse (4,025 each). Exclude all conference edges and conference identity from inputs. Bag-of-words paper features have 1,903 columns, row-normalized; auxiliary features are constant ones. No label propagation preprocessing or target encoding.
- Static transductive graph. All selected node features/edges are visible to all arms. No claim of future-information availability or inductive evaluation.
- Class-stratified split from NumPy RandomState(99), ascending class order and sorted IDs after partition. Per-class floor(.2*n) train, floor(.3*n)-floor(.2*n) validation, remainder test:804/403/2818. Save all IDs and hashes.
- Arms: R-GCN, HGT, uniform HGT, feature-only MLP. Same paper adapter and classifier shapes, width32, two hidden layers, dropout0. HGT heads4. HGT/ablation common initial tensors are identical per seed. No sampler; full graph for every update.
- Six fits per arm: rates .003/.01 × seeds0/1/2. Adam with L2 weight decay .001 (not AdamW), all60 epochs, train-only CE. Every epoch receives a validation CE. Earliest strict minimum selects checkpoint; mean of these validation minima across seeds selects one LR per arm, ties smaller LR. Freeze before test scoring.
- Score selected checkpoints once on the full fixed test split: accuracy and macro F1. Report per-seed predictions, paired differences and descriptive sample SD. Seeds on one split are not independent datasets.
- Equal update/search budget, not equal parameter count/FLOPs/runtime. Allocated versus autograd-active parameter counts and wall time are recorded. Active counts operate at parameter-tensor granularity; a tensor can contain zero-gradient entries. Runtime includes validation and one final training-gradient audit, but not checkpoint serialization. Runs are sequential; this is not a dedicated performance benchmark.

The LR budget was frozen before fits. Before test scoring, common initialization for the attention ablation was tightened; nine complete preliminary validation-only fits and part of another were discarded and all24 fits restarted. No preliminary test results were produced. All delivered results use the corrected code and full schedule.

## Architecture and source parity

R-GCN retains relation-wise mean → relation sum → self transform → ReLU. Unlike the AIFB release, this course wrapper has learned feature adapters, biased self transforms and a classification head. It is not a paper-protocol AIFB run.

HGT matches the modern pyHGT operator pinned by L093: typed Q/K/V and output transforms, relation attention/message matrices and priors, all-incoming-edge softmax, GELU, gated residual and LayerNorm. RTE is disabled because this ACM experiment has no temporal task. No explicit self edges are added; the residual provides the self path. The independent output/gradient check uses the archived L093 modern-release port with identical weights. That port's original-source checks live in `_verify_l093_results.json`; this check is not an independent historical backend execution.

Uniform HGT removes Q/K, relation attention and priors, retaining V/message/output/skip/norm and initializing every common tensor from the corresponding HGT initialization. Uniform weights divide by total incoming degree, not relation degree. Parameter count and optimization change as part of this intervention.

MLP uses the paper adapter, two width32 affine/ReLU layers, and classifier. No unused auxiliary adapters are allocated. This modest fixed-budget baseline is not an optimized tabular benchmark.

## Named published-experiment tracks

### AIFB — complete fresh replay executed

Target: Schlichtkrull et al. Table2 AIFB R-GCN mean95.83%, ten trainings. The full port from L091 is rerun by `_paper_l099.py --target aifb`: original140/36 label split, full graph/support construction, featureless input, width16,50 updates, no dropout/L2/basis compression, release Adam equation, ten declared seeds0–9. Result:95.833333% mean,1.464017pp sample SD. Raw fresh traces/predictions and implementation hash: `_paper_l099_aifb_results.json`.

Exact historical identity remains **INCOMPARABLE**: PyTorch replaces Keras/Theano; original RNG/node ordering unavailable; mirrored graph historical byte identity not established. Full historical tuning and other paper datasets/tasks remain **NOT_RUN**. A close or equal score is not full-paper reproduction. See `l091-reproduction.md`, `_sources_l091.json` and archived source/license.

### HGT CS — executable full-setting port, NOT_RUN

Target: Hu et al. Table2, CS Paper–Field L2, HGT(+Heter,+RTE), NDCG.403±.041 / MRR.439±.078 over five trainings. The existing hash-pinned CS archive is about8.1GiB. Full config: width256,3 layers,8 heads,200 epochs×32 batches×2 updates,256 seed papers, sampling depth6/width128,10 test batches, seeds0–4, validation KL selection. `_paper_l099.py --target hgt-cs` invokes the complete L093 port on that exact dataset, not ACM/NN. `--with-rgcn` additionally runs the course R-GCN comparator, which is not an exact reconstruction of the paper's baseline.

L093 recorded MemoryError at a10GiB virtual-address guard and GPU execution rejected pending a payment method. These are referenced prior resource findings, **not re-tested capacity claims in L099**. No full CS training was attempted in L099. Current machine has about15GiB total RAM; no claim that increasing the guard would be sufficient. Modern-release operators, sampling/RNG and paper/release protocol discrepancies persist even on a larger machine. See `l093-reproduction.md`, `_sources_l093.json`, `_cs_resource_l093_results.json` and archived publication-era source for the complete deviation ledger.

## Execution and delivery

```bash
# From relational/; install requirements-l099-runtime.txt in a compatible CPU environment.
.venv/bin/python labs/_verify_l099.py
.venv/bin/python labs/_run_l099.py
.venv/bin/python labs/_audit_l099.py
.venv/bin/python labs/_paper_l099.py --target aifb
.venv/bin/python labs/_figures_l099.py
.venv/bin/python labs/_build_l099.py
.venv/bin/python labs/_execute_l099.py
.venv/bin/python labs/_build_l099.py
.venv/bin/python labs/_delivery_l099.py
# Requires full verified CS bytes and a machine with sufficient memory:
.venv/bin/python labs/_paper_l099.py --target hgt-cs --device cuda
```

The CPU requirement file is for the course experiment. AIFB additionally uses the L091 pins; CS uses the L093 dependencies and its own hardware-compatible torch build. Do not install a CPU-only wheel and expect the CUDA command to work. The launchers reuse visible repository modules; the course notebooks embed all of their own data/model/trainer code.

Author evidence: `_experiment_l099_results.json` (24 fits), `_experiment_l099_results-checkpoints/` (24 selected-per-fit checkpoints), `_audit_l099_results.json` (all24 restored; independent sklearn metrics), `_verify_l099_results.json` (hand/dense/gradient and label-isolation checks), `_paper_l099_aifb_results.json` (fresh named replay). The execution harness independently downloads data with bounded curl into a fresh working directory after the first urllib attempt stalled; the standalone notebook verifies those bytes and recomputes24 fits; `_execution_l099_results.json` records exact numeric agreement excluding timings/serialization hashes. It uses the same author kernel; clean dependency installation is **NOT_CHECKED**. Notebook replay checkpoints are temporary and not delivered; the author checkpoints are retained and hashed.

Delivery is recorded separately in `_delivery_l099_results.json`: browser widths, interactive states, print, deterministic rebuild, portable figure and copied-Pages links. Live Colab and deployment remain **NOT_CHECKED**. No publication requested. No mastery inferred from teacher execution.
