# Lesson 112 — complete selected OGB GCN reproduction

## Frozen scientific question

Reproduce Hu et al., [OGB v6 §4.3, Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3), GCN on ogbn-arxiv. Ten **fresh** full-data runs, seeds 0–9, 500 epochs each; select the first maximum validation accuracy. Published test 71.74 ± 0.29%, validation 73.00 ± 0.17%. Predeclared mean tolerance: absolute difference <= **0.5 percentage point**, for each mean. This descriptive rule is not an equivalence test. Variance is reported, not forced to match.

[Results](evidence/l112/summary.json), [independent full checkpoint replay](_replay_l112_results.json), [data audit](_audit_l112_results.json), [budget](_budget_l112.json). Historical bitwise identity and full-paper reproduction remain NOT_ESTABLISHED. Learner status PENDING_WRITTEN_DEFENSE; live Colab and deployment NOT_CHECKED.

## Protocol ledger

| Component | Frozen definition and source |
|---|---|
| Source | snap-stanford/ogb commit `61e9784ca76edeaa6e259ba0f836099608ff0586`, unmodified `gnn.py`, `logger.py`, README, metadata and MIT license under `sources/l112/` |
| Dataset | OGB arxiv release v1, HTTPS mirror of the URL in released master.csv; 169,343 nodes, 1,166,243 directed edges, 128 float32 features, 40 classes |
| Archive SHA256 | `49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276` |
| Official splits | 90,941 train (1971–2017), 29,799 validation (2018), 48,603 test (2019–2020); exact released IDs |
| Visibility | Full graph and all feature rows for training/inference, including batch-normalization statistics; train labels only in NLL |
| Graph processing | Binary undirected union, self-loops once, symmetric degree normalization; 2,484,941 nonzero normalized entries |
| Model | Three GCN layers, 128→256→256→40, BN/ReLU/dropout 0.5 after first two layers, log-softmax output; 110,120 trainable parameters |
| Initialization | Xavier-uniform weights, zero biases, default BatchNorm parameters; explicit seeds 0–9, model reset once per fresh run |
| Optimizer | Adam lr 0.01, betas (0.9,0.999), eps 1e-8, weight_decay 0, default amsgrad=False; no scheduler |
| Schedule | 500 full-graph epochs, one optimizer step each, evaluate every epoch; no early stopping or retuning |
| Selection | First validation maximum; save all weights and BN running buffers; replay selected state |
| Test access | Scores logged each epoch, as in release; no test-based selection or hyperparameter changes |
| Report | Official OGB accuracy; arithmetic mean and sample seed SD (ddof=1) across all ten seeds |
| Runtime | Python 3.12, torch 2.8.0, torch-geometric 2.6.1, OGB 1.3.6, NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1; T4. Exact runtime identity in each seed directory |
| Budget | <=USD10 total; two pilot attempts bounded at 600 s each plus ten full workers bounded at 600 s each, no automatic retries. Conservative resource ceiling USD1.497312; reserve covers startup/build/storage and other overhead |

## Deviations and limits

1. **Sparse backend:** visible native PyTorch CSR instead of the released `torch_sparse.SparseTensor`. The small forward/gradient check uses the original GCN with edge-index aggregation. Full replay uses the original GCN/GCNConv with native loop-free binary CSR input, independently normalized by PyG. These test computation, not the historical sparse library build.
2. **Randomness:** explicit separately seeded processes replace the release's continuous unseeded reset loop. Initialization distributions match, but random-number consumption and trajectories are not identical. No original seed list was recovered.
3. **Current runtime and hardware:** original leaderboard hardware is RTX 2080; author runs use T4 and pinned modern libraries. CPU notebook/replay use the recorded workspace runtime, distinct from GPU training.
4. **Data reader:** visible gzip-CSV reader replaces OGB's wrapper. The independent library-agnostic OGB loader returns exactly matching features, labels, edge indices, and split indices. No feature standardization is introduced.
5. **Selected state:** release logger selects an epoch's scores without storing its predictor. We additionally save weights and all BN buffers, archive predictions, restore the chosen state, and verify its metrics. Checkpoints are for inference replay; optimizer/RNG-exact training resume is not implemented. Existing completed output directories are rejected.
6. **Scope:** one published selected baseline; original hyperparameter tuning, other models/datasets, OGB link prediction, and strict historical forecasting are NOT_RUN in L112. This is not reproduction of the entire OGB paper.
7. **Numerical checks:** float32 output/gradient comparison has explicit tolerances. Adam comparison uses evaluation-mode BN: training-mode pre-BN bias gradients are analytically zero, and floating-point noise around zero can be amplified by Adam. Do not infer bitwise equality of complete optimization trajectories from this test.

The initial replay adapter supplied a CSR matrix that already contained self-loops. PyG's native sparse path adds loops (unlike `SparseTensor.fill_diag`, which replaces them), producing double self-loops. A three-node regression detected this. The corrected adapter supplies a loop-free binary graph; training data and all ten trained checkpoints are unchanged. See `_sparse_input_l112_results.json`.

## Exact commands

From repository root. Do not replace an existing unrelated environment:

```bash
python3 -m venv /tmp/l112-runtime
/tmp/l112-runtime/bin/pip install -r labs/requirements-l112-runtime.txt
OMP_NUM_THREADS=1 /tmp/l112-runtime/bin/python labs/_check_l112.py
OMP_NUM_THREADS=1 /tmp/l112-runtime/bin/python labs/_source_check_l112.py
OMP_NUM_THREADS=1 /tmp/l112-runtime/bin/python labs/_audit_l112.py
# Full graph, two epochs: teaching evidence only.
OMP_NUM_THREADS=1 /tmp/l112-runtime/bin/python labs/_run_l112.py --preset smoke --output labs/results/l112/new-smoke
# Complete selected experiment: all ten seeds, 500 epochs each.
OMP_NUM_THREADS=1 /tmp/l112-runtime/bin/python labs/_run_l112.py --preset paper --device cuda --output labs/results/l112/new-paper
```

The teaching notebook contains the model, data loader, trainer and official evaluator call inline. Its default is two complete-graph CPU epochs. `RUN_FULL_REPRODUCTION=True` enables the full ten-seed schedule with live student functions; use an external runtime/spend cap. Python package installation can require a kernel restart.

Author cloud execution, using configured Modal credentials:

```bash
.venv/bin/modal run --detach modal/l112_repro.py --mode pilot
# Inspect the pilot and update the budget gate only after confirming all runs fit.
.venv/bin/modal run --detach modal/l112_repro.py --mode paper
# Read existing evidence; does not submit training:
.venv/bin/python labs/_collect_l112.py
.venv/bin/python labs/_analyze_l112.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_replay_l112.py
.venv/bin/python labs/_figures_l112.py
.venv/bin/python labs/_build_l112.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l112.py
.venv/bin/python labs/_mutation_l112.py
.venv/bin/python labs/_delivery_l112.py
```

Cloud evidence lives under the canonical implementation SHA256 in volume `l112-ogb-evidence`. Repeating an already-completed cloud run deliberately fails rather than overwriting its evidence. For a genuinely fresh rerun, use a fresh local output directory or an explicitly distinct cloud namespace, and account for its additional cost. The evidence collector downloads compact predictions/traces plus ~0.5 MiB checkpoints. Local raw data and checkpoints are ignored by Git; predictions, hashes, histories and metrics are distributed.

## Evidence ladder

- Hand-worked chain and independent dense oracle: normalization and gradient mechanics.
- Label mutation and checkpoint-selection fixtures: eligibility and tie behavior.
- Original source comparison: current-runtime forward/gradient agreement and eval-BN Adam step.
- Official loader equality: complete dataset/split identity, not just matching counts.
- Ten full fits: complete selected schedule; numerical verdict determined from all ten, without dropping seeds.
- Independent metric reconstruction: count correct classes from saved predictions, recover first maximum from 500-epoch histories, verify identities and hashes.
- All-ten original-model checkpoint replay: reproduce all 1,693,430 archived class predictions; numerical log-probability comparison uses stated tolerances.
- Executed solution, deliberately broken task rejection, desktop/mobile/keyboard/print/no-JS and copied Pages checks: delivery evidence, separate from scientific reproduction.

The first GPU pilot stopped before training because the checker paired parameters by registration order; original PyG registers bias and weight differently. It was fixed to compare corresponding names, validated locally, and rerun. Both pilot timeout allowances are included in the conservative budget. Successful-call resource costs are estimates from measured runtime, not an invoice. No spend is hidden by claiming that the failed attempt was free.
