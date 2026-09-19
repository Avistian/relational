# L090 · GNN checkpoint reproduction contract

## Named target and completed scope

Kipf & Welling (ICLR 2017), Table 2, GCN, fixed Cora split: reported 81.5% mean accuracy over 100 initializations. L090 freshly executed all 100 initializations of the complete L082 PyTorch release-protocol port. Mean: **81.401%**, sample SD **0.658 percentage points**, SE **0.0658 percentage points**. The ±1 percentage point course tolerance was declared before execution. Verdict: CLOSE. This covers one published experiment, not the whole paper.

A separate inductive Cora extension completed seeds 0,1,2 for 100 epochs each: **72.467% ±1.124 percentage points** sample SD. It is not a GraphSAGE table reproduction and not directly comparable with the transductive GCN score. All raw validation curves, seed scores and inductive test predictions are retained.

## Reproduce from the repository root

Python 3.12 CPU is sufficient. The author environment is recorded in `requirements-l090-observed.txt`; it includes a newer PyTorch build. For the portable pinned runtime use:

```bash
python3.12 -m venv /tmp/l090-env
/tmp/l090-env/bin/python -m pip install -r labs/requirements-l090-runtime.txt
/tmp/l090-env/bin/python labs/_verify_l090.py
/tmp/l090-env/bin/python labs/_run_l090.py --lane paper --preset paper --output _portable_paper_l090_results.json
/tmp/l090-env/bin/python labs/_run_l090.py --lane inductive --preset paper --output _portable_inductive_l090_results.json
```

The runtime uses torch 2.8.0, numpy 2.2.6, scipy 1.15.3; pandas2.3.2 and scikit-learn1.7.1 satisfy shared `relkit` package initialization. The inline notebook does not need the latter two packages. Results from a different environment are new evidence, not assumed bitwise identical. Presets: `smoke` = one GCN initialization / one 3-epoch extension run; `closer` = ten GCN initializations / three full extension runs; `paper` = 100 GCN initializations / three full extension runs. Reduced seed counts are INCOMPLETE for the published target. No caching or resume shortcuts: each command starts a fresh run. Explicit `--output` prevents overwriting author evidence during environment comparisons.

Build and validation (authoring environment with notebook/plot/browser packages):

```bash
python labs/_figures_l090.py
python labs/_build_l090.py
python labs/_audit_l090.py
python labs/_execute_l090.py
python labs/_delivery_l090.py
```

The solution execution starts in an empty working directory, freshly downloads hash-verified data and runs **all 100 GCN seeds plus all three extension seeds** inline. It compares all validation traces and scores against the canonical command outputs. The student notebook keeps five tasks blank and leaves the full-run switch off until the learner is ready. The teacher solution enables the full experiment. Colab installs the portable torch/numpy/scipy pins; live Colab is not tested. Notebook execution uses the current student's global functions. No `relkit` model import supplies an alternative finished implementation.

## Protocol audit

| Component | GCN contract | Source / evidence |
|---|---|---|
| Data identity | Release Cora: 2,708 nodes, 1,433 features, seven known classes | `_sources_l078.json`: SHA-256 for all eight raw files |
| Source identity | tkipf/gcn revision `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81` | Seven source/license files also hash-pinned; not asserted as exact submission revision |
| Split | 140 train, next 500 validation, released 1,000 test indices | Restore shuffled test rows before use; audit disjointness |
| Features | Per-row sum normalization; zero rows retained | `utils.py` |
| Graph | Binary undirected edges; one self-loop per node; symmetric augmented-degree scaling | `utils.py`; independent hand/dense oracle |
| Architecture | Two bias-free GCN layers, width 16, ReLU, seven output logits | `models.py`, `layers.py` |
| Initialization | Glorot uniform; no pretrained weights | `inits.py` |
| Dropout | 0.5 on nonzero inputs and hidden activations; off during evaluation | `layers.py` |
| Objective | Train-only mean cross-entropy plus 5e-4 × half squared norm of first-layer weights | `models.py`, `metrics.py`; label/gradient checks |
| Optimizer | Adam .01, betas .9/.999, epsilon 1e-8; explicit L2 | Modern port of release; framework arithmetic difference declared |
| Schedule | Full batch, at most 200 epochs | `train.py` |
| Selection | After zero-based epoch 10, stop when current regularized validation loss exceeds preceding-ten mean | Follow released code; audit every trace |
| Checkpoint | Last weights, no best restoration | `train.py`; paper prose differs |
| Test freeze | Access test targets after stopping | `train_cora` |
| Seeds | Locally declared 0–99, no seed selection | Original 100 identities unavailable |
| Aggregation | Mean, sample SD with ddof=1, SE=SD/√100 | Recomputed from raw scores |
| Tolerance | ±0.01 absolute accuracy around .815 | Course convention, not historical statistical test |

## Inductive extension contract

Remove all validation/test nodes and their edges before training. The 1,208-node training context includes unlabeled nodes; only 140 labels supervise updates. Validation context adds validation nodes only. Final test context adds test nodes only, excluding validation nodes. Features are row-normalized independently; the class vocabulary is the known seven Cora classes. A root is one supervised target; context nodes provide messages without receiving extra supervision.

Two self/neighbor concatenation transforms, width 32, ReLU and L2-normalized hidden representations; output layer emits seven logits. Standard PyTorch Linear initialization and biases, no dropout. Uniform neighbor draws with replacement, fanouts [10,10], batch size64; empty-neighbor masks produce zero. Full-neighbor validation and test inference. Adam .01, weight decay .0005 over all parameters, 100 epochs, best validation-loss checkpoint, seeds0–2. Test labels do not select epochs. This is a complete teaching extension, not Hamilton's released supervised benchmark recipe.

Checks include excluded-node feature intervention, degree-one sampled/full agreement, isolated-node handling, masked mean gradients and GCN hand arithmetic. Same graph dataset does not make these two protocols interchangeable; there is no cross-model superiority claim.

## Provenance, attribution and evidence boundaries

Canonical GCN: `relkit/gcn_l082.py`, shared unchanged with L082. Canonical extension: `relkit/checkpoint_l090.py`. Runner results contain SHA-256 hashes of both implementations, the runner and the raw source manifest, plus package/platform/thread details. `_audit_l090_results.json` verifies current source identity, raw data/source hashes, split and coverage. The standalone notebook embeds the data manifest and full readable implementation. GCN code is adapted from the MIT-licensed [tkipf/gcn](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81); retained license: `sources/l078/LICENCE`.

- Complete modern-framework reconstruction: EXECUTED, 100 runs; course tolerance CLOSE.
- Exact historical training parity: INCOMPARABLE (framework/RNG/Adam, unavailable historical seeds and exact experiment revision).
- Original TensorFlow1 replay, other Table2 datasets/baselines, Hamilton benchmarks: NOT_RUN.
- Portable environment results, standalone notebook execution and browser/Pages staging are recorded separately; consult their reports rather than inferring them from training success.
- Live Colab and deployed Pages: NOT_CHECKED. A local browser and copied Pages build do not demonstrate remote deployment.
- Learner mastery: PENDING. Prepared/teacher-executed materials never establish learner completion.

## Portable runtime replay

A newly created Python3.12 virtual environment with system site packages disabled installed `requirements-l090-runtime.txt` successfully. All100 GCN runs completed under torch2.8.0+cpu, numpy2.2.6 and scipy1.15.3. All scores and validation-loss traces exactly matched the author environment in this CPU check:81.401% mean. This verifies these two observed CPU environments; it does not imply platform-independent bitwise identity or live Colab execution. Portable raw evidence: `_portable_paper_l090_results.json`; full dependency freeze: `requirements-l090-lock.txt`. `_clean_environment_l090_results.json` records the independent extension replay and behavioral checks.
