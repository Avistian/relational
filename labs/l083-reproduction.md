# L083 — GraphSAGE reproduction contract

Named published target: Hamilton, Ying & Leskovec (NeurIPS 2017), **Table 1, supervised GraphSAGE-mean on PPI, micro-F1 0.598**. Scope: this model/dataset/setting only. Full original data and ten-epoch training are executed locally through a visible PyTorch port. Original TensorFlow training and exact historical table parity remain NOT_RUN / INCOMPARABLE respectively.

## Exact commands

From the relational repository root:

```bash
.venv/bin/python labs/_verify_l083.py
.venv/bin/python labs/_run_l083.py --preset paper --seeds 3
.venv/bin/python labs/_figures_l083.py
.venv/bin/python labs/_build_l083.py
.venv/bin/python labs/_execute_l083.py
.venv/bin/python labs/_browser_l083.py
.venv/bin/python labs/_delivery_l083.py
```

`--preset smoke --seeds 1` executes the complete dataset for one epoch at width32, one rate. It is a teaching diagnostic, not a result reproduction. Paper preset uses256-dimensional states and full schedule. There is no resume mode: each invocation is fresh training, and the output is overwritten deliberately. The result file is written after each completed seed; its run count tells you whether it is complete.

CPU is sufficient (one torch thread; three seeds × three rates run in minutes in the author environment). No paid service or cloud operator is needed. The same full pipeline is inline in both notebooks, including data download and verification. Colab installs numpy, torch and scikit-learn; live Colab remains NOT_CHECKED. `requirements-l083-observed.txt` captures exact installed author versions on Python3.12; a fresh installation of that platform-specific snapshot is NOT_CHECKED. For a new environment, install the notebook dependencies plus nbformat, nbclient, nbconvert, matplotlib, networkx, beautifulsoup4, ipykernel and playwright for authoring checks. No new lab dependency beyond the existing course environment is introduced.

## Sources and data

- [Paper, version4 with appendix](https://arxiv.org/html/1706.02216v4), §3, Algorithm2, Table1 and AppendixC.
- [Official release](https://github.com/williamleif/GraphSAGE/tree/a0fdef95dca7b456dab01cb35034717c8b6dd017), revision `a0fdef95dca7b456dab01cb35034717c8b6dd017`. This identifies the inspected public snapshot, not a proven submission-era snapshot.
- [Stanford full PPI archive](https://snap.stanford.edu/graphsage/ppi.zip), SHA-256 `53aeb76e54fd41b645e7edb48b62929240b89839495396b048086fd212503fbd`.
- `_sources_l083.json` pins archive and13 reference files. Source copies and full MIT attribution are in `sources/l083/`. The original archive is downloaded and verified if absent; it need not be checked into Git.

The ZIP is read directly without extraction and the feature NPY is loaded with `allow_pickle=False`. The complete release has56,944 nodes,50 features,121 labels; split counts44,906 /6,514 /5,524. The paper describes20/2/2 tissue graphs. Tissue graphs may have disconnected pieces: our connected-component audit finds235/30/30 components, all split-pure; it does not pretend each component is a tissue.

## Protocol audit

| Element | Implementation / source decision |
|---|---|
| Feature preprocessing | StandardScaler fitted to all training rows, including any training nodes later excluded as roots for zero degree; release `utils.load_data` |
| Graph parsing | Legacy node-link endpoints index graph nodes; map through id_map to feature rows; binary undirected neighbors |
| Training access | Both endpoints training; held-out rows absent from training neighbor table |
| Inference access | Full released graph, disjoint across splits; scaler and network fixed |
| Adjacency | Fixed128 entries per row; high degree cap without replacement; low degree resampling with replacement; all-zero feature sentinel for no neighbors |
| Sampling | Shared random column permutation per hop; flags25,10 expand outward10,25; same tables per seed/rate construction |
| Aggregator | Release concat of separate self and mean-neighbor matrix products; no aggregator biases |
| Layers | 50→(128+128)→(128+128); ReLU first, identity last; final L2 normalization |
| Prediction | Biased linear256→121; no softmax; independent sigmoid decisions >.5 |
| Objective | BCE averaged over roots and labels; no dropout/weight decay (release defaults) |
| Initialization | Glorot uniform matrices; zero head bias; no pretrained weights or ID embedding table |
| Optimization | Adam eps1e-8, default betas; elementwise gradient clip[-5,5] |
| Training | Ten complete epochs; batch512; exclude training roots with zero training degree; final weights, no best-epoch restoration |
| Search | Full supervised learning-rate grid .01/.001/.0001 from AppendixC |
| Local selection | Full validation micro-F1 after ten epochs; maximum wins; deterministic first-rate tie break |
| Test freeze | Only selected model scored; test labels never enter search/training |
| Seeds | 123,124,125; repeat full search per seed; report all; no seed selection |
| Metric | Pooled TP/FP/FN across every test node-label pair, then micro-F1; never average batch F1 |
| Summary | Mean and sample SD across three seed/search runs on one fixed split |

## Published versus released versus local

Algorithm1 depicts concatenation before an unrestricted learned transform and per-layer normalization/nonlinearity. The inspected release separately transforms self and mean-neighbor states, concatenates their outputs, applies identity on the last aggregator, and normalizes only the final embedding. These choices are explained rather than combined into an untraceable hybrid.

Historical gaps: winning exact configuration, complete model-selection details, seeds and table replicate aggregation are not supplied in the paper/release. We declare the local final-validation selection explicitly; the released trainer prints intermediate validation but trains all10 epochs. We do not claim that our selection rule is the undisclosed historical rule.

PyTorch changes RNG streams, initialization draw order, Adam numerical behavior and kernels from TensorFlow1. We sort neighbor lists and root IDs for deterministic modern Python behavior; the original uses networkx/set iteration. Independent evaluation RNG avoids evaluation frequency altering subsequent training draws. These are deliberate reproducibility differences. No bitwise cross-framework or source-training parity is claimed. Dense NumPy arithmetic checks validate our mathematical path, not original TensorFlow execution.

The curriculum mentions PyG NeighborLoader. Here the mini-batch sampler is implemented visibly in native PyTorch to reproduce the release's padded-table and shared-column semantics, which a default modern NeighborLoader does not guarantee. A PyG API lab follows in the course; no PyG dependency is needed for this one.

## Evidence ledger

- **Verified mechanisms:** source/archive hashes, induced graph access, support order and sentinel, exact concatenation arithmetic, neighbor-order invariance, independent two-layer NumPy oracle, real graph split isolation, complete one-epoch intervention proving held-out features/labels do not change training weights.
- **Executed full experiment port:** three seeds × three learning rates × ten epochs, full data and width. Mean0.59092968, sample SD0.00605956; all rates selected using validation, all seeds select .01. Paper target0.598; observed difference−0.00707032. This is descriptive, with no post-hoc acceptance band.
- **Historical paper-result parity:** INCOMPARABLE due to explicit gaps above. The run is full-scale for the selected experiment; it does not reproduce the complete paper or validate the unexecuted original TensorFlow trainer.
- **Other variants/datasets/unsupervised experiments:** NOT_RUN.
- **Delivery:** solution replay, browser and copied-site checks have separate JSON records; live Colab and deployed Pages remain NOT_CHECKED.

Generating a lesson is not learner completion. EXIT requires the learner's code, evidence and written access/shape/protocol explanations.
