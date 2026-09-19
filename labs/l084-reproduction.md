# L084 · GAT Cora reproduction contract

## Named published experiment

Veličković et al., *Graph Attention Networks*, ICLR 2018, [Table 2](https://arxiv.org/html/1710.10903v3#S3): Cora GAT **83.0 ± 0.7%**, mean and standard deviation over **100 runs**. This package provides a complete fresh-training port of that named dataset/model experiment. It does not reproduce the entire paper's four datasets and baselines.

Primary implementation: [PetarV-/GAT revision 5af87e7fce2b90ae1cbd621cd58059036a3c7436](https://github.com/PetarV-/GAT/tree/5af87e7fce2b90ae1cbd621cd58059036a3c7436). Archived files and MIT attribution are under `sources/l084/`; SHA-256 values and exact data URLs are in `_sources_l084.json`. Source is evidence, not a claim that the original TensorFlow program was executed.

## Protocol audit

| Component | Paper / released program | This port |
|---|---|---|
| Dataset | Cora, Planetoid fixed split | Full verified archive bytes; 2708 nodes, 1433 features, seven classes |
| Split | 140 train / 500 validation / 1000 test | Exact indices including original test-index reordering |
| Features | Row-normalized bag of words | Per-row normalization, zero rows remain zero; all features visible |
| Graph | Undirected citations and self | Binary deduplicated edges in both directions, one self-loop per node; 13264 directed entries |
| Hidden layer | 8 independent heads × 8, ELU, concat | Same; separate per-head projections and score functions |
| Output | One head × 7; no hidden activation | Same; class logits into CE; general merge supports mean |
| Attention | Additive source/receiver projections, LeakyReLU .2, neighborhood softmax | Same deterministic equations; edge-only segmented softmax |
| Biases | Release has two scalar score biases + output vector bias | Preserved (paper equations omit explicit biases) |
| Dropout | p=.6 input, normalized coefficients; release additionally drops projected values | All three locations preserved, independent across heads; sparse nonzero input dropout |
| Initialization | Glorot uniform kernels, zero biases | PyTorch Xavier uniform per matching matrix; different RNG |
| Objective | Masked CE + lambda=.0005 times half squared norm | Same. Release full-name filter fails to exclude scoped bias names, so all trainable parameters are penalized |
| Optimizer | TF Adam .005, beta .9/.999, epsilon1e-8 | PyTorch Adam same settings; epsilon/update implementation differences remain |
| Validation | Unregularized CE and accuracy, no dropout | Same, all 500 validation nodes |
| Selection | Reset patience on either best/tied metric; save only when both best/tied | Same OR-reset / AND-save; restore saved weights |
| Schedule | At most100000 epochs; patience100 | Same, no shortened schedule in full lane |
| Test | Evaluate restored checkpoint once | Same, no test-based tuning or seed selection |
| Repetitions |100 runs; historical seed IDs unspecified | Declared local seeds0–99, all reported |
| Aggregation | Mean ± SD | Mean and sample SD (ddof=1); no original SD divisor specified |

The paper counts5429 citation edges. The raw released adjacency lists contain10858 entries but only10556 unique directed pairs; NetworkX coalesces duplicates in the original loader. This port likewise has5278 unique undirected pairs, then adds2708 self-loops. All eight data files were separately compared with the pinned GAT repository and match byte-for-byte (`_data_identity_l084_results.json`).

A self-loop is an allowed sender, not a GCN degree-normalization factor. The port's attention is not modern GATv2 or transformer dot-product attention. The architecture could transfer to unseen graphs; this Cora protocol is transductive.

## Exact commands

For a clean Python3.12 CPU training/plotting environment (the author used Linux aarch64):

```bash
python3 -m venv .venv-l084
.venv-l084/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r labs/requirements-l084-runtime.txt
.venv-l084/bin/python labs/_run_l084.py --seeds 100 --workers 4
.venv-l084/bin/python labs/_clean_environment_l084.py
.venv-l084/bin/python labs/_check_fresh_cli_l084.py
```

The clean Linux aarch64 CPU install and full659-epoch seed0 replay passed, including identical epoch losses and test accuracy; `requirements-l084-runtime.txt` freezes that isolated environment. Notebook authoring/browser tools are separate and listed in the observed snapshot.

From the repository root, using the existing authoring environment:

```bash
.venv/bin/python labs/_sources_l084.py
.venv/bin/python labs/_verify_l084.py
.venv/bin/python labs/_run_l084.py --seeds 100 --workers 4
.venv/bin/python labs/_figures_l084.py
.venv/bin/python labs/_build_l084.py
.venv/bin/python labs/_execute_l084.py
.venv/bin/python labs/_browser_l084.py
.venv/bin/python labs/_delivery_l084.py
```

`_run_l084.py` defaults to the full data/model/schedule and writes `_paper_l084_results.json`, per-seed epoch traces in `runs/l084/`, seed0's checkpoint and `_attention_l084.json`. It starts fresh; it does not silently reuse completed seeds. The parent verifies/populates the cache before spawning workers, preventing concurrent first-download writes. `--max-epochs 5 --seeds 1 --output /tmp/l084-smoke.json` is only a teaching smoke run and is INCOMPARABLE to the published repetition/schedule. Run a smoke command in an isolated copy if you want to retain full-run per-seed artifacts.

The complete model and trainer are inline in both notebooks. Student TODOs directly feed training. Default solution execution runs five diagnostic epochs plus a complete seed0; `RUN_100_SEEDS=True` enables the complete100-run notebook lane. The author CLI100-run execution and independent notebook1-run replay are reported separately. The notebook does not require local `relkit` imports or checkout-relative files; it embeds the data manifest and downloads verified data if absent. It is also usable outside this repository.

`requirements-l084-observed.txt` records installed versions; it is an observed CPU environment, not a claim of a successfully rebuilt environment on every platform. For a new CPU environment, install torch from the PyTorch CPU index and the remaining listed versions from PyPI. The notebook pins the principal numerical packages on Colab. Live Colab and other operating systems remain NOT_CHECKED. The isolated CPU environment replay is recorded in `_clean_environment_l084_results.json`.

## Evidence and limits

- `_clean_environment_l084_results.json`: isolated CPU install, fresh empty-directory data download and complete seed0 trace/score comparison.
- `_verify_l084_results.json`: independent dense masked attention, gradient checks, permutation equivariance, head merging, shapes, self-loop count and test-label intervention.
- `_paper_l084_results.json`: fresh training scores, every epoch, stopping flags, environment and implementation/source-manifest hashes.
- `_execution_l084_results.json`: actual solution execution; full seed0 score and all epoch losses equal the CLI record. Notebook100-run lane is separately NOT_RUN by default.
- `_attention_l084.json`: actual learned first-head routing for receiver0, selected by fixed ID before test-score inspection. This is not causal importance.
- `_browser_l084_results.json`: tested interactive intervention, keyboard behavior, responsive page and portable images.
- `_delivery_l084_results.json`: copied Pages staging, links, canonical/inline implementation parity and data/source identities.

**Historical parity: INCOMPARABLE.** Sparse versus dense computation, random streams, framework/Adam numerics and unavailable historical seeds remain deviations even if the mean is close. Original TensorFlow execution is NOT_RUN. Citeseer, Pubmed, PPI and paper baselines are NOT_RUN. No multi-dataset superiority claim follows from this lesson. SD across seeds describes this split's training variability only.

Creation of this package does not establish learner completion or mastery. Live Colab and remote deployment remain NOT_CHECKED.

<!-- MEASURED-L084:begin -->
## Measured full experiment

All100 declared seeds completed the full stopping schedule. Mean **83.193%**, sample SD **0.821 percentage points**, versus paper83.0 ±0.7%. Epoch counts ranged from514 to1286. Full epoch traces and file identities: `_paper_l084_results.json`. This is a measured modern port, not original-framework parity.
<!-- MEASURED-L084:end -->
