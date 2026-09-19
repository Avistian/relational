# L074 CARTE reproduction contract

## What this package implements

Primary skill: build schema-variable row graphs, extract real CARTE embeddings, and compare transfer with scratch target learning. A visible PyTorch implementation matches the pinned release's one-readout `CARTE_Base` architecture at zero dropout. Source commit: `f54690da4cddbedd1e1a9113a312f85783d2c125`. Source code and BSD license are retained under `sources/carte-l074/`.

The local encoder strictly loads every selected `ft_base.initial_x`, `initial_e` and `read_out_block` tensor. Other pretrained layers and the pretraining projection are unused. This loading rule deliberately differs from the release estimator, which renames initial_x keys before a non-strict load. The lab trains a linear head or fits ridge, rather than the paper's downstream multilayer head and bagging. It follows the released edge-conditioned center, sender indexing and node block without residual additions. These distinctions are teaching content. The `rows` execution path skips unused final leaf outputs; output and parameter-gradient parity against the full forward path are checked. It changes execution cost, not the prediction function.

## Data and protocol

Three actual example tables from the pinned repository: wine_pl, wine_dot_com_prices, wine_vivino_price. Exact repeated entity-name rows are removed within source; 384 rows per source are chosen by a label-blind seed74 sampler. `data/l074/manifest.json` records original row IDs, original parquet digests, string-cache digest and checkpoint digest. No cross-source entity resolution is performed; tables are fitted separately. The already-transformed target values are preserved. This is background-YAGO-to-table transfer, not supervised source-table joint learning.

For seeds 0/1/2, the same permutation supplies 64 train, 64 validation and 256 test rows to every arm. Fit numerical PowerTransformer per varying column on train only. Constant observed training columns use StandardScaler; entirely missing training columns are omitted for all rows of that split. Wine Poland seed2 has only volume750 in training; the initial power fit gave exponent≈35 and overflowed float32. `_numeric_l074_results.json` records the regression check. All arms are rerun with this train-only fallback. FastText sentence vectors are fixed pretrained features, cached for required strings only; no learned statistics or target labels enter vector extraction. Unknown strings are rejected by dictionary lookup. Potential entity overlap with YAGO/FastText pretraining is not audited.

Frozen arms: language-conditioned center, random CARTE encoder, pretrained CARTE encoder; StandardScaler fitted on train, Ridge alpha in [1,10,100] chosen by validation MSE. Trainable arms: scratch and pretrained encoder with identical linear-head seed; 40 full-batch epochs of AdamW, lr1e-4, weight_decay1e-3, dropout0; validation chooses epoch. CatBoost uses 150 iterations, depth4, lr.05, CPU1, fixed before evaluation. Numerical missing values in the tree baseline use training medians; strings use a missing token.

R² is recomputed from saved predictions. Six arms × three datasets × three paired repetitions = 54 evaluations. Means/SD describe these repetitions; ranks average repetitions first and then weight datasets equally. Friedman/Nemenyi results with three related wine datasets are exploratory, not a broad significance claim. Paper tables/splits, target budgets, full pretraining, ensembles and the remaining benchmark datasets are not reproduced. Verdict: **INCOMPARABLE**.

## Run locally

From the repository root, use the course virtual environment:

```
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l074.py
.venv/bin/python labs/_build_l074.py
.venv/bin/python labs/_execute_l074.py
.venv/bin/python labs/_browser_l074.py
.venv/bin/python labs/_delivery_l074.py
```

The default lab uses existing course dependencies and the compact cache. Independent upstream checks additionally require `torch-geometric==2.8.0.post1`; optional packages are listed in `requirements-l074-audit.txt`. Re-extracting the language cache requires `fasttext-wheel==0.9.2`, the 7.24 GB English FastText binary and enough RAM. This is unnecessary for student execution:

```
git clone https://github.com/soda-inria/carte /tmp/carte-l074
git -C /tmp/carte-l074 checkout f54690da4cddbedd1e1a9113a312f85783d2c125
.venv/bin/python labs/_prepare_l074.py --repo /tmp/carte-l074 --fasttext /path/to/cc.en.300.bin
.venv/bin/python labs/_check_l074.py
```

FastText source: https://huggingface.co/hi-paris/fastText/resolve/main/cc.en.300.bin (binary hash in manifest). English vectors originate with FastText; see https://fasttext.cc/docs/en/crawl-vectors.html for distribution and CC BY-SA 3.0 terms. The cache contains derived sentence vectors with attribution. Source parquet/checkpoint provenance is retained from the CARTE repository under its license; the three supplied subsets remain research teaching examples.

## Post-EXIT run

```
cd labs
../.venv/bin/python _run_l074.py --preset closer
```

Presets: smoke = one seed, 2 epochs, 64 train; closer = three seeds, 100 epochs, 128 train. These retain the same implementation and 64 validation rows. The test set consequently has 192 rows in closer; between-preset score differences confound the resource change with a changed test set. The operator rejects `paper`: no configuration here establishes full-paper fidelity. CPU suffices; no cloud action is needed. The notebook includes the same gated closer call. Larger run remains NOT_RUN unless its output is explicitly present and inspected.

## Verification boundaries

`_check_l074_results.json`: actual model parity and graph/attention invariants.
`_execution_l074_results.json`: executed solution and fresh record parity.
`_browser_l074_results.json`: local responsive browser interactions.
`_delivery_l074_results.json`: canonical definition parity, portable images and actual copied Pages links.

Live Colab and remote deployment: **NOT_CHECKED**. New YAGO pretraining, paper-scale benchmark and joint supervised multi-table experiment: **NOT_RUN**. Authored material does not establish learner completion.
