# L052 — TabR-S reproduction contract

The lesson implements the complete numeric TabR-S forward pass, not full TabR with numerical embeddings. Source: arXiv:2307.14338v2, Figures 2–4, Eq. 5; released revision `17baa9082506f8e7a0f8d11bb1e08212926a1507`. Original source and MIT notice are preserved in `sources/l052/`; `_sources_l052.json` hashes those files.

## Source validation

`_source_check_l052.py` executes the pinned official Model class through an AST extraction. The only adapters are exact nearest-neighbor search in PyTorch (instead of Faiss), a Lambda layer and output-dimension utility. Copied weights, dropout disabled, classification/regression and training/evaluation candidate paths: outputs agree within 1.20e-7; query-input gradients within 5.97e-8. This is not Faiss runtime, preprocessing, optimizer or complete training parity. Cutoff distance ties are not covered by parity assertions.

## Local measurement

`_fetch_l052.py` uses validated HTTP ranges to extract only the authors' California Housing, House 16H and Higgs Small numeric arrays and targets. `_data_l052.json` records archive byte offsets, sizes and SHA256 hashes. These arrays are cached, not committed; fresh local/Colab runs fetch them using the manifest. The archive contains 3 GB; extraction does not download it all.

Original splits are retained. Label-blind sampling caps train/val/test at 1200/600/600 with seeds 52/53/54. Training-only median imputation and normal-quantile transforms; no released quantile jitter. Regression targets use the training mean and population SD. Library versions are in `_verify_l052_results.json`.

Models: matched retrieval-free TabR-S backbone (MLP), complete TabR-S, packaged XGBoost. Neural width 32, m=16, AdamW lr=.001, weight decay 0, activation/context dropout .1, at most 25 epochs, batch size 256, patience 8. Tree depth 4, lr=.05, 150 rounds, .8 row/column sampling, validation early stopping 16. Model seeds 0/1/2. No hyperparameter search or matched-time claim. Accuracy for Higgs; RMSE in the archive's original target units for regression. House table scaling in the paper is not reproduced.

`_verify_l052.py` produces complete per-seed predictions, validation history, selected row indices, means, sample SD, conditional 95% Student-t seed intervals, mean dataset ranks, Friedman and Nemenyi CD. Test targets are excluded from all selection. The inference-only label permutation keeps keys and fitted weights fixed; it is not a retrained ablation. No temporal experiment was performed.

## Required larger run

Run from repository root:

```bash
.venv/bin/python labs/_fetch_l052.py
.venv/bin/python labs/_check_l052.py
.venv/bin/python labs/_source_check_l052.py
.venv/bin/python labs/_verify_l052.py
.venv/bin/python labs/_paper_repro_l052.py --preset closer --out labs/data/cache/l052-your-run
modal run --detach modal/l052_paper_repro.py --preset closer
```

Both Modal and the notebook's gated Colab cell use the same from-scratch implementation. The Colab loop receives live student functions, not an imported replacement. Smoke tests a small fit and resume; closer uses California 6000 training rows, full validation/test, width 64, m=96, 60 epochs, three seeds. Paper uses full California splits, 15 seeds, and the pinned selected configuration (d303, dropout .5508268685, context dropout .2325842283, lr .000280842117, m96, patience16, zero decay) with a finite 1000-epoch cap.

Reference target: **Table 3, TabR-S, California RMSE .403**, absolute tolerance .01 only after protocol compatibility. Current verdict is **INCOMPARABLE**: no new search; no quantile jitter; modern library/initialization/batch-RNG differences; torch search; finite epoch cap. Closer additionally changes training size, width, optimization and seed count. No protocol-match flag is inferred from a similar numeric score.

Resume stores completed seeds atomically. Code (including current notebook helper bytecode), operator, data, configuration, Python/platform and library versions are checked; a changed contract requires a new output directory. Partial seeds restart. Cached execution times describe the original completed seeds, not the cost of resuming them.

Full selected-configuration run, original tuning, ensembles, 43-task comparison, full TabR embeddings, context freeze, temporal transfer and cloud execution are NOT_RUN. Completed larger local evidence: `_paper_repro_l052_closer_summary.json`, California RMSE 0.45651 ± 0.00400 over three seeds; 1345.3 CPU seconds. Verdict INCOMPARABLE.

## Delivery

Builder: `_build_l052.py`; student notebook, executed teacher solution, prepared HTML, six inline PNGs. Source/protocol checks, arithmetic, live-code gate, source identity, pedagogy and copied Pages staging are tracked by `_delivery_l052_results.json`. Browser, live Colab UI and deployed Pages checks remain separate from notebook payload or static rendering checks.
