# L051 reproduction contract

The deliverable is a three-intervention investigation, with an explicit **key-parts** mirror of Grinsztajn et al. §5. No new model is introduced. The MLP and complete numeric ReGLU FT-Transformer from L050 are visible in the notebook; XGBoost is the existing packaged baseline.

Primary sources: [paper v1](https://arxiv.org/html/2207.08815v1), [released transformations](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/src/data_transforms.py), [OpenML suite 337](https://www.openml.org/search?type=benchmark&study_type=task&id=337). Source revision and file SHA-256 values are in `_sources_l051.json`; dataset IDs, dimensions and local parquet hashes are in `_data_l051.json`.

| Aspect | This implementation | Fidelity boundary |
|---|---|---|
| Target smoothing | Training-only top five; Gaussian weights including self; h=.5; strict >.5 classification | Matches released hard-label behavior; source parity uses classic nonsingular covariance, while training uses MinCovDet plus explicit eigenvalue floor |
| Rotation | scipy special_ortho_group, same R on every partition | Released operation matches; not an Adam-training invariance assertion |
| Extra noise | 2d independent N(0,1) columns appended to d existing columns | Follows paper caption; released code instead matches selected training-column mean/IQR |
| Data | electricity 44120, MagicTelescope 44125, bank-marketing 44126 | Authors' January 2023 processed release, later than paper v1; three-task subset |
| Preprocessing | Training-only normal quantile transform, before intervention | No post-rotation scaling; no independently reconstructed upstream data-cleaning pipeline |
| Splits | Label-blind row subsample, stratified 60/20/20 seed 51 | Local split, not upstream split IDs or replication count |
| Model recipes | MLP width64, FT-T d32/two blocks/four heads, depth4 XGBoost | Fixed local recipe; no published hyperparameter search/reordering, no ResNet/SAINT/RF arm |
| Selection | Early stop on original validation labels, restore best validation loss | Same caps across conditions, not equal wall-clock budgets |
| Evidence | Accuracy, AUROC, paired effects, conditional t intervals, dataset ranks/Friedman/CD | Model seeds share a split and transform; only three tasks; five exploratory condition tests |

## Regenerate

From repository root, using the lab environment:

```bash
python labs/_fetch_l051.py
python labs/_check_l051.py
python labs/_source_check_l051.py
python labs/_verify_l051.py
node labs/_viz_check_l051.js
python labs/_figures_l051.py
python labs/_publish_l051.py
python labs/_build_l051.py
python labs/_execute_l051.py
python labs/_render_l051.py
python labs/_delivery_check_l051.py
```

The first download requires network access. Subsequent measurements reuse cached parquet files. Versions for the actual run are recorded in each results artifact. `_verify_l051_results.json` includes original row indices, split indices, covariance, selected features, rotation, smoothed probabilities, seed-wise test probabilities, best validation losses, metrics and source hashes. Some bank-marketing training coordinates cause MinCovDet determinant warnings; the measured covariance and eigenvalue floor are retained. This is a numerical/protocol limitation, not suppressed evidence of a different winner. Warnings also appear in the executed teacher notebook.

## Required scale-up

```bash
python labs/_paper_repro_l051.py --preset smoke --out labs/data/cache/l051-smoke
python labs/_paper_repro_l051.py --preset closer --out labs/data/cache/l051-closer
modal run --detach modal/l051_paper_repro.py --preset closer
```

The notebook has a gated cell (`RUN_PAPER_REPRO=False`) that passes the current live implementation to the same visible operator. It rehashes student functions, model methods and settings at gate execution. A changed source, dataset, environment, device or configuration rejects old cache reuse. Checkpoints contain completed datasets; interrupted datasets restart. Modal commits the volume when the job finishes; for durable unattended interruption recovery, ensure the platform's volume persistence semantics meet your needs. No cloud job is launched by creating this lesson.

| Preset | Selected rows/task | Epochs / tree rounds | Model seeds |
|---|---:|---:|---|
| smoke | 500 | 3 / 25 | 0 |
| lab | 1800 | 20 / 120 | 0,1,2 |
| closer | 6000 | 50 / 400 | 0,1,2 |
| paper (resource name) | up to 20000 | 100 / 1000 | 0–4 |

All presets keep five intervention conditions and the same three tasks. The full benchmark roster, search curves, smoothing-lengthscale sweep and feature-removal sweep remain **NOT_RUN**. A larger run is still **INCOMPARABLE**. There is no arbitrary scalar tolerance for matching an aggregate figure drawn under a different protocol.

## Delivery boundary

The solution is executed with IPython in-process, capturing streams and rich displays into standard notebook outputs after the separate kernel runner proved unreliable/slow. The solution must execute every code cell; student TODOs stay blank and on the actual training path. Six PNG figures are embedded as inline data URLs in both notebooks and prepared HTML. Numerical/interaction checks and vector-renderer inspections are distinct from live browser and Colab checks, which are recorded separately in `_delivery_l051_results.json`.
