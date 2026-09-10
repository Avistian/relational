# L051 reproduction contract

The deliverable is a three-intervention investigation, with an explicit **key-parts** mirror of Grinsztajn et al. §5. No new model is introduced. The MLP and complete numeric ReGLU FT-Transformer from L050 are visible in the notebook; XGBoost is the existing packaged baseline.

Primary sources: [paper v1](https://arxiv.org/html/2207.08815v1), [released transformations](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/src/data_transforms.py), [OpenML suite 337](https://www.openml.org/search?type=benchmark&study_type=task&id=337). Source revision and file SHA-256 values are in `_sources_l051.json`; dataset IDs, dimensions and local parquet hashes are in `_data_l051.json`.

| Aspect | This implementation | Fidelity boundary |
|---|---|---|
| Target smoothing | Training-only top five; Gaussian weights including self; h=.5; strict >.5 classification | Matches released hard-label behavior; parity checks classic and coupled-seed robust nonsingular covariance, while training adds an explicit eigenvalue floor |
| Rotation | scipy special_ortho_group, same R on every partition | Released operation matches; not an Adam-training invariance assertion |
| Extra noise | 2d independent N(0,1) columns appended to d existing columns | Follows paper caption; released code instead matches selected training-column mean/IQR |
| Data | electricity 44120, MagicTelescope 44125, bank-marketing 44126 | Authors' January 2023 processed release, later than paper v1; three-task subset |
| Preprocessing | Training-only normal quantile transform, before intervention | No post-rotation scaling; no independently reconstructed upstream data-cleaning pipeline |
| Splits | Label-blind row subsample, stratified 60/20/20 seed 51 | Local split, not upstream split IDs or replication count |
| Model recipes | MLP width64, FT-T d32/two blocks/four heads, depth4 XGBoost | Fixed local recipe; no published hyperparameter search/reordering, no ResNet/SAINT/RF arm |
| Selection | Early stop on original validation labels, restore best validation loss | Same caps across conditions, not equal wall-clock budgets |
| Evidence | Accuracy, AUROC, paired effects, conditional t intervals, dataset ranks/Friedman/CD | Model seeds share a split and transform; only three tasks; five exploratory condition tests |

Important Appendix A.3 difference: the paper's nominal partition is 70% training (subject to cap), then 30%/70% of the remainder for search validation/test. Its neural early-stopping subset is **inside training**, separate from hyperparameter-selection validation. The local fixed-recipe experiment uses a single 60/20/20 partition and no separate search. Neural caps are 20 rather than 300 epochs, and patience 12 rather than 40 for the compared MLP/FT-T. The paper varies evaluation folds with test-set size; model seeds here do not replace those folds. Fig. 3's 15 search reorderings and Fig. 5's 30 reorderings are not local seed confidence intervals.

In the pinned robust smoother, a failed covariance can trigger an empirical-covariance fallback that does not reapply `cov_mult`. This behavior is deliberately not mirrored. The source check's robust example couples estimator randomness and verifies that no fallback occurred. Both classic and robust probability errors were 4.44e-16 in the audit; this is operation parity, not trainer or singular-case parity.

## Added bandwidth diagnosis (2026-09-10)

`_diagnose_l051.py` is an additive operator, separate from historical `_verify_l051_results.json` and the closer artifact. It reuses the unchanged `prepare_task`, `smooth_targets`, `fit_arm` and `paired_effect`, with electricity's same 1,800 selected rows, split, top-five features, covariance, seeds and recipe. It tries h=0, .25, .5, 1, retains raw validation/test labels, saves every test probability and selection log, and explicitly records `NOT_FIT_ONE_CLASS` if thresholding collapses the training classes. Such a point receives no fabricated model score.

The h=0 and h=.5 prediction arrays match the historical top-five and smoothed conditions. Changed-label counts are 0, 13, 125, 230 out of 1,080. At h=1 mean accuracy changes are −3.15, −3.52 and −3.15 percentage points for MLP, FT-T and XGBoost. All decline; this is **not** the paper's aggregate pattern of stronger tree sensitivity at modest smoothing. Evidence: [`_diagnostic_l051_results.json`](_diagnostic_l051_results.json), with source/data hashes and versions. Verdict remains **INCOMPARABLE**. The teacher notebook runs the same visible operator with current live student functions and writes `data/cache/l051-bandwidth-live.json`; author-reference plots remain separately labeled.

```bash
.venv/bin/python labs/_diagnose_check_l051.py
.venv/bin/python labs/_diagnose_l051.py
```

The first command runs six tiny real-data model fits, validates raw targets/split/metric arithmetic, and probes a very large bandwidth to verify explicit one-class reporting. The second regenerates 36 local fits (4 bandwidths × 3 models × 3 seeds). Do not change the historical result hashes to accommodate edits to a diagnostic; a model/recipe change requires a separately identified rerun.

## Regenerate

From repository root, using the lab environment:

```bash
python labs/_fetch_l051.py
python labs/_check_l051.py
python labs/_source_check_l051.py
python labs/_verify_l051.py
python labs/_diagnose_l051.py
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

All presets keep five intervention conditions and the same three tasks. The full benchmark roster, search curves, published smoothing-lengthscale aggregation and feature-removal sweep remain **NOT_RUN**; the added one-task bandwidth curve does not fill these gaps. A larger run is still **INCOMPARABLE**. There is no arbitrary scalar tolerance for matching an aggregate figure drawn under a different protocol.

For the original experiment route, inspect the [pinned official README](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/README.md): it specifies an isolated Python 3.8 environment, WandB configuration, `src/launch_benchmarks/launch_benchmarks.py`, sweep agents and R analyses. Those sweeps and analyses were **not run** in this audit. Their published raw-search download supports re-analysis of existing author results, which is different from retraining. Before an exact reproduction, resolve paper-version versus release-date differences, choose the matching transform configs/figure analysis, reconstruct the full data and nested selection protocol, then regenerate curves from all candidate scores. This lab's `paper` name is a resource preset, not that official route.

## Delivery boundary

The solution is executed with IPython in-process, capturing streams and rich displays into standard notebook outputs after the separate kernel runner proved unreliable/slow. The solution must execute every code cell; student TODOs stay blank and on the actual training path. Seven PNG figures are embedded as inline data URLs in both notebooks and prepared HTML. Numerical/interaction checks and vector-renderer inspections are distinct from live browser and Colab checks; the current audit's browser result is in `reviews/lesson-quality-audit-047-070/051-browser.json`. Live Colab remains untested.
