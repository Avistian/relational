# Lesson 062 · historical TabPFN v1 reproduction contract

[Lesson](../lessons/0062-tabpfn-v1.html) · [Runnable notebook](0062-tabpfn-v1.ipynb) · [Sources](_sources_l062_v2.json).

## What is implemented and measured

`relkit/tabpfn_l062_v2.py` implements the full released numeric inference path: context-fitted sample normalization, constant-column removal, optional power transforms and source-style column fallback, two-pass outlier softening,100/k scaling, zero padding, linear feature and context-label encoders,12 postnorm GELU blocks, four 128-wide heads,512 hidden width, 1024 FFN width, 10-logit head, class unrotation, logit averaging and temperature .8. It copies the original released checkpoint into the visible model. Every core function/class/helper is inlined in coherent notebook chunks. Five live code tasks plus a written EXIT are required.

This is **pretrained inference**. No original SCM/BNN prior training is newly run. The paper reports 9,216,000 synthetic datasets and 20 hours on 8 RTX 2080 Ti GPUs; the checkpoint is inherited evidence, not a new training result. The copied-weight/source checks certify the named forward computation and numeric wrapper paths. They do not recover the original distributed optimizer run or prove that all package options match.

Unsupported wrapper options: explicit categorical-column handling, ranking preprocessing, query-fitted normalization, feature subsampling beyond 100. All-constant context features are rejected explicitly. Numeric missing entries are handled by the copied path's moment calculation and final nan-to-zero encoder, not by a dedicated missingness embedding.

## Fresh local commands

From the repository root, use the existing course environment:

```bash
PYTHONPATH=labs .venv/bin/python labs/_check_l062_v2.py
PYTHONPATH=labs .venv/bin/python labs/_verify_l062_v2.py --preset smoke --output labs/data/cache/l062-smoke-fresh.json
PYTHONPATH=labs .venv/bin/python labs/_verify_l062_v2.py --preset lab --output labs/data/cache/l062-lab-fresh.json
PYTHONPATH=labs .venv/bin/python labs/_verify_l062_v2.py --preset closer --output labs/data/cache/l062-closer-fresh.json
.venv/bin/python labs/_figures_l062_v2.py
.venv/bin/python labs/_build_l062.py
.venv/bin/python labs/_execute_l062.py
```

Outputs must use fresh filenames; the runner refuses an existing file and offers no resume. The scoped builder preserves teacher outputs only if code cells are byte-identical. Teacher solutions are ignored locally; the student notebook remains blank and unexecuted. `build_package(notebooks=True,render=True)` and `build(solution=False)` are the scoped builder APIs. Generic foundation routes delegate to these APIs.

The first run downloads the original checkpoint if absent (about 103 MB), verifies its full SHA256 before deserialization, and downloads a hash-verified 0.1.11 reference wheel for comparisons. It does not install a moving default TabPFN package as the predictor. The source wheel is re-extracted from verified bytes for each source check. Cached checkpoint path is `data/cache/foundation/v1/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt` relative to labs.

| Preset | Data | Splits | Views | Meaning |
|---|---|---|---|---|
| smoke |120 stratified diabetes rows |seed 0,50/50 |1 and 4 |execution wiring only |
| lab |all diabetes 37,blood 1464,WDBC 569 rows |seeds 0,1,2,50/50 |1 and 4 |new measured local subset |
| closer |same full three datasets |seeds 0–4,50/50 |1 and up to 32 |closer subset inference protocol |

The default notebook runs full diabetes with split/view seed 7 and one/four views, then shuffles only context labels with seed 6207. It records predictions, per-view logits/configuration, row IDs and metrics. It additionally recreates the power-transform fallback counterexample using the learner's live predictor. An off-by-default post-EXIT cell invokes `closer` through the same live functions. The existing optional CPU Modal operator `modal run --detach modal/foundation_repro.py --lesson 62 --preset closer` uses the shared delegated runner. It has not been run; no cloud expenditure is authorized by local packaging. A `paper` preset is intentionally unsupported.

## Evidence files and limits

- `_verify_l062_v2_results.json`: new full-data author panel,18 records, three datasets×three local 50/50split seeds×two view budgets. It preserves every probability, target, view logit, class shift and feature-transform configuration; data arrays are hashed and original row IDs saved. WDBC is sklearn's UCI copy, not a proven identical ordered OpenML 1510 release.
- `_check_l062_v2_results.json`: full pretrained float32/float64 parity, input/all 152 mapped parameter-gradient checks, complete binary/multiclass/constants/missing/outlier wrapper cases, context permutation and label sensitivity.
- `_query_fallback_l062_results.json`: independently discovered, original-source-confirmed numerical counterexample to universal wrapper query independence. Context-only power fitting does not prevent an added query from triggering whole-column fallback during transformation. In the recorded environment probability changes by .10233; none-transform control stays unchanged. The historical operator is intentionally preserved.
- `_execution_l062_v2_results.json`: actual ignored teacher execution status. Its EXIT at `data/cache/l062-student/exit.json` points to a fresh unique run file containing actual live functions/methods/defaults/kwdefaults/configs/presets and library versions, call counts, current predictions and written interpretation. Repository hashes and checkpoint hashes are distinct identities.
- Historical `_verify_l062_results.json`, `_verify_l062.py`, original `foundation_core.py` and earlier pretrained runner remain unchanged. Their 180-context/60-test evidence is not rehashed or relabeled as the corrected operator.

The paper's main comparison has 18 numeric no-missing datasets, five 50/50 splits, normally 32 views, tuned competitors and declared time budgets. Our new local panel shares three source datasets and the split proportion but uses different row-order/split construction, fewer splits/views and no tuned competitor panel. We therefore do **not** claim Table 1/2 reproduction, a paper speedup or a model-family ranking. Split SD is descriptive variation across overlapping resamples, not an independent-dataset confidence interval. AUC and log loss are both retained. Timings include preprocessing and inference but exclude data/checkpoint loading; one CPU thread, local host conditions, context recomputed per querychunk.

Code parity: checked. Actual full pretrained inference through live code: checked. Fresh original prior fitting: NOT_RUN. Full original benchmark: INCOMPARABLE. Closer protocol: NOT_RUN unless a separate fresh artifact says otherwise. Browser, copied Pages, live Colab and deployment are separate delivery checks owned by the parent review; prepared HTML is not live Colab. No learner mastery is recorded.
