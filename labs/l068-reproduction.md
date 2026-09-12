# L068 reproduction contract · Built with TabPFN

The complete released baseline, drift and NoT2V model computations are visible in `relkit/driftpfn_l068_v2.py` and the notebook. The original source is pinned at `a6e75afb82d13e7abb46deade3669c4106b3d636`. `_sources_l068_v2.json` records source-file and final NeurIPS PDF digests; the core pins all seven checkpoint and three raw-data hashes. `sources/l068-v2/LICENSE.txt` and `NOTICE.md` preserve license/attribution. Checkpoints and downloaded data/source remain in ignored caches.

## Run supported local experiments

From the repository root, with the established environment:

```bash
.venv/bin/python labs/_paper_audit_l068_v2.py
.venv/bin/python labs/_check_l068_v2.py
.venv/bin/python labs/_identity_l068_v2.py
.venv/bin/python labs/_verify_l068_v2.py --preset smoke --output data/cache/l068-smoke-fresh.json
.venv/bin/python labs/_verify_l068_v2.py --preset full_local --output data/cache/l068-full-fresh.json
```

Use a fresh output path. Existing experiment files are rejected; there is no silent resume or overwrite. Source checks are regenerated reports; experimental prediction files remain immutable. The default notebook runs `PRESETS68['lab']`: full Electricity and Blobs, the first cutoff, checkpoint 1, five arms and one reconstructed-SCM diagnostic seed. Its optional `RUN_BROADER` gate runs the entire `full_local` panel with the **same live namespace**, not a packaged replacement. CPU is sufficient; no Modal/cloud dependency is required for this run. Live Colab remains NOT_CHECKED unless explicitly verified in delivery evidence.

Presets: `smoke` uses 12 rows per domain, one Blobs repetition; `lab` retains all Electricity/Blobs rows for repetition 0; `closer` uses four tasks, three repetitions, a 64-row/domain cap; `full_local` removes that cap. None is named a paper preset, because none implements the full paper selection and preprocessing protocol. The final measured panel used `full_local`, 156.8 CPU wall seconds on the author machine; this is not a model-speed benchmark.

## Exact local protocol

Dataset preparation byte-matches original-loader output for Electricity (1260×5), Parking (1294×4), Chess (533×9), and original three-class Intersecting Blobs (1680×2). Hashes include X/y/domain vectors. Categorical columns retain original ordinal codes. The source-prepared rows form the full universe; main measurements impose no further cap.

Cutoff fractions `.4,.55,.7` are predeclared and floored to whole domain counts. The context has source rows minus `max(1,floor(.1*domain_rows))` randomly held-out ID rows per source domain. Every future row is OOD test. Repetition seeds 0/1/2 are paired with baseline/drift checkpoint IDs 1/2/3 and feature RNG seeds 17/18/19. NoT2V has only checkpoint 1. These joint repetitions differ from a three-split × three-pretraining-initialization factorial experiment and cannot estimate each variance component separately. No class-coverage retries are performed; absent ID classes yield unavailable full-class AUC with counts reported.

All five arms share context/query IDs and numeric inference settings. `base_time` appends the clock as a feature; `base_no_time` omits it. `drift` passes time through the learned 100-coordinate path. `drift_zero` sets all times to zero in the same drift model. `noT2V` uses a separately pretrained scalar-time model. Temperature 1, one raw numeric view, fp32 and no preprocessing search are common diagnostic choices. The source's all-row constant compaction/active-count behavior is preserved and exposed, while means/std/time endpoints are source-fitted.

Each run saves complete probabilities/targets, original row IDs, query domains, per-domain errors, model weights/runtime identities, source digests and the recursive live-code identity. The identity traverses nested CodeTypes and global helpers, stable function-valued and numeric defaults, model methods/settings and actual weights. It rejects hooks and instance-method overrides. Fresh-process and hidden-generator-helper mutations were checked before final measurement. The source checker and notebook identity remain separate objects.

The reconstructed SCM diagnostic uses four causal variables expanded to eight scalar nodes, twelve functional edges and an explicit causal-edge map. A five-hidden/four-hidden nonlinear H emits twelve correlated shifts; only selected causal relationships update G weights. Saved traces include all H parameters, per-domain weights, first-row node values, class fractions and actual future checkpoint predictions for three seeds. Exact original prior topology/activation/noise/hyperparameter sampling is unavailable and is not claimed.

## Original-paper route and unresolved gaps

The original [released demo](https://github.com/automl/Drift-Resilient_TabPFN/blob/a6e75afb82d13e7abb46deade3669c4106b3d636/Drift_Resilient_TabPFN_Demo.ipynb) contains executable full evaluator calls. To prepare it in an isolated environment:

```bash
git clone https://github.com/automl/Drift-Resilient_TabPFN.git /tmp/l068-paper-reproduction
git -C /tmp/l068-paper-reproduction checkout a6e75afb82d13e7abb46deade3669c4106b3d636
```

Install that checkout and its `requirements-release.txt`/`requirements-experiments.txt` in a separate environment following its demo installation cells; do not replace the course environment with historical dependencies. Run its dataset preparation, model-loading and evaluator cells. The model loader explicitly pairs `tabpfn_dist_model_{1,2,3}` with `best_dist`, `tabpfn_base_model_{1,2,3}` with `best_base`, and the available NoT2V checkpoint with `best_dist`. Its evaluation cell calls `evaluate_and_score(..., num_splits=3, metric_with_model=partial(transformer_metric,classifier=model), eval_kwargs=get_eval_kwargs(setting), device='cuda', save=True, overwrite=False)` for each released checkpoint and baseline setting. That original notebook is a prepared route; it has NOT_RUN as a full benchmark in this repair.

Original `DistributionShiftDataset.generate_valid_split` draws distinct eligible cutoffs satisfying 30–80% of both domain and sample counts, with class checks/retries and source-domain 10% ID holdouts. Retain its selected split identities across methods. The test roster is 18 datasets (8 synthetic/10 real), with a separate 12-dataset preprocessing-validation roster. Reconcile source data versions, loader decisions and original score aggregation before claiming table correspondence. Age groups/house construction years are ordered proxies and should not be described as ordinary future deployment periods without qualification.

`best_dist` applies robust preprocessing with original features appended, numeric categories, fingerprints, outlier setting 7, 32 feature/class-shuffled views, 99% context subsampling, temperature .9, probability averaging and fp16. `best_base` uses safe-power preprocessing with original features, one-hot categories, 90% feature subsampling, SVD, outlier setting 9 and temperature .75. Preserve these **different validation-selected packages** when targeting published results. The matched course wrapper deliberately omits them. Tree and Wild-Time baselines require their original 1,200-second per-dataset/split HPO budget and three initializations; the demo contains those calls. The paper's Figure 10 uses Wilcoxon-Holm, unlike our exploratory three-real-dataset Friedman/Nemenyi calculation.

Reproducing synthetic pretraining additionally needs the exact original prior generator and sampling distributions, optimizer/schedule, 30 epochs and 30,720,000 synthetic tasks, original random streams and all relevant software/hardware settings. The checked release has no pretraining-prior sampler; replacing it with our small declared reconstruction cannot reproduce that distribution. The paper describes eight GPUs and multiple days per pretraining run. None was launched here.

## Evidence ledger

- **Verified here:** strict all-tensor original checkpoint loading; 18 source full-forward fixtures and every block; exact raw dataset preparation; live sparse SCM data-generation intervention; complete four-task all-row prediction panel; separate actual boundary-grid predictions; dataset-level summaries/ranks/bootstrap; current-kernel source checks and an executed teacher EXIT (execution report provides its exact path/hash).
- **Paper claim, cited:** Table 1/2 accuracy/AUC/calibration, 18-task benchmark, original synthetic training and timing. **INCOMPARABLE** to the local wrapper. The paper's NoT2V difference is statistically insignificant, not a universal statement that Time2Vec has no effect.
- **Broader work:** original optimized evaluator and full benchmark/pretraining **NOT_RUN**. No original-result reproduction is claimed. Browser/copied Pages/publication and live Colab have separate statuses in the review; source parity does not establish them.

`_verify_l068_v2_capped_results.json` and `_verify_l068_v2_rounded_results.json` are immutable pilots with exact archived operators `sources/l068-v2/capped_operator.py` and `rounded_scale_operator.py`. The latter used the legacy rounded attention scalar; final source checks established the active Torch SDPA branch uses exact inverse-square-root scaling, so the full panel was rerun. The pilots are not the final measurements. Historical pre-v2 files are also retained unchanged.
