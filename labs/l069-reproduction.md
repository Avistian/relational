# L069: executable protocol and reproduction boundary

[Lesson](../lessons/0069-tabpfn-open-environment-failures.html) · [Reference](../reference/0069-tabpfn-open-environment-failures.html) · [Paper v1](https://arxiv.org/html/2505.16226v1) · [Source snapshot manifest](sources/l069-v2/manifest.json).

## Supported local experiments

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l069_v2.py \
  --preset smoke --output /tmp/l069-smoke-fresh.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l069_v2.py \
  --preset closer --output /tmp/l069-closer-fresh.json
```

Destinations must be fresh; existing measured artifacts are not overwritten. `smoke` uses a stratified cap of 120 Iris rows and one split. `lab` uses full CMC and Iris at seed 42. `closer` is the actual full local author panel described below. All presets additionally run the generated IID/covariate/concept family. In each saved artifact, `config` identifies the executed dataset roster, seeds and cap; `protocol.seeds` retains the broader three-seed declaration, so use `config.seeds` and the actual record roster when interpreting a smoke or lab run. There is no resume claim: an interrupted run must be restarted, and unfinished stdout is not accepted evidence. No paid/cloud operator is needed for the supported CPU experiment.

The notebook's optional `RUN_BROADER` gate runs this exact `closer` protocol through the **live notebook namespace**, preserving six student functions and the visible full model. The default gate is off. A fresh notebook output is `labs/data/cache/l069-student/<run-id>/exit-v2.json`. The ignored executed teacher is under `labs/solutions/0069-tabpfn-open-environment-failures.ipynb`; the committed notebook contains blank student implementations.

## Freeze the scientific scope before scores

| Component | Declared local protocol |
|---|---|
| Novelty datasets | Complete released CMC 1,473×9, red Wine 1,599×11, white Wine 4,898×11; 3/6/7 classes; original CSV digests pinned |
| Novelty split | Every class held out in turn; all its rows novel; equal known query sample; every other known row context; disjoint IDs and complete row union |
| Natural-prevalence task | Separate stratified 80/20 split, class 0 removed only from context; all test rows scored; excluded training IDs retained |
| Feature datasets | Complete Iris 150×4, CMC and red Wine; no row caps |
| Feature split | Non-stratified 80/20, fresh actual split seeds 42/2023/789 |
| Feature perturbation | Context mean replacement at six nominal levels 0/.2/.4/.6/.8/1; floor(level×F) columns from a seeded nested permutation; actual fraction retained |
| Additional feature control | All 15 nonempty Iris subsets plus clean at seed 42; all complete-model probabilities saved separately; actual added-column alignment predictions compared with baseline |
| Model | Full historical Nature-v2 classifier, all 81 checkpoint tensors / 7,244,554 parameters; 12 layers / width 192 / 6 heads / FF 768 / group 2 / max 10 classes |
| Model recipe | Single raw numeric view, inference seed 0, temperature .9; full declared query set in one call; original class map; no latest-package defaults |
| Comparator | XGBoost 3.3.0; 100 trees, depth 6, learning rate .3, hist, full row/column sampling, one thread, task split seed |
| Controlled distribution | 256 standard-normal source rows, 512 query rows, two features, threshold y=1[x0>0]; query x0+=1.5 for covariate shift, same X/opposite y for concept shift |
| Metrics | Continuous novelty ROC-AUC/AP and source binary-interval ROC-AUC/AP; all-row/known-only accuracy and clipped loss; clean/removed accuracy, balanced accuracy, macro F1, macro OVR AUC |
| Uncertainty unit | Equal class averages within each seed; seeds within dataset; separate dataset means; paired bootstrap resamples the three dataset units, not individual rows/classes/seeds |
| Undefined metric | Red feature seed 789 omits class 0 in test: macro OVR AUC=None; every metric reports available seed IDs/counts; other scores retain that split |

Exact class averaging follows the measured sklearn adapters: balanced accuracy averages recall over true-present test classes; macro F1 uses labels=None and averages the union of true and predicted labels. On red seed 789, absent class 0 can therefore enter macro F1 only if predicted. Macro OVR AUC requires every context class in test and is None on that split; per-metric seed counts are shown.

The generated covariate law changes the positive prevalence to Φ(1.5)≈.9332 in expectation, so accuracy changes do not isolate difficulty from class composition. Exact same-X label reversal is an information control, not evidence that one model family uniquely fails. Generated tasks do not enter the real-dataset bootstrap.

## Original network and source identities

Checkpoint URL: `https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/f851f2a3c941544733b712d8c0f96dfae9b28862/tabpfn-v2-classifier.ckpt`.

SHA256: `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`.

`relkit/tabpfn_l064_v2.py` remains unchanged at SHA256 `6f50292d3cb41597e217a4e6a7f71eb6bc120189f6295bd5975a7ec8dc75e747`. L069's `relkit/tabpfn_l069_v2.py` (SHA256 `8db9c27b03e794cb36a74eb80e644978f579465a79c45cb254355f045cf5e275`) retains that full architecture and strict checkpoint mapping. Its only network-operation change uses torch SDPA after flattening independent leading axes to four dimensions and broadcasting the shared KV heads. Scale remains the source's float32-rounded `sqrt(1/head_width)` scalar, including when weights are converted to float64. All context and query dimensions stay intact. This avoids the explicit attention matrix storage; it does not cap rows, shorten context, remove layers, change class support or split query sets.

`_model_check_l069_v2_results.json` records complete original 2.0.9 model logits, input gradients, all 81 parameter gradients, numeric wrapper fixtures, original four-view wrapper bridge and query-coupling probes. The default wrapper bridge is a parity check, not the wrapper used in local scores. Runtime state hashes bind actual module methods/settings/weights; hooks or instance forward overrides invalidate certification. The semantic code identity traverses nested generator code, global helpers, function-valued/numeric defaults and closures. `_identity_l069_v2_results.json` includes a fresh-process check and actual `PackedAttention.forward` mutation.

[Original model license](sources/foundation/v2-LICENSE). The measured evaluator SHA256 is `fb9d928b99c938181f0ab29d61d7ff06893933cd884de37ec0f8455cdb7aa17f`; its module kernel identity is `4cd8fb11f974772169930880c3475af33d362fa170989a17e71d21b41adf04b1`. These identify the new local run, not the historical result operator.

The released evaluation code is commit `744c010457f68284faa7ae6ded793a8b3f3e03a4` of [LAMDA-NeSy's repository](https://github.com/LAMDA-NeSy/Evaluation-on-Tabular-Model-in-Open-Environments/tree/744c010457f68284faa7ae6ded793a8b3f3e03a4). Its visible history starts June 2026; README describes v2.5 and requirements select 6.4.1. This is a current released snapshot, not proof of the May 2025 training/evaluation identity. Source files retain original bytes and attribution. Released CSVs download by exact commit with independent hashes.

## Current-source findings that change interpretation

- `newclass.py` truncates all features to integers; scores an inclusive confidence interval as a binary decision; calls average precision, not trapezoidal PR area; uses seed 42 only; overwrites each held class's result. Its final saved value is the last class. TALENT models also lose 10% context to validation while plain model branches use all remaining known rows.
- L069 preserves float values, all held-class results and real split seeds. It reports continuous confidence and binary decisions separately. The balanced query task and natural-prevalence task have different contexts and query mixtures; their difference is not an isolated prevalence intervention.
- `run_experiment.py` random/all enumerates every subset and concatenates repeated test rows. Numeric degree counts the target, so degree 1 requests F+1 of F features and fails. `least/most` uses test labels in correlation ranking and appends repeated references to one mutable frame. These paths are audited, not used as valid feature-selection protocols. The isolated source audit restores object string dtype for pandas 3 when exercising those historical branches.
- The local main feature protocol samples three nested label-blind paths, rather than enumerating every subset on every dataset. The separate complete Iris all-subset control is measured and saved. This is a deliberate procedure difference.
- `distribution_objectives.py` omits stratify in sampling, mutates the reduced cohort inside the seed loop, and uses California for the temporal setting where Table 10 says New York. Current source F1 is macro averaged; Table 11 BRFSS accuracy .875/F1 .044 cannot be binary macro F1 on the same predictions because macro F1≥a/(1+a)=.4667.
- Paper Table 1 contradicts the “wins all four novelty tasks” wording; red-wine MLP exceeds v2. Appendix F defines relative gap while Table 2 displays absolute differences. Table 3's average RF rank 3.93 is worse than v2's 3.53 despite broader prose. These are table-specific corrections, not claims that every historical experiment is false.

## What remains before an original-paper result claim

The seven original methods are RandomForest, XGBoost, CatBoost, MLP, RealMLP, ModernNCA and TabPFN v2. Appendix D specifies five-fold exhaustive tree searches and 100 neural Optuna trials with batch 1024; it reports three seeds. The local fixed two-arm experiment performs none of that optimization. Full original benchmark metrics and hardware-dependent costs remain **NOT_RUN**.

The original novelty suite additionally includes Eye Movements. Feature tests span 12 tasks: Credit, Electricity, Heart, MiniBooNE, Iris, Jannis, Penguins, Eye Movements, Abalone, Bike, Concrete and Laptop, including regression and categorical processing. Distribution/objective tests use 9 WhyShift/TableShift settings: ACS Income CA→PR, Mobility MS→HI, Public Coverage NE→LA, temporal Public Coverage 2010→2017, two young-population ACS Income settings, College Scorecard, BRFSS Diabetes and hospital readmission. Full source/target row identities, historical package/checkpoint wrapper, search selections, exact F1 convention, class aggregation and conflicting geography/counts must be reconciled before reproduction can be scored.

The current original command route, in a separately cloned source environment, is:

```bash
python newclass.py --dataset dataset/cmc/cmc.csv --model tabpfn --export_dataset false
python run_experiment.py --dataset iris --model TabPFN --task random --degree all --export_dataset true
```

These are **source entry points, not certified historical reproduction commands**. The wrapper `run.py` passes `--setting` to a novelty parser expecting `--dataset`; source dependencies also refer to TALENT. Before a historical rerun, pin the intended old package/checkpoint explicitly, repair or version the aggregation/metric defects, recover task identities, and reproduce the stated tuning. Merely installing the current requirements runs a different model version. The local `paper` configuration deliberately raises instead of silently pretending a resource budget closes these gaps.

## Regenerate and verify this package

```bash
.venv/bin/python labs/_check_l069_v2.py
.venv/bin/python labs/_paper_audit_l069_v2.py
.venv/bin/python labs/_identity_l069_v2.py
.venv/bin/python labs/_analyze_l069_v2.py
.venv/bin/python labs/_figures_l069_v2.py
.venv/bin/python labs/_build_l069.py
.venv/bin/python labs/_execute_l069.py
```

The feature control has its own fresh operator `_feature_control_l069_v2.py` and immutable `_feature_control_l069_v2_results.json`. Historical `_verify_l069_results.json`, historical foundation operators and old figures remain untouched. An aborted pre-correction evaluator and stdout are retained under `sources/l069-v2`; they are not measured result artifacts. No existing source hash is reassigned to new weights or predictions.

The prepared HTML is read-only. Inline PNG payloads support portable notebook display, but packaging/nbconvert/browser checks do not establish live Colab execution. Publication and copied-Pages/browser checks have separate reports. The review records what was actually checked; original paper results remain **INCOMPARABLE**, not reproduced.

The parent additionally checks the entire prediction panel through the pinned original TabPFN 2.0.9 implementation. After preparing the source CSVs and historical checkpoint, run these in that isolated original-package environment with the repository dependencies available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python labs/_reference_l069.py /tmp/l069-original-probabilities.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python labs/_cached_evidence_l069.py labs/_verify_l069_v2_results.json /tmp/l069-independent-check.json /tmp/l069-original-probabilities.json
```

The first command generates every declared reference case without reading lesson predictions. It requires a fresh output and preserves a per-case journal. The second independently rebuilds raw-data partitions and metrics, compares every v2 probability, and freshly fits every XGBoost case. Pass a fresh teacher EXIT path instead of the author JSON to check its smaller panel against the same reference. The parent reports separate float64 operator equality from bounded float32 inference differences; the checked attention adapter changes CPU scheduling while preserving complete context and all model layers.
