# L064 historical TabPFN v2 reproduction contract

**Built with TabPFN.** [Prior Labs License](sources/foundation/v2-LICENSE). The visible implementation is an educational re-expression of the historical inference architecture and encoders, with explicit tensor operations, numerical boundary checks and a deliberately restricted numeric wrapper. Original checkpoint weights are downloaded unchanged and checksum-verified, never redistributed inside the notebook.

## What is actually reproduced

- **Full classifier architecture and weights:** 12 layers, 192 hidden coordinates, six 32-wide attention heads, two features per group, missingness encoders, 48→192 random group identities, context MHA/query first-KV-head MQA, three non-affine postnorm residual sublayers, GELU feed-forward and prediction head. All 81 parameter tensors/7,244,554 values participate.
- **Complete selected numeric recipe:** context-only raw constant selection, context label encoding, one untransformed view, no fingerprint/order/label permutations, no outlier compression, temperature 0.9, probability renormalization. This matches a supported explicitly configured `tabpfn==2.0.9` wrapper, not the paper default.
- **Actual default four-view bridge:** execute original default wrapper, capture its transformed tables, then recompute every network output with the live implementation and reconstruct class-aligned temperature-scaled probability averaging. This validates a real whole-wrapper connection. It borrows original peripheral preprocessing rather than claiming independent quantile/SVD/category/fingerprint implementations. The worker records actual model seed, temperature and outlier settings.
- **Fresh local inference:** every row of diabetes, blood transfusion and WDBC; three stratified 50/50 split seeds; same test rows for observed/shuffled context labels; saved input row IDs, context labels, targets, logits, probabilities, log loss, AUC, timing and versions. This is a mechanism intervention with a frozen pretrained model, not a new model-ranking benchmark.

## Identities

- Historical wheel: `tabpfn==2.0.9`, SHA256 `04e3bb989e9328d510ea4fccb6c6a36c8269630685d460af4df5eb268206bf21`.
- Classifier checkpoint: `tabpfn-v2-classifier.ckpt`, pinned Hugging Face revision `f851f2a3c941544733b712d8c0f96dfae9b28862`, SHA256 `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`.
- Visible core: `relkit/tabpfn_l064_v2.py`; the author measurement records its SHA. `_sources_l064_v2.json` records the primary paper/supplement bytes and historical source-file hashes/functions.
- CPU math matches source's float32-rounded inverse-sqrt head scale and active-group factor even in float64 tests. Raw double versus float32 precision may yield noticeably different outputs at the constructed zero-variance preprocessing boundary. No cross-precision equivalence is claimed.

## Commands from the repository root

Use the workspace environment; the visible model needs the usual lab dependencies, not a global current `tabpfn` install.

```bash
.venv/bin/python labs/_check_l064_v2.py
.venv/bin/python labs/_verify_l064_v2.py --preset closer --output /tmp/l064-fresh-panel.json
.venv/bin/python labs/_evidence_l064.py /tmp/l064-fresh-panel.json /tmp/l064-fresh-panel-check.json
.venv/bin/python labs/_build_l064.py
.venv/bin/python labs/_execute_l064.py
```

The verifier rejects an existing output path. There is no resume and no silent reuse of old lesson scores. `smoke` is a 120-row diabetes fixture; `lab` is full diabetes at split seed 7; `closer` is the full three-dataset/three-seed author panel. There is deliberately no `paper` preset that falsely labels this subset a complete replication.

The source checker verifies/extracts the pinned wheel to `labs/data/cache/l064-source/official` and installs `scikit-learn==1.6.1` there without dependencies. It invokes a subprocess with that directory on PYTHONPATH, preserving the main notebook's installed sklearn. Its historical environment includes current author torch/numpy/pandas versions recorded in the manifest; author pandas 3 is outside the old package's declared pandas<3 range, so this is a documented environment deviation, not a recreated original training environment. For a separate historically compatible wrapper environment use `requirements-foundation-v2.txt`; never change the global tabpfn version mid-notebook.

Independent checks contributed by the parent audit can reuse the prepared historical source directory:

```bash
PYTHONPATH=labs/data/cache/l064-source/official OMP_NUM_THREADS=1 .venv/bin/python labs/_independent_l064.py
PYTHONPATH=labs/data/cache/l064-source/official OMP_NUM_THREADS=1 .venv/bin/python labs/_query_coupling_l064.py
```

`_independent_l064.py` reconstructs all layers with NumPy/SciPy without importing lesson operators. `_query_coupling_l064.py` directly tests original-release one-view, finite-constant, cached and default-four-view wrappers. `_evidence_l064.py` independently rebuilds raw data/splits/labels, softmax, negative log likelihood and rank-based AUC from saved evidence. These checks answer different questions.

## Student EXIT

The notebook inlines the complete model beside explanations. The five live functions are `group_features(x,group_size=2)`, `encode_targets(y,total_rows)`, `attention_mix(q,k,v)`, `row_attention(h,n,attention)`, and `postnorm_update(h,update)`. Each is called during the actual full-data pretrained experiment. The saved kernel identity includes nested semantic bytecode, closures, model methods, defaults, configuration, recipe, versions, data-loader source/global configuration and checker dependencies. A model-state hash additionally detects weight edits.

A fresh run is written to `labs/data/cache/l064-student/run-<time_ns>.json`; the latest receipt is `labs/data/cache/l064-student/student-l064-v2-exit.json`. The receipt binds run bytes with `run_sha256`. EXIT rejects stale operator/default/config/dependency/weight edits. There is no claim that the student's blank notebook has been executed. Only the ignored teacher notebook has complete functions and measured outputs.

## Source findings that constrain interpretation

The release's wrapper keeps a partly missing constant feature because NaN fails raw equality. Imputation can make its main channel constant; an appended query can change the uncached encoder's all-row active-value count from one to two. Existing encoded inputs then change before context-only attention. The simple float32 path changes the first class-1 probability by about 0.014825; the independent original float64 one-view path changes it by about 0.011651. The independent default four-view float64 path changes it by about 0.007824. Fixed-token and finite-constant controls preserve isolation. The cached fixture remains independent of the extra query but has a different baseline, so cached/uncached equality is not claimed.

The Methods' row identifier description differs from release2.0.9's salted row-byte hash; the code's outlier operation compresses tails rather than deleting samples. Default category transformations, view ordering, SVD and fingerprints must be read from the pinned source, not inferred from one architectural figure.

## What a closer paper replication still requires

The main Nature result used 29 classification and 28 regression datasets, ten official OpenML 90/10 splits, appropriate classification/regression checkpoints, four/eight default views, tuning with five-fold CV, fixed time budgets and specified CPU/GPU hardware. The present panel uses the paper's blood-transfusion dataset and diabetes/WDBC tasks also reported in supplementary benchmark results; it does not represent three draws from the 29-task main roster. WDBC comes from sklearn's same UCI source rather than recovered OpenML task rows. None of the tuned competing methods or the normalized complete-pool paper average is reproduced here.

A full replication must recover those exact tasks/splits, reproduce every preprocessing view and comparator budget, freeze validation choices before scoring, and count fit/preprocessing/cache/predict timing consistently. Extend the **same visible model** as default views are independently implemented and checked; do not substitute another package predictor and relabel it the student's mechanism. Regression requires its separate learned head/checkpoint and transformed distribution borders. The subset runner is a concrete closer inference track; the complete benchmark remains **NOT_RUN / INCOMPARABLE**.

Original synthetic pretraining also remains **NOT_RUN**. The paper reports roughly 2M optimizer steps×64 datasets and did not release its complete synthetic generator. A random axial encoder trained on a simplified SCM is not a substitute reproduction of that procedure. No Modal or other paid/cloud run was launched. Live Colab and publication/browser checks are tracked separately by the parent delivery audit.
