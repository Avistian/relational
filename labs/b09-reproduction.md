# B09 reproduction ledger

## Approved scope and experiment

B09-MATCHED-COST: current TabFM base, EXAONE regression and Nori-6M release inference on raw sklearn diabetes, seeds0/1/2, 80/20 support/query splits. Nine declared model/split runs, no silent reduction. No gradient training. Data packets preserve all442 rows, ten features and full353/89 split identities.

Operating point: CPU float32, four Torch and BLAS threads, fresh process per model/split. TabFM base explicitly uses one estimator, no normalization view ensemble, no feature shuffle/cross/SVD/NNLS and no support cache. EXAONE retains current release defaults including8 estimators and SVD split behavior, but uses float32 instead of default float16. Nori retains wrapper defaults and internal predictor seed0; outer NumPy/Torch RNG and splits use0/1/2. Nori's internal seed is not claimed paired with other wrappers. Native preprocessing may differ; Nori includes unlabeled query-feature distribution information. This is a matched-input system comparison, not an equal-preprocessing architecture ablation.

Download excluded from load timing but charged to preparation. Model import/construction/deserialization is load; Python/numpy/torch startup precedes that interval and remains in process wall time. Some wrappers lazily load at fit/predict, so interval names do not imply identical work. First prediction separate, two warmups, ten timed calls. Save the first predictions and record maximum difference across repeated calls. Warm median/P90 are within-process descriptive measurements on89 queries. Peak RSS is absolute process high-water memory (MiB), not incremental tensors or GPU memory. This is a shared workstation, not a dedicated timing appliance.

## Source and checkpoint identity

See `sources/b09/provenance.json`, per-model commit/HF metadata, `weights.json`, `tensor-inventory.json`, `environment.json`, config copies and archives. Weight digests are checked against immutable Hugging Face LFS SHA256 records. External weights stay in `/tmp/b09-weights` and are not site artifacts.

The current TabFM regression file contains1,647,783,213 tensor elements (including any stored buffers),6,591,243,724 bytes; its config uses width256. This is not automatically the report's400M configuration. EXAONE's selected regression file contains21,111,678 tensor elements; parameter and buffer counts need distinction. Nori-6M is not the Nori-30M entry in the current EXAONE comparison table. Current source/runtime identities do not establish original paper identities.

## Execution and budget

Machine: ARM64 WSL, CPU only. Local paid-service spend USD0. USD10 aggregate cap, USD8 planned plus2 reserve. Numerical execution cap3600seconds. `evidence/b09/budget.json` contains all fresh-run attempts and conservative preparation/validation allowances. Source/weight preparation and ARM dependency compilation are included. Three initial readiness failures (missing/not-yet-complete weights and Nori dependency) remain recorded. Never reinterpret an incomplete arm as a smaller completed protocol.

Read `evidence/b09/course-audit.json` for exact completed/expected matrix, keyed independent rescoring and per-seed records. The notebook executes saved-evidence replay and three primitives; it does not rerun checkpoints. Author validation is separate from learner PENDING_WRITTEN_DEFENSE.

## Fresh-inference operator

From repository root:

```bash
.venv/bin/python labs/_prepare_b09.py
.venv/bin/python labs/_download_b09.py
# Install pinned inference dependencies from sources/b09/environment.json
# into /tmp/b09-deps, or use a separate environment with those versions.
.venv/bin/python labs/_run_b09.py
.venv/bin/python labs/_audit_b09.py
.venv/bin/python labs/_build_b09.py
.venv/bin/python labs/_execute_b09.py
```

The preparation operator extracts the three pinned code archives to `/tmp/b09-src`. `synthefy` is taken from Nori's pinned `libs/synthefy/src` subtree, not a paid API. On ARM, pyvinecopulib1.0.0 needs Eigen3 and Boost development headers; this run built it with CMAKE_BUILD_PARALLEL_LEVEL=2 in an isolated target. Inspect source licenses independently from weight licenses. These educational runs do not establish commercial deployment permission.

Do not run concurrently with another cost benchmark. Do not overwrite sealed successful results: use a new evidence directory for an explicitly budgeted rerun. Existing completed jobs are resume artifacts, not new inference. Budget exhaustion stops remaining jobs. The default runner uses the full declared protocol, not a smoke/downscaled replacement.

## Published target — INCOMPLETE_SOURCE_PROTOCOL

EXAONE2608.25774v1 Figure3(b), regression accuracy–latency frontier. Missing authenticated publication-era checkpoints/code per plotted variant; complete original fold identities/preprocessing; hardware/precision/batch/warmup/cache details per method; original baseline timing records and complete Elo pool; matching regression-only aggregation. Baseline timing provenance in the report is TabArena, not our CPU process. `evidence/b09/paper-gate.json` is the machine-readable gate.

```bash
.venv/bin/python labs/_reproduce_b09.py
.venv/bin/python labs/_reproduce_b09.py --execute
```

The second command refuses execution while gaps remain. This is a runnable audit gate, not a runnable complete historical reproducer. Full suite, whole-paper reproduction, pretraining, TabFM+/Auto execution, proprietary Seldon/NEXUS access and deployment are NOT_RUN. Source visibility and a passed saved-evidence notebook cannot change those labels.

## Delivery

See `_execution_b09_results.json`, `_delivery_b09_results.json` and `_checkout_b09_results.json` for actual notebook/browser/copied-build checks. Live Colab NOT_CHECKED; deployment NOT_RUN. Source HTML, notebook builder, source appendix and evidence packet are kept together. A prepared lesson does not record learner mastery.

### Resource preflight

The unmodified TabFM loader constructs a float32 model and maps checkpoint tensors. We conservatively reserve twice the6.59GB checkpoint size plus1GiB runtime overhead before dispatch. If this exceeds available physical memory, its three arms remain INCOMPLETE_RESOURCE_GATE. This is a conservative scheduling decision, not a measured OOM or proof that every optimized loader would fail. We do not substitute a smaller checkpoint, reduced precision, reduced support, or new hardware for the approved CPU operating point.

Nori's file contains both model_state_dict and ema_state_dict (5,868,069 tensor elements each). The pinned release loader prefers EMA weights and reads the embedded model_config. The sidecar config's stale classification/file-name fields are not used as the architecture authority. Native RMSNorm and other current runtime optimizations remain part of this release operating point.

### Tested dependency boundary

The exact tested versions are in `sources/b09/requirements-inference.txt`. This run imports pinned source trees with an isolated `/tmp/b09-deps` overlay. It does not claim that every upstream package's declared installation constraints are satisfied: Nori declares huggingface-hub>=1, whereas the tested local inference environment has0.36.2. No Hub API is used inside the timed Nori path because weights are local. Import success is not a general compatibility guarantee. The portable exercise notebook uses only NumPy and embedded evidence; heavy inference dependencies belong to the separate lane.

### What the source audit did recover

The three current code archives and weight revisions are available and pinned. EXAONE publishes a standalone inference runtime and current checkpoint presets; its package metadata notes that its evaluation harness is unshipped. TabFM includes examples and published result artifacts; Nori includes an evaluation subsystem and benchmark lists. `sources/b09/evaluation-inventory.json` records those paths. These findings support a current-release operator, but do not authenticate the joint historical Figure3 pool and timing protocol. The paper explicitly reuses TabArena baseline measurements. A fresh run on our CPU cannot be compared numerically to those published timing points without a new matched experiment.

On an existing course environment matching `environment.json`, the exact overlay command is:

```bash
.venv/bin/python -m pip install --target /tmp/b09-deps --no-deps -r labs/sources/b09/requirements-overlay.txt
.venv/bin/python labs/_preflight_b09.py
```

If pyvinecopulib requires a source build on ARM, provide Eigen3 and Boost>=1.75 development headers and set CMAKE_PREFIX_PATH to their extracted prefix. The live session used Ubuntu libeigen3-dev3.4.0 and libboost1.83-dev with CMAKE_BUILD_PARALLEL_LEVEL=2. Installation and model downloads are preparation work, not a measured prediction call.


## Final author evidence

6/9 declared runs completed: EXAONE and Nori, all three seeds each;267 query predictions per model,534 total. TabFM arms remain INCOMPLETE_RESOURCE_GATE. Historical Figure3(b) remains INCOMPLETE_SOURCE_PROTOCOL. All six complete runs repeat predictions exactly across ten timed calls.

exaone: RMSE 56.470 ± 1.311 (sample SD across three splits); median of per-split warm medians 36.120s per89rows; maximum observed process peakRSS 3605.3MiB.

nori: RMSE 56.938 ± 1.046 (sample SD across three splits); median of per-split warm medians 5.949s per89rows; maximum observed process peakRSS 711.1MiB.

Conservative aggregate accounting: 3096.876/3600seconds, including1200seconds preparation allowance and180seconds validation reserve. Paid services USD0. The allowance/reserve is not a measured hardware cost; raw timed attempts remain separately recorded.
