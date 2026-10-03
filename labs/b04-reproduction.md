# B04 reproduction contract

Approved target **B04-TABICLV2-FIG3-ATTENTION-FADING**. Status **INCOMPLETE_SOURCE_PROTOCOL**, zero Figure 3 benchmark runs. Separate course diagnostic **COMPLETE**, six configurations / 384 predictions / nine predict calls. No pretraining, full benchmark, historical identity, million-row scaling or learner mastery claim.

## Paper lane and exact blockers

[TabICLv2 v1 Figure 3](https://arxiv.org/html/2602.11139v1#S3.F3) compares no scalable softmax, SSMax and QASSMax on a two-dimensional synthetic task: four negative clusters, one support anchor and 20 fixed queries near it. Entropy is normalized by log(context size) and reduced over ICL heads/layers. Figure assets are archived for inspection, not digitized as exact raw references.

Pinned upstream `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3` contains 153 archived files. The complete tree contains no needle/haystack generator reference. Author HF release `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` lists v1, v1.1, v2 classifier and v2 regressor, not three authenticated Figure 3 ablation checkpoints. The paper, both figure assets, both author model inventories and two GitHub issue pages (165 entries) were inspected. This is a bounded search, not proof that artifacts exist nowhere. Missing: exact generator parameters/seeds/ordered input rows, full original negative-count grid, all three checkpoint-to-figure identities and exact numeric reference/entropy implementation. A production QASSMax checkpoint does not authenticate the other ablations; disabling its module is an inference intervention, not matched pretraining.

The paper v1 describes cautious weight decay; the pinned README says released checkpoints used `use_cautious_wd=False`. Preserve that discrepancy. Current release includes prior and pretraining code, whereas paper v1 describes those as forthcoming. Current access does not establish historical identity.

```bash
.venv/bin/python labs/_reproduce_b04.py
.venv/bin/python labs/_reproduce_b04.py --run
```

First command reconstructs the source gate; `--run` exits nonzero before inference. This is an executable preflight, **not a complete working Figure 3 runner**. To unblock, authenticate the missing identities, freeze reference values and tolerances, implement original dispatch/entropy hooks, then forecast the complete grid before running it. No override or random substitute is provided. Full source archive is retained; no claim of a finished paper implementation is made.

## Separate diagnostic: frozen before execution

`labs/evidence/b04/diagnostic-protocol.json` fixes every configuration. `inputs.json` holds all 569 sklearn breast-cancer rows, original row IDs/labels, ordered support pool, 64 fixed query IDs and one missingness mask. Input SHA256 `62e65575a393eb9345241923932ebf8718b588996530408cea0320f29e23e614`. Stratified 64-query split seed 0; remaining rows reordered with NumPy default_rng(4). Nested prefixes 32/128/384; no score-based selection. This is an educational dataset diagnostic, not clinical evaluation.

Checkpoint `tabicl-classifier-v2-20260212.ckpt`, author HF revision above, SHA256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`, 110,368,038 bytes. Worker imports the archived source and verifies every source byte against the tarball. Exact environment is recorded in each result. CPU float32, one thread, no AMP, no FA3, no KV cache/offload, one estimator, no feature/class shuffle, normalization option `none`, temperature .9, seed0. `none` disables the optional normalization family; the wrapper still removes constant features, clips outliers and standardizes. One estimator differs from the release default eight: declared course protocol, not reduced paper reproduction.

Six configurations: clean support32/128/384; support128 queries split into four batches of16; support128 with15% independent missing entries using NumPy default_rng(404), mean filling; same corrupted inputs/means plus one missingness flag for every original feature. Means fit only on the selected support rows; all-missing columns fill0; query labels never enter the model function. Batching comparison tolerance atol1e-6, rtol0. Indicators give60 submitted columns; upstream filtering may remove constant flags. This changes input dimensionality and content, not just compute. Missingness is MCAR by construction, not a shift test. Categorical missingness is untested.

Per-configuration fresh processes report one fit and prediction sequence and Linux peak process RSS, including imports/model/data. Timings are single observations without warm repeats or cache controls; they do not reproduce paper speedups. CPU RSS is not GPU allocation or attention-tensor memory. Source-level query-label exclusion and equality on this batching intervention do not prove universal information isolation.

## Results and independent reconstruction

| Configuration | Correct/64 | Log loss | Prediction seconds | Peak process RSS MiB |
|---|---:|---:|---:|---:|
| clean32 |61|.174837|.505|664.58|
| clean128 |63|.095919|.609|677.20|
| clean384 |63|.044251|1.236|680.25|
| batch128 |63|.095919|1.975|677.64|
| mean128 |62|.132061|.652|674.66|
| indicator128 |62|.127319|1.135|693.95|

Batch maximum probability difference0. Indicators reduce this split's log loss slightly but increase Brier score (.029792 to .030223); accuracy stays fixed. Neither representation wins on every measure. Larger support improved log loss here; no monotonicity guarantee or million-row extrapolation follows.

`_audit_b04.py` uses scalar arithmetic, authenticates row identities/support pairing/means/class columns and reconstructs accuracy/log loss/Brier for all384 predictions. `_verify_b04.py` rejects ten raw-record corruptions. The random-weight QASSMax equation check covers outputs and query gradients at2/32/384/15001 keys; it is source parity of query scaling only. The analytical softmax toy changes scaling with fixed keys/values; it is not a trained QASSMax benchmark.

## Commands, portability and budget

Offline replay and learner checks:

```bash
.venv/bin/python labs/_budget_b04.py .venv/bin/python labs/_verify_b04.py
.venv/bin/python labs/_budget_b04.py .venv/bin/python labs/_test_b04.py
```

Portable student/solution notebooks need NumPy, use embedded immutable evidence and execute from an empty directory. The archive includes complete upstream source and all diagnostic inputs/results, but excludes the110MB checkpoint. For a fresh run: extract `sources/b04/upstream.tar.gz` under `sources/b04/upstream` with its first path component stripped, create an isolated environment matching recorded package versions, fetch the checkpoint URL in `checkpoint.json` and verify its hash, use a separate copy of the packet with result files removed, then run each exact configuration with the worker. The worker refuses to overwrite existing results. Example:

```bash
python labs/_budget_b04.py python labs/_diagnostic_b04.py clean-32 /absolute/path/to/tabicl-classifier-v2-20260212.ckpt
```

Repeat the remaining five names from the frozen protocol only after the complete forecast fits. Score with `_verify_b04.py`; new floating-point results need a new report, not an overwritten author archive. Full fresh inference requires PyTorch, scikit-learn, NumPy, SciPy, einops, psutil, tqdm and huggingface-hub as pinned in `environment.json`; offline replay does not.

Approved USD10 aggregate cap with USD2 reserved; no new paid reservations above USD8. Actual cloud/API spend USD0; no paid worker used. Local3600-second cap enforced by `_budget_b04.py`, including failures/retries. Initial preparation allowance30s; pilot-based remaining forecast196.24s admitted before the five remaining runs. Final ledger includes author checking/preparation allowances. No hidden cloud costs. Live Colab and deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE.
