# L164 Griffin reproduction contract

Approved2026-10-01: **Table12 Others-2 SFT→rel-f1-driver-dnf versus no-pretrain**,512/4096labels ×5subset seeds42–46 ×2arms =20fits. AggregateUSD10 cap covers preparation, retries, seeds and validation. Scope is selected downstream release replay using pretrained weights; full pretraining and whole-paper benchmark NOT_RUN. Lesson delivery and experimental completion are separate.

## Immutable inputs

- Paper: https://arxiv.org/html/2505.05568v1 ; archived bytes and SHA256 in `sources/l164/source-ledger.json`.
- Official code: `yanxwb/Griffin`, commit `b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427`; source files retained unmodified under `sources/l164/upstream`, Apache2.0 license included.
- Processed data: `yamboo/Griffin_datasets_joint_v65`, revision `e0c54ceada75317b06f11f8dcda7aa8304fbb593`.
- Others-2 FULL checkpoint: `yamboo/Griffin_models`, revision `bd8c5be5130f34e7faa31099d0bd81d95d0aa995`, `others-2/FULL/model.safetensors`.
- `input-manifest.json` records106downloaded files and all hashes, including float encoder/decoder. Download size63,983,784bytes. All11F1 node types and all their rows/edges are retained. A derived metadata view excludes disconnected non-F1 components and other task loaders; raw metadata remains unchanged. This is component selection, not a within-task subsample.
- Task rows:11,411/566/702train/validation/test;12,679unique `(nodeidx,cutoff)` keys. Original order and label indices retained. Dataset has negative pre-1970 cutoffs; timestamp sign alone is not a missing-value test.

## Protocol audit

| Component | Required released setting / evidence |
|---|---|
| Model | D512,8attention heads,4message layers,forward+reverse,optional graph gates off |
| Initialization | Fixed model seed42; no-pretrain random initialization or strict Others-2 checkpoint loading |
| Subset | `RandomState(seed).permutation(train_rows)[:n]`, seeds42–46; same subset selected every epoch, shuffled batches |
| Sampling | hop2,fanout20perrelation,fewshotfanout3; original source sampler retained |
| Optimizer | AdamW,lr3e-4,weight_decay2e-4,default remaining optimizer parameters |
| Schedule | batch256,max200epochs,validation every2epochs,stop when epoch−best_epoch>10 |
| Selection | Highest validation score; strict improvement; no train+validation refit |
| Evaluation | Full566validation and702test rows; final best checkpoint; test also logged on each improvement in upstream code |
| Metric | Released2-class macro AUROC; independently recompute softmax label-index1AUROC, require absolute difference<1e-6 |
| Aggregation | All5subset runs per arm/size; report each score, mean/sampleSD and paired differences; no whole-benchmark inference |
| Numerical target | Table12 means: no-pretrain .6558/.7176; Others-2 .7098/.7275. A future complete release replay can use predeclared ±.02AUROC descriptive closeness, never as proof of historical identity |

**Source differences and limits:** The release first performs metadata self-attention, later task-conditioned attention with elementwise query modulation, residual feedforward paths and task-state updates; these extend the paper's compact equations. The notebook's visible model preserves checkpoint names and these operations. Tiny float64 output/gradient parity and real512-wide checkpoint logit parity passed. This does not establish training trajectory equivalence across PyTorch/CUDA versions.

Checkpoint `config.json` says `use_gate=true`, but all43weight tensors strictly match the transfer script's `use_gate=false`. Tensor compatibility and the executed transfer script define this lane; the stale configuration is recorded rather than trusted blindly.

The original trainer refers to `model.device`, absent on its nn.Module. Our wrapper adds a read-only property returning the first parameter's device. It wraps metric collection to retain labels/predictions and independent scoring; it does not alter model computation. Full training keeps source16loader workers and evaluation8workers; the timing probe deliberately uses0workers and sequential samples to measure resource demand. Full16worker throughput/prefetch memory and the full wrapper training path remain unexecuted. The probe uses torch2.5.1/cu124 on L4; local port checks use torch2.13.0 CPU. Transitive cloud environment was not fully frozen; selected direct pins are in `l164-requirements.txt` and the Modal image.

The main graph sampling audit checks1,494sampled edges with per-owner strict cutoffs, plus complete query-key uniqueness and separated cutoff ranges. The few-shot helper ignores timestamp and samples earlier row indices; a synthetic counterexample FAIL is retained. The selected F1 root table is static with sentinel timestamps earlier than allqueries and no task-label feature; its selected few-shot roots pass the narrow cutoff audit. This does not prove historical feature availability. Original preprocessing fit scope and exact label-SQL identity are NOT_ESTABLISHED from processed tensors. No current RelBench archive is silently substituted.

## Execution and stop decision

The L4 probe ran two256-query optimizer steps and all566validation queries. It did not evaluate test metrics, complete any fit, or retain a prediction model as a result. Its max observed allocatedGPUmemory was12,721,967,104bytes. Worker body19.345634seconds, estimatedUSD0.006683at reserved resource rates; this excludes build/startup/idle. No itemized invoice is available. Conservative budget accounting retainsUSD0.51816runtime reservation +USD0.35overhead =USD0.86816, even though execution finished early. USD2reserve remains untouched.

Allowed-schedule scenario:200epochs for all20fits,100full validations per fit, up to100test-on-improvement evaluations plus final test. Measured maximum train-batch elapsed3.355671seconds; measured full validation6.028422seconds. Test cost is estimated by702/566row scaling, not observed. Estimated raw computeUSD51.112833; ×1.25marginUSD63.891041 exceedsUSD7fit/check allowance. **STOP: INCOMPLETE_BUDGET_GATE; zero full fits; final test NOT_RUN.** Early stopping may reduce actual cost, and more loader workers may change throughput; neither was assumed to make20fits affordable.

`cost-decision.json` contains the arithmetic, assumptions and preserved stop. Managed full execution refuses while decision!=PROCEED and reserves every fit's maximum timeout before dispatch (10×3600seconds +10×16000seconds). Larger budget or a newly approved measured protocol is required; the lesson does not silently lower sample counts, seeds or validation coverage.

## Runnable lanes

From repository root:

```sh
# Local bounded preparation: downloads pinned files; no GPU or paid service.
.venv/bin/python labs/_prepare_l164.py /tmp/l164-release
OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l164.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l164.py
# Author managed probe: already executed; duplicate reservation is rejected.
~/.local/bin/modal run modal/l164_repro.py --phase pilot
# Complete20fit release lane: currently fails closed at the budget decision.
~/.local/bin/modal run modal/l164_repro.py --phase full
# Single explicit manual fit (not automatically budgeted; full protocol, no smoke preset):
OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l164.py --phase fit --root /tmp/l164-release --out /tmp/griffin-512-42 --arm others-2 --size 512 --seed 42
```

The notebook's default lane recomputes synthetic operators, the small four-layer model and cached real projected attention; it does not contact Modal or run a full fit. A separately gated post-EXIT cell can materialize the pinned release source/data and invoke the full20fit command grid on a learner GPU. Default OFF; runtime/budget acknowledgment is explicit. This manual path is not covered by the author's completed run receipt. Prepared notebook HTML is not live Colab verification.

## Evidence ledger

- Mechanism and source parity: PASS; details in `_verify_l164_results.json`.
- Released-data integrity/query/selected temporal audit: PASS with recorded limitations.
- Timed source training probe: COMPLETE_TIMING_ONLY.
- Selected20fit reproduction: INCOMPLETE_BUDGET_GATE, final test NOT_RUN.
- Full pretraining/whole benchmark: NOT_RUN; historical identity/transfer gain: NOT_ESTABLISHED.
- Author preparation is separate from learner mastery: PENDING_WRITTEN_DEFENSE.
- Live Colab/deployment: NOT_CHECKED. No push or deployment requested.
