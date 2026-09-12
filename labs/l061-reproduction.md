# L061 v2 — fixed-GP posterior approximation

[Lesson](../lessons/0061-prior-data-fitted-networks.html) · [Student notebook](0061-prior-data-fitted-networks.ipynb) · [Read-only preview](html/0061-prior-data-fitted-networks.html) · [Reference](../reference/0061-prior-data-fitted-networks.html) · [Source inventory](_sources_l061_v2.json).

The v2 package implements the **original row Transformer and full-support Riemann head from scratch**, including the source's query-self diagonal, postnorm GELU residual blocks, zero initial residual output projections and context-only labels. It trains on sampled GP query targets, not analytic posterior targets. CountPFN and its historical `_verify_l061_results.json` are retained unchanged and are not relabeled as this model.

## Run locally

From the repository root, with requirements-labs.txt installed:

```bash
.venv/bin/python labs/_check_l061_v2.py
.venv/bin/python labs/_verify_l061_v2.py --preset smoke --output labs/data/cache/l061-smoke-new.json
.venv/bin/python labs/_verify_l061_v2.py --preset lab --output labs/data/cache/l061-lab-new.json
.venv/bin/python labs/_analysis_l061_v2.py
.venv/bin/python labs/_figures_l061_v2.py
.venv/bin/python labs/_build_l061.py
.venv/bin/python labs/_execute_l061.py
```

Choose a **fresh output name** for each fit; existing outputs raise an error. There is no resume. `_analysis_l061_v2.py` reanalyzes the committed author reference by design, while `_verify_l061_v2.py` can create separate reruns. The teacher solution is locally built and executed but remains gitignored; the published student notebook is blank. The notebook default performs both a 30-update wiring smoke and an actual 2,000-update seed 7 fit using the live student functions. EXIT defaults to `labs/data/cache/l061-student/exit.json` and uses a timestamped name if that already exists. It saves current functions, class methods, actual defaults, keyword defaults, preset/config values and versions separately from repository reference hashes.

## What was actually measured

The committed [v2 author evidence](_verify_l061_v2_results.json) has seeds 0/1/2, 2,000 updates per seed, batch 32, total 17 rows, contexts sampled with weights proportional to 1/(17−n), one-dimensional inputs, width 64, 3 blocks, 4 heads, FFN/head hidden width 128 and 64 bins. Adam uses LR 0.0003, gradient clipping at 1 and a per-step 10% warmup/cosine schedule. The final weights are evaluated; no candidate or epoch selection uses the test panel.

The prior matches Appendix F's fixed RBF-GP parameters: signal variance 1, length scale .6, observation-noise variance .0001. A second evaluation regime changes only the true task length scale to .1. The model stays frozen. Each context size 0/1/4/8/16/32 has 256 fresh test tasks, each with one query. Training task seeds are model seed+10000; test task seeds are 20000+n, plus 1000 for the shifted regime. Test streams are shared across model seeds. Border calibration uses 65,536 marginal normal draws with seed 731.

All x/y/logit arrays and GP moments are compressed with dtype/shape/SHA-256 in each audit batch. The result retains per-task metric values, per-seed traces and model-state hashes. Training-seed SD is over three fits to fixed tasks; the paired task t interval is conditional on these trained models and averages reused-task values before treating independent tasks as units. No real-dataset comparison or Friedman claim is made for this synthetic approximation study.

At matched context 16, PFN NLL is −1.0189 versus exact GP −3.0532. In the shifted context 16 panel PFN NLL is 4.4927 with central 95% coverage .3932. The [analytic-mass diagnostic](_analysis_l061_v2_results.json) supplies exact GP region masses to the same fixed Riemann head: matched context 16 NLL is −2.2763. This exposes head representation error separately from imperfect learned masses. It is not a trained predictor, and finite-sample gaps are not universal bounds.

## Source validation is a separate claim

The [original paper repository](https://github.com/automl/TransformersCanDoBayesianInference/tree/9c20031b355923bdd456d5fcfe4e98092b016b97) is pinned at commit `9c20031b355923bdd456d5fcfe4e98092b016b97`. `_fetch_l061_v2.py` fetches immutable source files into an isolated ignored cache. `_check_l061_v2.py` compares the whole live model with copied nonzero weights, so zero initial branches cannot conceal bugs. It checks float32 implicit-mask outputs, float64 explicit-mask outputs/input gradients/32 mapped parameter-gradient tensors; density values and logit gradients; normalization and CDF integration; empty context, context permutation, query packing independence and context-label sensitivity.

The float64 source test explicitly casts an otherwise identical mask to float64. The released default creates a float32 mask, which behaved differently in local mixed-dtype SDPA; production float32 is separately checked without this adapter. The released tail-scale helper computes its unit half-normal median in float32, accounting for submicro NLL differences from the float64 implementation. These adapters are recorded rather than represented as training parity. Neither original checkpoints nor pretrained TabPFN weights are used.

## Required scale-up: same code, explicit remaining gaps

The notebook includes an off-by-default `RUN_PAPER_REPRO` gate calling the same inlined trainer/evaluator. `closer` changes to five features, width 128, 6 blocks, 256 bins, 65 total rows, 10,000 updates per seed and 1,000 evaluation tasks per context; contexts include 128 as extrapolation. The `paper` preset intentionally raises: the full original protocol has not been implemented or run.

```bash
.venv/bin/python labs/_verify_l061_v2.py --preset closer --device cuda --output labs/data/cache/l061-closer-new.json
modal run --detach modal/l061_paper_repro.py --preset closer
```

The Modal wrapper has been syntax checked only. Its T4 GPU environment and the live Colab browser have **not** been run. Gaussian sampling is deliberately computed on CPU in float64 before transfer; do not interpret GPU selection as an optimized source-training runtime comparison. The historical shared `modal/foundation_repro.py` CPU route also remains available, but the scoped wrapper above identifies the current model and unique result path explicitly.

The immutable [GP experiment notebook](https://github.com/automl/TransformersCanDoBayesianInference/blob/9c20031b355923bdd456d5fcfe4e98092b016b97/notebooks/SetupForGPFittingExperiments.ipynb), five-feature training cell, specifies width 512, 6 layers, 4 heads, 1,000 or 10,000 bars, two million prior target samples for boundaries, total sequence length 2,010 and context split up to 2,000. It explores long epoch/LR/batch combinations. One loop variable named `nhid` is not passed into its `train` call; the pinned trainer's default is 200. Do not convert the unused loop variable into an asserted executed FFN width. The notebook also contains longer training/control settings and gradient-accumulation variants; exact original result-to-run correspondence remains unresolved. Its evaluation uses 1,000 fresh datasets per context and noisy-target NLL.

The v2 `closer` run is therefore an executable intermediate experiment, not a reproduction of Figure 4. Further alignment requires the original long-context/border/model/training settings, an explicit chosen original configuration with resolved defaults, old dependency behavior, exact evaluation sizes and a result comparison with stated uncertainty. Retain `INCOMPARABLE` until those conditions are met. Hyperprior GP, BNN, real-table and Omniglot experiments are not selected targets of this lab. In particular, Table 1's tabular speed claims and calibration rankings are not measured here.

## Three-bucket conclusion ledger

| Bucket | Current status | What it means |
|---|---|---|
| Verified here: model computation | PASS | Original-style row model and full-support density match the checked source computation within stated tolerances. |
| Verified here: learned GP approximation | MEASURED_WITH_GAPS | Fresh local learning and prior mismatch are measured, with exact and fixed-head oracles. |
| Paper claim | CITED | Read the fixed-GP study §5.1/F and original code; full Figure 4 results are not replicated. |
| Larger scale-up | NOT_RUN | Both Colab and scoped Modal operators are provided and remain off/unrun. |
| Original paper result match | INCOMPARABLE | Architecture capacity, training scale and experimental protocol still differ. |
