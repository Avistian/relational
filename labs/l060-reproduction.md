# L060 corrected comparison and reproduction contract

[Lesson](../lessons/0060-broad-model-comparison.html) · [Runnable notebook](0060-broad-model-comparison.ipynb) · [Read-only preview](html/0060-broad-model-comparison.html) · [Source inventory](_sources_l060_v2.json)

## Three distinct evidence tracks

- **Historical:** `_verify_l060_results.json`, 210 selected records. The archived operator and `relkit/tabm.py` are unchanged. Historical TabM-mini had extra adapters/biases and incorrect fan-in initialization. Its scores cannot stand in for corrected TabM. Its last invocation time is a resumed tail, not the total experiment cost.
- **Corrected local:** `_verify_l060_v2_results.json` contains 210 newly measured selected predictions from `relkit/checkpoint_l060_v2.py` using `relkit/tabm_v2.py`. Full IDs, data/dependency hashes, class order, target values, candidate losses, per-candidate epoch choices and timings are retained. `_analysis_l060_v2_results.json` reconstructs these predictions with explicit expected dataset/arm/seed coverage, seed-first ranks, paired effects and conditional permutation calibration.
- **Paper:** TabArena v1 benchmark and TabReD v4 results remain **INCOMPARABLE / NOT_REPRODUCED**. The local fixes and rerun do not turn the course convenience sample into either original benchmark. The already inspected dataset splits are not a new independent confirmatory test population.

## Exact local protocol

Eight public datasets: diabetes, blood_transfusion, kc1, phoneme, credit_g, churn, bank_marketing, adult. Sample at most 900 rows with label-blind RNG 60; stratified first split RNG 60 and second split RNG 61 give 60/20/20 proportions. TabReD Ecom Offers, Homesite Insurance and Sberbank Housing add released random-0 and sliding-window-0 pairs, sampled at caps 540/180/180 using RNG 550/551/552. All IDs are saved. This makes 14 dataset/regime cells, 11 underlying datasets, five arms and three seeds (0/1/2).

The five arms use two candidates each. Trees have 100 fixed rounds; neural models have three width-64 hidden layers, 24 full epochs and validation-selected checkpoints. First minimum candidate wins; last minimum neural epoch wins. No train+validation refit. MLP/TabM use AdamW learning rates .001/.003, decay 1e-4 and dropout .1; TabM-mini has eight members and shared batches. Reduced RealMLP-TD-S uses original simplified mechanisms, a compressed full coslog4 schedule and classification label smoothing .1. Numeric medians and one-hot vocabulary are training-only; all-missing numeric training columns use zero. TabReD categorical columns are omitted and counted. CatBoost receives common one-hot inputs, which removes its native categorical route.

Binary metric: mean log loss of P(y=1), finite inputs, float64-epsilon endpoint clipping. Regression metric: RMSE in the released target units. Sberbank is log(price_doc/full_sq), not currency. Released Sberbank train/validation time ranges have a timestamp tie; strict_time_boundaries remains false. Aggregate/label availability at the forecast date is not established solely by row disjointness.

Uncertainty: three model seeds conditional on fixed rows; paired per-task t95 intervals are approximate and unadjusted. Datasets, not seeds, are rank blocks. Random and temporal panels are separate; the latter has only three blocks. Asymptotic Friedman and Nemenyi values are exploratory. Conditional method-label permutation: 20,000 draws with plus-one correction on random tasks; exact 120² temporal enumeration. Exchangeability assumptions and lack of representative task sampling remain limitations.

Timing includes all candidate fits and validation plus selected prediction; common loading and median/one-hot encoding are excluded. The author run encountered variable host contention, so recorded CPU times cannot establish hardware-normalized efficiency or serving latency superiority.

## Run the corrected local operator

Use the repository's `.venv` with `requirements-labs.txt`. The installed versions are saved in the run. Download the released TabReD inputs with the established loader if not cached; it is an explicit data download, not a pretrained-model download.

```bash
.venv/bin/python labs/_verify_l060.py --preset smoke --output labs/data/cache/l060-smoke-new.json
.venv/bin/python labs/_fetch_l055.py
.venv/bin/python labs/_verify_l060.py --preset lab --output labs/data/cache/l060-lab-new.json
.venv/bin/python labs/_verify_l060.py --preset closer --output labs/data/cache/l060-closer-new.json
```

Every path must be new: this operator deliberately refuses overwrite/resume. An exception leaves an IN_PROGRESS artifact; summaries are generated only after full declared coverage passes. Preserve a failed artifact and choose a new output path for a new run. `smoke` uses one dataset, one seed, one candidate, six neural epochs and 30 tree rounds. It proves execution, not comparative performance. `closer` uses up to 4,000 public rows, 600 TabReD evaluation rows, 128 neural epochs and 800 tree rounds, keeping the same two candidates and three seeds. No `paper` preset is available.

The notebook provides the same visible driver after EXIT, behind `RUN_LARGER_LOCAL=False`. Its functions use the student's code, and its EXIT records source text for the model methods and all protocol functions, not just a disk hash. Each notebook-created smoke or larger-run JSON explicitly labels repository hashes as reference-only and adds actual kernel function/class-method sources, PRESETS and their digest. The default notebook performs one fresh five-arm smoke and audits the complete author result; it does not silently rerun the full larger suite.

For unattended CPU execution the shared operator dispatches lesson 60 to the corrected runner:

```bash
modal run --detach modal/foundation_repro.py --lesson 60 --preset closer
```

This wrapper has not been run for this audit. It uses the declared remote dependencies and CPU allocation; downloaded data and resulting environment may differ, so its hashes/versions must be checked. A repeated output path is rejected rather than silently resumed.

## Regenerate only this lesson

```bash
.venv/bin/python labs/_check_l060_v2.py
.venv/bin/python labs/_figures_l060_v2.py
.venv/bin/python labs/_build_l060.py
.venv/bin/python labs/_execute_l060.py
```

The figure operator reads the committed corrected evidence; it does not train. The scoped builder creates the student notebook, ignored local solution, prepared student HTML, lesson and reference from the canonical manuscript and depth source. `--keep-notebooks` rebuilds prose/preview without clearing already executed solution outputs when notebook source is unchanged. Parent-owned generic routes delegate to this builder, so future regeneration preserves the repair.

## Move toward original paper results

**TabArena v1:** follow the authors' [benchmarking examples](https://github.com/autogluon/tabarena/tree/main/examples/benchmarking) and the frozen paper version's §2/C/D contract. Fix the release and wrappers before running: current repository/main has evolved beyond v0.1. The paper used OpenML suite 457, 51 curated tasks, repeated outer evaluation and eight-fold inner ensembles with the declared 201 candidate configurations and post-hoc ensemble variants. Use the original metric and preprocessing for each method. The four-method frozen-result reanalysis in Lesson 056 is a useful audit reference; it is not training reproduction. This L060 operator does not implement that full training protocol.

**TabReD v4:** use the authors' [pinned paper operator](https://github.com/yandex-research/tabred/tree/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/paper) and exact experiment configs. Reproduce Figure 2 on all eight tasks, three split pairs, 15 initializations, the specified model roster and task-specific metrics. Restore the original full preprocessing and dataset artifacts, not just larger caps of the current release. Lesson 055 independently audits released author reports and documents release/target mismatches. Our five methods are not that figure's four-method study.

**TabM/RealMLP:** a correct architecture does not replace their complete tuning/evaluation recipes. TabM's original benchmark uses k=32, quantile-based numeric preprocessing, nonlinear embeddings in dagger variants, typically independent member batches, larger search/stopping budgets and a different rank convention. RealMLP's full TD variant includes components omitted from TD-S; its main best-epoch metric, 256-epoch schedule and meta-train/meta-test benchmark differ. Use the pinned official sources in `_sources_l060_v2.json` to align one declared target at a time. Never compare the current small score against a paper table with a numeric tolerance and call it MATCH.

## Delivery and assessment

Local teacher execution, behavioral checks, browser rendering and copied Pages are separate evidence. Live Colab and paid/cloud execution were NOT_RUN/NOT_CHECKED by the author. Parent records publication verification after review. Learner mastery remains unassessed until a completed EXIT and explanation are reviewed.
