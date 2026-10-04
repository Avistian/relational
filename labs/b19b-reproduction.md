# B19b reproduction contract — approved 2026-10-04

## Named target and frozen comparison

**B19b-TABPFN-TS-V4-FIGURE4.1**, arXiv2501.02945v4 (26 January2026). Complete97 GIFT-Eval tasks,13 actual plotted models, all 39 printed point estimates and original95% intervals. Figure roster (not just prose): TiRex,Toto1.0,Moirai2,TabPFN-TS,TimesFM2.0,Chronos-Bolt-Base,Sundial-Base,PatchTST,TFT,DeepAR,AutoArima,AutoTheta,SeasonalNaive. AutoETS is discussed but not plotted.

Primary source packet: `sources/b19b/manifest.json`. GIFT snapshot `2c13c6ff6fd7557c4353674c800d41995f280233`; wrapper snapshot `47dc9c46bb379f40b455d20b07aa1a891b21907e`. Snapshot selection is chronological (last commit by submission date), NOT proof of original-run identity. GIFT notebook additionally references wrapperv1.0.0 (`e8137ab8df46d82ab9a8a63ae1814cde6b50627f`); that source is archived separately.

Paper contract: missing context values removed, latest 4096 retained, running index+calendar+top5 automatic seasonalities, fixed TabPFN-v2 checkpoint linked from Appendix A.3, quantiles0.1,…,0.9. MASE uses the median and training seasonal-error denominator. WQL is twice pinball loss normalized by absolute held-out targets, averaged across quantiles. Benchmark aggregation within series/tasks is the original evaluator's policy; do not substitute the course's equal-origin aggregation.

## Executed saved-score lane

All13 model files from the SAME pinned GIFT snapshot,97 unique identical task keys each:1261 records. Recompute per-task average tied WQL ranks and geometric means of model/SeasonalNaive ratios, equal task weights. Preserve every task. Positive finite metrics required; missing/duplicate/nonfinite/zero records fail. The portable notebook embeds the entire compact score matrix; the independent verifier compares every compact row with original CSV bytes, then uses SciPy rankdata/gmean and a scalar diagnostic scorer.

Only2/39 printed fields agree to 3 decimals (SeasonalNaive's relative scores, both1 by construction). This is **COMPLETE_RELEASED_SCORE_RECONSTRUCTION**, while the named historical figure is **INCOMPLETE_SOURCE_PROTOCOL_GATE**. A valid calculation from these files is not figure parity. Original95% intervals remain **NOT_RECONSTRUCTED**: resampling unit, method, repetitions and seed are unestablished. Do not invent IID task bootstraps:97 tasks share datasets and series.

Source findings: wrapper submission dates to January2025, while GIFT's TabPFN scores changed May2025. The GIFT README documents a July2025 baseline correction. Pre-correction repository's naive file contains only4tasks. Later SeasonalNaive and naive files differ. Neither selecting a baseline because it matches nor relabeling naive as seasonal authenticates the paper. A bounded history search is not proof that no historical artifacts exist.

The v4 appendix spells the linked checkpoint `regression`; source uses `regressor` (v1 uses a short alias). The dated wrapper declares lower-bounded dependencies and its CLI hardcodes client mode, while the GIFT notebook selects local mode. Model family/name and snapshot dates do not identify actual inference bytes. No new calls to the API, GPU or model download were made.

The archived source uses Hann **multiplication** and right zero-padding; paper pseudocode describes convolution and symmetric padding. Calendar source preserves its period-minus-one convention. Our visible seasonal implementation mirrors the archived default code, verified on four signal cases, rather than silently repairing it. Constant-input extraction can report numerical residual peaks; source agreement is not scientific validation.

## Complete execution path and unrun work

`_paper_b19b.py --plan` writes the full55-series-configuration/97-task execution plan from the archived dataset roster (the exact counts are asserted by the operator). It does not call any model. `--execute` is an optional future operator for an explicitly prepared local checkout, with a fixed local checkpoint, its SHA256, and an explicit wall-time limit. It selects local mode instead of the source CLI's client default and records that deviation. The archived published source supplies the full model wrapper/evaluator, not a stub. Dependencies/checkpoint/data must already be installed and authenticated. It is a candidate-release fresh evaluation path, NOT an authenticated historical reproducer. Syntax and plan are checked; full backend execution is NOT_RUN/NOT_CHECKED. The notebook includes this path gated OFF.

Fresh97-task inference, raw prediction rescoring, retraining other baselines, pretraining and all other paper experiments: **NOT_RUN**. Do not infer their completion from saved scores, source parity, notebook execution or site checks.

## Course diagnostic, fixed before outcomes

B19b-FORECAST-AVAILABILITY:3 seeds(0,1,2),156 steps each; origins96,120,144; horizon12;27arm/origin/seed scores and324keyed predictions. Issue time origin−1; training prefix[0,origin). Target=20+3sin(2πt/12)+2promotion+4weather+noise; promotion=(t mod24<3) announced24steps ahead; weather~N(0,1) observed at event time; noise~N(0,.5). No validation-driven choice.

Arms: seasonal-naive with median-centered historical seasonal-difference quantiles; fixed ridge(1e−6) with intercept/sin/cos/promotion; same ridge with observed future weather (explicit illegal oracle). Residual quantiles use historical in-sample residuals and have no calibration guarantee. Scores average origins within seeds, then seeds; do not treat overlapping histories as independent datasets. All legal predictions unchanged by unavailable-weather intervention; oracle sensitivity reported. Legal regression does not beat seasonal naive on every seed. This is a controlled synthetic mechanism, not TabPFN inference or a real-world benchmark.

## Budget and evidence boundary

USD 0paid.3600aggregate local numerical seconds for preparation, failures/retries, experiment, notebooks and verification; ledger `evidence/b19b/local-budget.json`, subprocess group cutoff. Standing per-lessonUSD10 total budget is unchanged but unused. No silent downscale or paid dispatch. LearnerPENDING_WRITTEN_DEFENSE; liveColabNOT_CHECKED; deploymentNOT_REQUESTED.
