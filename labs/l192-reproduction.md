# L192 — RDBLearn study-outcome selected-task reproduction

**Selected reproduction: INCOMPLETE_SOURCE_PREPROCESSING_GATE.** Complete original preprocessing diagnostic and full task-key/schedule audit executed. All27 planned validation candidates and3selected tests remain NOT_RUN. New cloud/API spendUSD0. Target0.7167AUROC is cited, not matched. No repaired pipeline, pretraining or whole-paper reproduction is claimed.

## Approved experiment

RDBLearn toolkit paper arXiv2602.18495v1 Table1, rel-trial/study-outcome; Table4 reports depth4+TabPFNv2. Full validation search depths2/3/4 × TabPFNv2/TabPFNv2.5/LimiX-16M. Fit limit10000; uniform release downsampling when needed; full target history precedes downsampling in the release default. Three declared seeds0/1/2 are an explicit repeatability extension: historical seeds/aggregation not specified. Select per seed by full validation AUROC with listed-order ties; score only selected winners on complete test. Report each score, mean, sampleSD and absolute difference to0.7167. Descriptive tolerance0.02 is not historical identity or a significance test. No metric was computed because admission failed.

Source: RDBLearn v0.1.2 peeled commit `b5b03ebf8091547285a6e06cba53d2d1a40cb171`; FastDFS0.2.1 wheel SHA256 `e651a3b5db092a11deab0c9f01fe5797425caacb5e312833106e508994b2b0e3`. Later paper-config documentation is separately pinned at `3c5c30f52c45a85ad8605fde57ec495a4c165487`. Complete original estimator, preprocessor and FastDFS source are archived. See [source inventory](sources/l192/source-ledger.json), [protocol](evidence/l192/protocol.json), [candidate schedule](evidence/l192/candidate-schedule.json).

## Data and source diagnostics

Official task archive SHA256 `20eb922c1a8f894563f4b4c900c912e396688d2bd71eeb6b13f19429aa74a649`, matching released registry; each extracted parquet member authenticated against the ZIP.13779complete keys:11994train,960validation,825test; positives7647/561/483. Independent parquet-to-packet equality passes for every key and label. No raw database label reconstruction inL192. Full database archive and features were not materialized after the source stop. Historical paper-data identity and raw-field arrival legality remain NOT_ESTABLISHED.

For every distinct task cutoff, the conservative `available_at = historical_cutoff +365days` policy finds zero earlier training rows with unfinished windows. This is a complete schedule check, not proof of full feature legality. Real clinical label publication may occur on a different clock. The source’s target-history timestamp concern was investigated and is not asserted as a demonstrated failure on this task.

Original full TabularPreprocessor (real AutoGluon1.5.0) executes a synthetic known-category consistency intervention. Fit12rows containing b/c/d plus a numeric control; transform b/c/d, then unseen a, then b/c/d again. Codes shift0/1/2→1/2/3, numeric control unchanged. A fresh b-only query gives0; adding unseen a to the same query batch yields b=1. Independent unmodified SafeLabelEncoderTransformer execution confirms the shift on100generated vocabularies. This is not a task-frequency estimate, backend forward pass or AUROC effect.

## Environment and failed attempts

Initial source imports failed due to missing FastDFS and loguru in an inherited environment. An exploratory inherited stack also exceeded AutoGluon’s supported NumPy/scikit-learn constraints. Its diagnostic is retained as initial evidence, then superseded by a clean isolated environment: NumPy2.3.5, pandas2.3.3, scikit-learn1.7.2, AutoGluon features/common1.5.0, pydantic2.11.7, FastDFS0.2.1, featuretools1.31.0. The clean setup initially lacked pkg_resources; setuptools80.9.0 resolves that import. `pip check` passes;51loaded source files match the frozen release byte-for-byte; a complete fresh original preprocessor run confirms the same counterexample. The CPU diagnostic environment is not the historical GPU model environment. Every failed attempt remains in the local budget and failure receipts.

## Tested commands from repository root

Authenticate the bundled source and task bytes:

```bash
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_prepare_l192.py
```

Set up the isolated source-preprocessing diagnostic and execute it (no model checkpoints):

```bash
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_setup_l192.py
.venv/bin/python labs/_budget_l192.py /tmp/l192-repro-env/bin/python labs/_source_check_l192.py
.venv/bin/python labs/_budget_l192.py /tmp/l192-repro-env/bin/python labs/_preflight_l192.py
```

Preflight writes timing/environment receipts. If deliberately refreshing those inputs, reseal them, then audit:

```bash
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_seal_l192.py
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_audit_l192.py
.venv/bin/python labs/_budget_l192.py /tmp/l192-repro-env/bin/python labs/_verify_l192.py
```

Rebuild and execute the portable solution:

```bash
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_figures_l192.py
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_build_l192.py
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_execute_l192.py
.venv/bin/python labs/_budget_l192.py .venv/bin/python labs/_delivery_l192.py
```

The notebook embeds the complete query packet and original source. Its default run executes the original categorical primitive and independent packet audit; it does not rerun the entire AutoGluon pipeline. Full original estimator/preprocessor source is displayed in coherent appendix sections. Backend neural architectures and pretraining are not reimplemented in this setup lesson. The full model launcher is not claimed runnable or admitted after the failed source gate; upstream examples are archived for continuation.

## Budget, continuation and claims

USD10aggregate cap; planned stopUSD8/reserveUSD2 includes preparation, runs, retries and validation. No admitted model pilot means no complete-search forecast was established. Zero paid dispatches. Local safety cap3600seconds records failed setup, diagnostics, build, notebook and delivery attempts; five additional seconds conservatively account for initial dependency probes. No silent replacement of backend, support size or seeds.

Continuation requires source clarification or an explicitly declared repair, backend/checkpoint hashes, a compatible full inference environment, all-feature temporal audits and a measured aggregate forecast. Any repaired-pipeline run is a new declared comparison. A synthetic implementation failure cannot establish that the paper’s reported number is wrong. Full task-set reproduction remains futureL193work; historical identity NOT_ESTABLISHED; liveColab/deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. No push/deployment is part of this authoring task.
