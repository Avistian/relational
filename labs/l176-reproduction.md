# L176 Nested-Support ICL Evaluation

Approved2026-10-02. Two separate tracks: complete independent replay of the selected published experiment from L169, and300fresh nested-support evaluations. Whole-paper reproduction and fresh pretraining are NOT_RUN.

## Frozen protocol

RDB-PFN arXiv2603.03805v5 AppendixA.3/Tables6–10. Code a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL checkpoint eaf789a9b25ee8486d6f48997ba076f850bbc30b. Every source/data/checkpoint hash is preserved in evidence/l176/input-manifest.json, sources/l176/source-ledger.json, and inherited L166/L168/L169 ledgers. SHA256 authenticates cached checkpoint bytes before worker use. Checkpoints are downloaded on an explicit fresh rerun, not bundled in the default notebook.

Full rel-f1/driver-dnf702queries and rel-trial/study-outcome825queries. Three fixed arms:RDBPFN,RDBPFN_single,TabICLv1.1; contexts64/128/256/512/1024; seeds0–9. All300cells freshly evaluated. NumPy1.26.4 default_rng seeded by the unsigned big-endian first4SHA256bytes of the original task/seed string. One1024draw without replacement per task/seed; ordered prefixes yield smaller contexts. All models share exact support identities at each size; all test rows retained. Published L169 supports were independently drawn per size. This intentional sampling change makes L176 a course intervention, not new paper-table reproduction. No k=0 result.

Original preprocessing and prediction operations remain byte-identical to L169. Source imputation uses support medians; FeatureEncoder normalizes using support mean/variance. Query target tokens use support-label mean. These intermediates change with k, so the experiment measures the whole inference pipeline response, not an isolated label-count causal effect. Both RDB-PFN arms use6blocks,width96,4heads,FF192; TabICLv1.1 has32estimators. No gradient updates or test-based model/context/checkpoint selection. Report AUROC, per-seed paired gains and sampleSD over10support draws on each fixed task, not confidence over databases.

Runtime: Python3.11,torch2.5.1+CUDA12.4,numpy1.26.4,pandas2.2.3,scikit-learn1.6.1,pydantic1.10.26,pyyaml6.0.2,tabicl0.1.3. L4,2physical CPU cores,16GiB. Full query matrices are scored with chunk_size2000, larger than either task's test population.

## Information and historical boundaries

All11411F1/11994trial candidate keys are unique, disjoint from test keys, and drawn from train only. Every smaller support is an ordered prefix. Both classes occur at every approved context. Conservative inherited outcome-window checks use60days for released F1 and365days for trial and precede the earliest test cutoff; they do not redefine the RT30day contract in L175. Released complemented label orientation is preserved. Original keyed raw-label/selected-MAX-timestamp evidence is authenticated through the complete inherited replay.

Full DFS regeneration NOT_RUN; historical feature arrival/identity, independent checkpoint lineage and exact target-schema exclusion NOT_ESTABLISHED. No extension of selected MAX-column audits into an all-feature availability proof. L175's RT-v1 temporal failure remains unchanged. This separate RDB-PFN result does not repair it.

## Reproduce the full saved-evidence audit

From the repository root:

```bash
.venv/bin/python labs/_budget_l176.py .venv/bin/python labs/_check_l176.py
.venv/bin/python labs/_budget_l176.py .venv/bin/python labs/_verify_l176.py
.venv/bin/python labs/_figures_l176.py
.venv/bin/python labs/_build_l176.py
.venv/bin/python labs/_budget_l176.py .venv/bin/python labs/_execute_l176.py
.venv/bin/python labs/_delivery_l176.py
```

Default portable notebook: all300published-protocol saved runs plus300fresh-author nested runs,458100probabilities across two fixed test populations. Notebook execution is saved-evidence replay, not fresh inference. Original model, source preprocessing, support sampler, scoring/paired aggregation, complete runner and fetcher are visible inline. Three learner TODOs directly control the audits. Author output does not establish learner mastery: PENDING_WRITTEN_DEFENSE.

Original L169 exact TabICL repeatability FAIL is retained. Shared1024supports permit a separate cross-track consistency check; any measured floating-point difference remains reported without selecting a preferred score. All40RDB-PFN shared-support reruns were exact;9/20TabICL reruns were exact. The remaining11differ, with maximum probability difference0.000797540and maximum absolute AUROC difference0.000096861. See evidence/l176/shared-1024-comparison.json. Smaller contexts are different experiments and have no paper-tolerance verdict.

## Explicit fresh inference

Install the pinned GPU environment above. Fetcher authenticates weights and the prepared nested task arrays. Use a new output directory; existing per-run outputs cannot be overwritten.

```bash
.venv/bin/python labs/_fetch_l176.py --out /tmp/l176-input
python labs/_run_l176.py --input /tmp/l176-input \
  --source labs/sources/l166/upstream --out /path/to/new-l176-attempt --phase all
```

The manual runner executes all300evaluations but has no provider billing guard. The author managed lane is modal/l176_repro.py: pilot600s, remaining7200s, nonblocking spawn, no automatic retries. Its immutable attempt names and reservations cannot be reused. Future paid work requires a separately scoped ledger/attempt identity; do not delete old receipts to force a rerun. Preflight, source hashes, pilot forecast and aggregate reservation are checked before submission.

## Cost, attempts and validation

USD10aggregate cap,USD8planned stop,USD2protected margin;USD3overhead reserve included within those limits. Current Modal L4+2CPU+16GiB rateUSD.00028372/sec. Six-run pilot27.568s; slowest1024run8.0205s.294remaining runs at that worst case with3xmargin plus pilot duration project7101.666s, within7200s. Pilot630billable-reserved seconds and main7230include lifecycle allowance. Total conservative reservationUSD5.2300392. All retries, preparation and validation count. Invoice NOT_ITEMIZED; worker-body estimates exclude unitemized build/startup/storage.

Two detached CLI final-log waits timed out after successful nonblocking submission. Both remote function calls completed; no inference retry or partial replacement. Original logs retained. TabICL/scikit-learn emitted preprocessing overflow warnings; all output probabilities must pass finiteness and metric checks. Expected initial missing-implementation failure is retained in the local ledger. One figure-rendering connection-style error was fixed during visual preparation; it did not execute model inference. Numerical local cap3600aggregate seconds includes checks and notebook attempts.

Complete author result: 300fresh evaluations/229050predictions; inherited replay300evaluations/229050predictions. Independent pairwise, rank and sklearn AUROCs agree exactly in the saved fresh results. F1 RDBPFN means for64/128/256/512/1024: .706566/.722002/.725577/.719240/.718759. Trial: .542557/.567657/.574554/.598839/.615947. No test-size selection. Total worker-body491.471seconds,estimatedUSD.139440excluding unitemized costs; conservativeUSD5.2300392reservation retained.

Final measured results: evidence/l176/report.json and report.md. Independent checks: _verify_l176_results.json. Cost: evidence/l176/cost.json; provider state: evidence/l176/cloud-apps.json. Notebook/browser/clean-index publication checks are separate receipts. LiveColab/deployment NOT_CHECKED; no push or deployment requested.
