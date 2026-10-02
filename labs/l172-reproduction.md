# L172 Full F1 Schema-Tokenization Audit

Approved 2026-10-02. Complete course pipeline execution, not a published model-performance reproduction.

## Frozen protocol

Input: complete L171 RelBench 1.1.0 F1 archive, all nine Parquet members, pinned F1 schema source and registry hashes. `evidence/l172/input-manifest.json` authenticates exact bytes. `schema.json` assigns all 67 columns a semantic kind, role, storage dtype and named descriptor. All 97,606 rows / 866,746 cells are transformed; no subsampling. Schema policies are course declarations, not inferred source semantics.

Fit admission: date<2005-01-01 for time-bearing tables; zero admitted rows for the three untimed tables. Fit numerical population mean/SD and sorted categorical vocabulary only on admitted nonnull rows. Empty numerical fit defaults 0/1; constant scale 1. Empty categories remain empty. Exact keys and Unicode text remain strings. Timestamp is UTC epoch days. Masked values are erased before encoding; MASKED overrides MISSING; observed out-of-vocabulary categories are UNKNOWN. States disambiguate neutral payloads. All-table column reorder, every-cell masking and all-number/category heldout interventions are complete.

Independent verification reconstructs each original cell via scalar arithmetic, compares numbers with rtol/atol 1e-12, checks every nonnull FK against target IDs, regenerates the full report, rejects three wrong learner implementations and three corrupted input manifests. Output table hashes capture exact deterministic encoding in the recorded runtime. They are not a cross-version floating-point identity promise.

## Reproduce from repository root

```bash
.venv/bin/python labs/_budget_l172.py .venv/bin/python labs/_prepare_l172.py
.venv/bin/python labs/_budget_l172.py .venv/bin/python labs/_check_l172.py
.venv/bin/python labs/_budget_l172.py .venv/bin/python labs/_audit_l172.py
.venv/bin/python labs/_budget_l172.py .venv/bin/python labs/_verify_l172.py
.venv/bin/python labs/_figures_l172.py
.venv/bin/python labs/_build_l172.py
.venv/bin/python labs/_budget_l172.py .venv/bin/python labs/_execute_l172.py
.venv/bin/python labs/_delivery_l172.py
.venv/bin/python labs/_checkout_l172.py
```

`_prepare` authenticates existing evidence and regenerates the frozen explicit policy; it does not download or upgrade data. Pin required data and sources from the repository. Student/solution notebooks embed their complete packet and visible functions, require pandas 3.0.3 / PyArrow 24.0.0 / NumPy, and need no network after dependencies. Runtime package versions are recorded by `_execute`. Prepared HTML hides only the large base64 data payload, not implementation.

Budget: USD 0 cloud/API, 1800 aggregate seconds numerical processing, tests and notebook execution including failures. `_budget_l172.py` records every attempt, reserves remaining time and terminates the process group at cutoff. An interrupted ledger fails closed pending reservation reconciliation. Never erase failed attempts to reclaim budget. Authoring, reading retrieval, figures, rendering and browser/Pages delivery are outside this numerical ceiling. No paid fallback or deployment.

## Results and evidence boundaries

See `evidence/l172/report.json`, `report.md`, `_verify_l172_results.json`, `_execution_l172_results.json`, `_delivery_l172_results.json` and `_checkout_l172_results.json`. Full selected tokenization COMPLETE; every row is transformed and every declared column handled. Historical availability remains NOT_ESTABLISHED, especially for 1,145 untimed rows. Fit admission is not a task split, time-travel-safe sampler or certified historical feature availability. Training-only statistics do not guarantee masked-target isolation for a future training objective: the downstream objective must specify its preprocessing and information contract.

RT v1 §3.1 is a primary comparison reading. The course does not implement its learned projections, frozen text embeddings, normalized timestamp convention, attention, losses or trainer. Category codes are local labels, not ordered quantities or transferable embeddings. Model performance, whole-paper reproduction and fresh pretraining NOT_RUN; live Colab/deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. Data rights remain subject to the L171 source ledger.
