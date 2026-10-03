# L186 Relational Serving Contract

Approved 2026-10-02. Full selected course simulation: 81 scenarios, 10,000 requests each; separate replay of 300 original L176 batch receipts. Huyen's readings are conceptual sources, not a published numerical target. Whole-paper numerical reproduction, fresh model inference and live production performance are NOT_RUN.

## Protocol

Frozen config: `evidence/l186/config.json`. Complete grid: precomputed/cached/request-time × nominal rates10/50/100 per second × normal/delayed/interrupted payments ×seeds0/1/2. Paired trace per rate/condition/seed; every request drains. Positive integer-ms exponential gaps, 20 customers, two upstream snapshot dependencies, 500ms source period with fixed offsets. Random arrivals and delays use Python Random; runtime version pinned in sources/l186/source-ledger.json, realized input and response trace SHA256 in each cell. Cross-runtime trace identity must be checked, not assumed.

Service times1/5/15ms; precomputed/cache refresh5000/1000ms and refresh work20ms; one foreground FIFO worker, independent unconstrained refresh resources. None of these durations is measured model latency. Bootstrap snapshots event-5000/arrival-4900, cached bootstrap ready-4880. Sources are state snapshots, not individual transactions; max observed source timestamp does not prove complete delivery. Older out-of-order snapshots cannot roll state backward. Read at service start or latest completed refresh, so request-time inputs may include events later than request arrival. Source generation continues200000ms beyond final arrival, sufficient for the frozen grid's drain.

Normal arrival delay0–100ms. During the middle third of the request-arrival horizon, payment snapshots add2000ms delay or are held until the interval ends. Orders remain normal. Deadline100ms, source-age2000ms, equality passes. Source age=response−minimum dependency event; material age=response−materialization. Unknown source age fails freshness. Request latency includes queue wait. Nearest-rank p50/p95/p99, no percentile interpolation. A full trailing100-response window triggers only above10% stale; report already-active alerts separately from new episodes. All three seeds reported; no test/seed/policy selection. Request-based label readiness at arrival+30500ms is hypothetical, only availability coverage is computed; performance NOT_SCORED.

Replayed batch evidence: original L176 2tasks×3arms×5contexts×10seeds,229050 batch prediction rows. All per-run JSON compared with phase receipts and inherited seals; no probabilities rescored. Distinct load events keyed by phase/task/model. Worker bodies enclose evaluation/loading; residual is not startup. Batch average time cannot reconstruct request p99.

## Exact commands

From repository root, Python environment needs nbformat, nbclient, nbconvert, matplotlib and playwright for author delivery. Numerical experiment and portable notebook use only standard Python.

```bash
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_prepare_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_check_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_run_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_verify_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_figures_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_build_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_execute_l186.py
.venv/bin/python labs/_budget_l186.py .venv/bin/python labs/_delivery_l186.py
```

The default notebook embeds the original receipt/source packet and runs all81simulations plus an independent810000-response reconstruction. Download student or executed solution; no GPU, package install or network. Three TODOs govern actual full-run behavior. The student notebook intentionally stops until they are implemented. The executed solution does not complete your written defense.

## Budget and evidence boundaries

USD0newcloud/API;1800aggregate local numerical seconds including setup, failures, repeats, audits, figures and portable execution. `local-budget.json` retains failures and measures all guarded commands; stop rather than shrink the experiment. Hashes authenticate artifacts, not historical clock truth. Hypothetical timings, uncosted refresh contention, simplified source snapshots, no model scores, no cancellation, network, autoscaling or actual fallback remain explicit limitations. The 81-cell simulator is not a load test of an actual server. LiveColab/deployment NOT_CHECKED. Historical arrival identity NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE. No paid runs or publishing requested.

Author checks: independent availability/queue/monitor reconstruction of every response, direct metric checks, SQL batch aggregation, random FIFO cases, rejected wrong learner functions and corrupt receipts, empty-directory solution execution, desktop/mobile/keyboard/reset/print/no-JS, exact source parity and deterministic builder. Delivery receipts report actual checks. Git-index publication is a separate copied-site check, not deployment.
