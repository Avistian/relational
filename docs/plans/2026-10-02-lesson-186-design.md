# Lesson 186 · Production constraints

Approved by the user in conversation, 2026-10-02. Named experiment: L186 Relational Serving Contract. This is a complete course simulation and separate saved batch-timing replay, not a published numerical benchmark or production deployment.

## Frozen experiment

81 cells = 3 policies × rates 10/50/100 requests per second × normal/delayed/interrupted upstream arrivals × seeds 0/1/2. Each cell serves all 10,000 requests and drains its single FIFO worker; no dropping requests to improve latency. Pair the same request/event trace across policies. Twenty customers each depend on orders and payments. Source snapshots occur every 500 ms with customer/table offsets; they describe an upstream state snapshot, not proof of a complete real event stream. Initial snapshots have event/arrival times -5000/-4900 ms. Random request interarrival gaps are positive integer milliseconds with mean 1000/rate, drawn with Python Random.expovariate and rounded, minimum 1; customer IDs sampled with the same seed. Record Python version and trace hashes.

Normal source arrival delay is uniform integer 0–100 ms. During the middle third of the request horizon, delayed payments add 2000 ms; interrupted payments are held until the incident ends. Orders continue normally. Out-of-order arrivals never roll state backward. Generate source snapshots through the last request arrival plus 200,000 ms to cover the complete drain. Future event/arrival information is inaccessible to a request snapshot.

Policies: precomputed prediction refreshed every 5000 ms, 1 ms foreground service; cached features every 1000 ms, 5 ms foreground service; request-time features read at worker start, 15 ms foreground service. Scheduled refresh reads at tick and becomes available 20 ms later; refresh resources are independent, unconstrained and uncosted. All service/refresh times are hypothetical constants; no fitted model scores or production speed claims. Feature/source age at response is response time minus the minimum observed event time of the two dependency snapshots. Materialization age is separate. Query reads at service start, never future completion time.

Frozen thresholds: deadline 100 ms (strict greater-than is a miss), source freshness 2000 ms (strict greater-than is stale). Rolling last 100 responses triggers a freshness alert if more than 10% are stale, with no alert before 100 observations. Report first alert, preincident alerts, first post-onset alert and delay relative to injected onset; an alert already active at onset is not new incident detection. Also report first false-to-true transition after onset and episodes. No empirical optimum selection. Labels become available 30,500 ms after each request arrival; compute availability coverage only, never invented accuracy/drift.

Metrics per cell: nearest-rank p50/p95/p99 request latency, deadline misses, source-age p95, stale-response fraction, materialization-age p95, queue maxima, freshness alert timings/episodes, mature-label count. Retain trace digests, a worked trace and per-cell summaries. Independently verify all cells using a separate event/refresh replay and queue arithmetic; reject broken learner implementations and corrupt receipts.

Separate receipt lane: authenticate and replay all 300 L176 run JSON records, original phase receipts and worker timer source. Verify full 2-task ×3-model ×5-context ×10-seed grid, deduplicate load events by phase/task/model, reconstruct nested worker totals. No request-level percentile from batch timers; no new model evaluation. Source snapshots: Huyen real-time ML/monitoring, Google SRE monitoring; URLs and SHA256 pinned locally.

## Delivery and limits

USD0 new cloud/API; 1800 aggregate local numerical seconds including setup, tests, failures, notebook execution and audits. Runtime guard stops descendants at cutoff. Missing evidence remains missing. Full course simulation can be COMPLETE while live production performance, fresh model inference and paper-result reproduction are NOT_RUN. Historical arrival evidence remains NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.

Create connected HTML lesson and reference, a reusable serving explorer, three portable figures, visible simulation/replay code, three live learner contracts and student/executed solution notebooks. Verify independent mathematics, complete grid, empty-directory execution, source parity, deterministic builds, desktop/mobile/keyboard/reset/print/no-JS, navigation and complete real Pages build from the Git index. Preserve concurrent L182 work and all earlier changes; no push or deployment.

The referenced writing-plans skill is unavailable in supplied locations. This approved document records the implementation plan: (1) freeze/test contracts, (2) implement/run simulation and receipt audit, (3) independently verify, (4) author/build complete teaching package, (5) execute and visually inspect, (6) stage intended deliverables and verify copied publication.
