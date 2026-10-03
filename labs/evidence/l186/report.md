# L186 Relational Serving Contract

**Executed:** complete **81-scenario course simulation / 810,000 responses**, plus **300 original batch-receipt replays**. All service times are hypothetical. Live production performance, fresh model inference and paper-result reproduction are `NOT_RUN`; request latency from original batch receipts is `NOT_MEASURED`. Learner status: `PENDING_WRITTEN_DEFENSE`.

| Policy · normal arrivals · seed 0 | Requests/s | p99 latency | Deadline misses | Stale responses |
|---|---:|---:|---:|---:|
| precomputed | 10 | 1 ms | 0.00% | 70.05% |
| cached | 10 | 9 ms | 0.00% | 0.08% |
| request | 10 | 31 ms | 0.00% | 0.00% |
| precomputed | 50 | 1 ms | 0.00% | 70.20% |
| cached | 50 | 13 ms | 0.00% | 0.43% |
| request | 50 | 127 ms | 2.87% | 0.04% |
| precomputed | 100 | 1 ms | 0.00% | 70.47% |
| cached | 100 | 21 ms | 0.00% | 1.00% |
| request | 100 | 47,983 ms | 99.71% | 0.06% |

This worked table shows seed 0; the figure includes all three seeds and the full report retains all 81 cells. No best seed or policy was selected.

**474.434 s** of inner evaluation time + **0.939 s** from 12 distinct loading events sit inside **491.471 s** of worker time. The difference is other worker work; it is not measured cloud startup. Do not add inner timers to the outer total.
