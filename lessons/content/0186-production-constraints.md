<p class="mission-tag">From benchmark evidence to a serving contract</p>

A relational model can score well offline and still return an answer too late—or return a quick answer based on old related rows. Your tangible win today: **write a contract that checks response time and relational information age separately**.

**Route:** 15-minute core reading, then a 35–45-minute lab. You need query identity and temporal cutoffs from [Lesson 175](0175-zero-shot-evaluation.html), context and batch evaluation from [Lesson 176](0176-few-shot-icl-evaluation.html), and nested timers from [Lesson 177](0177-compute-budget-realism.html). Each is recapped below. [Lesson 185](0185-causal-relational-data.html) asks whether a prediction justifies an action; this lesson asks whether the required information can arrive in time. You do not need that lesson’s simulator to follow this one. No previous practical exit is marked complete here.

[[STATUS]]

## 1 · The missing contract

In an offline benchmark, features and predictions can already be on disk. At serving time, a customer request must find the customer’s orders, connect payments to those orders, form the model input, and return an answer. The slowest or least current dependency can determine whether that answer is usable.

Huyen distinguishes precomputed predictions, online prediction with historical features, and prediction using fresh features. These offer different ways to move work off the request path. Our lesson makes those choices executable with a deliberately small relational simulation. [Primary reading: Huyen, stages 1–3](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html).

[[FIG:serving]]

This is a **serving architecture**, not a new model architecture. Twenty customer IDs join two upstream snapshot streams, orders and payments. We model when their state is available, not their business values or prediction accuracy. Production systems need additional measurements for network hops, real joins, model execution, shared refresh resources, failures and capacity.

## 2 · Follow four clocks

**Event time** says when a source snapshot describes the world. **Arrival time** says when our service can see it. **Materialization time** says when a derived feature or precomputed prediction becomes ready. **Response time** says when the caller receives the answer. Knowing an event’s timestamp does not prove we had received it then.

At service start, the request can read only dependency snapshots that have arrived. For cached policies, it sees only a refresh that has completed. Within each source, a late older snapshot cannot replace a newer one. For this fixture, each source update is a state snapshot: accepting the latest observed snapshot does **not** prove that every real transaction arrived.

We define:

- **Request latency** = response time − request arrival.
- **Observed source age** = response time − the oldest event time among required dependency snapshots.
- **Materialization age** = response time − materialization completion.

These definitions are course policy, with units of milliseconds. The source-age limit is conservative for this fixture; it is not proof of feature correctness. A missing dependency produces unknown source age and fails the freshness contract.

[[FIG:clocks]]

[[PREDICT]]

A fast cache can contain stale data. Conversely, a request can wait a long time in a queue, then read a recent snapshot at service start: its features are fresh at response but its answer is late. The chosen read point matters; a product requiring a snapshot as of **request arrival** needs a different information contract.

**Compute two responses, one clock at a time.** These hypothetical timestamps use the same millisecond units and limits as the experiment. A cached request arrives at 5,000 and responds at 5,001; its materialization completed at 4,990, but its orders/payments event times are 4,900 and 2,000. A queued request arrives at 4,800, starts at 5,000 and responds at 5,015; it materializes at service start from events at 4,990 and 4,980.

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:3px"><thead><tr><th>Response</th><th>Latency</th><th>Source age</th><th>Material age</th></tr></thead><tbody><tr><td>Cached</td><td>1 ms</td><td>3,001 ms</td><td>11 ms</td></tr><tr><td>Queued</td><td>215 ms</td><td>35 ms</td><td>15 ms</td></tr></tbody></table>

The cached response meets the 100 ms deadline but breaches the 2,000 ms freshness limit. Its recent materialization only repackaged old payment information. The queued response meets freshness but misses the deadline: `200 ms wait + 15 ms service = 215 ms`.

**Transfer the trace.** Remove the queued response's payment snapshot. Source age becomes unknown, even though its order snapshot is recent; averaging known timestamps would silently waive a required dependency. Decide which fallback your contract requires in this case.

## 3 · Freeze the entire experiment

The executable target is **L186 Relational Serving Contract**, a course experiment. Huyen’s reading supplies design concepts, not numerical results to replicate. We run every cell in the frozen grid: **3 policies × 3 request rates × 3 arrival conditions × 3 seeds = 81 scenarios**, each with **10,000 requests**.

| Policy | Foreground service | Refresh | What waits on the caller’s path? |
|---|---:|---:|---|
| Precomputed prediction | 1 ms | Every 5,000 ms | Prediction lookup |
| Cached features | 5 ms | Every 1,000 ms | Feature lookup and nominal scoring |
| Request-time features | 15 ms | At service start | Nominal feature assembly and scoring |

All durations are **hypothetical constants**. Background refresh takes 20 ms on independent, unconstrained resources. No trained model runs. A single first-in, first-out (**FIFO**) foreground worker serves requests in arrival order. It serves every request to completion; there are no silent drops, retries, cancellation or autoscaling. Nominal rates are 10, 50 and 100 requests/s, with seeded, rounded exponential gaps. All three policies get exactly the same trace within each condition and seed. Bootstrap snapshots remain in the reported metrics; no warm-up responses are discarded.

Orders and payments publish state snapshots every 500 ms. Normal arrivals lag by 0–100 ms. In the middle third of the request horizon, payments either add 2,000 ms of delay or are held until the interruption ends. Orders continue. The simulation generates enough future source updates to drain the foreground queue. It cannot access those updates before their arrival.

The fixed limits are **100 ms request latency** and **2,000 ms source age**; equality passes. p99 uses the nearest-rank definition: sort 10,000 latencies and take the 9,900th. This is a finite-trace statistic, not an estimate with a calibrated production confidence bound. The three seeds expose trace variation; they do not cover hardware uncertainty.

[[FIG:tradeoffs]]

[[RESULTS]]

**Utilization** compares arriving work with available worker time. At 100 requests/s, each request needs 0.015 seconds, so a 15 ms worker receives `100 × 0.015 = 1.5` worker-seconds of work per second: work arrives faster than that one worker can finish it. The observed finite-trace queue grows. Freshly reading dependencies cannot recover the already spent waiting time. Independent background refresh capacity favors cached policies in this experiment; measuring that cost is required before making a deployment decision.

## 4 · Change an assumption, predict, then inspect

Before selecting a new cell, predict whether the change affects queueing, source age, or both. Reset returns to cached features, 50 requests/s, normal arrivals, seed 0. The explorer displays complete saved cells; it does not interpolate a new result.

[[EXPLORER]]

**Try:** hold rate and seed fixed, switch from normal to interrupted payments. Then switch policy. Can any policy make an unavailable payment snapshot arrive sooner? Why can the request-time policy have a lower stale-response fraction at high load? Check that it drains after the incident ends; this aggregate mixes different response times and is not evidence of better incident handling.

## 5 · Monitor what you can actually observe

Monitor latency, traffic, errors and saturation to understand the service. Here, deadline misses and queue wait expose operational trouble. Real error rates and resource saturation need instrumentation absent from this simulation. [Google SRE, monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/).

Our freshness monitor examines the last 100 completed responses. It triggers only when **more than ten** are stale, after a complete window exists. This is a response-count window: at lower traffic, accumulating 100 observations takes longer. It is not a fixed-duration detector.

Report whether an alert was already active at the injected onset. “First alert response after onset” can be almost immediate for a policy that was stale all along. That is not new fault detection. The report also records new false-to-true episodes. Recovery bursts, sparse observations and this fixed threshold limit what either time means.

Quality needs a separate clock. In the simulation, labels become available 30,500 ms after each request’s arrival. We count their availability at the final response but compute no accuracy: no scores or labels were generated. This artificial delay illustrates the contract; it is not an F1 target window. Production scoring needs mature labels joined to the exact original request and model version. A latency or freshness alert alone does not establish model degradation. [Huyen, natural labels and monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html).

## 6 · The real batch evidence answers a narrower question

We authenticate all **300 original L176 run receipts** and check the full two-task, three-model, five-context, ten-seed grid. Their batches contain 702 F1 or 825 trial queries. We compare individual JSON records with the phase receipts and reconstruct their timers independently using SQL.

[[RECEIPTS]]

Dividing a batch duration by its query count can describe amortized batch throughput cost. It cannot recover queueing, network or individual request latency. There are no per-request samples here, so request p99 remains **NOT_MEASURED**. This lane reads timing metadata, not checkpoint weights or saved prediction values. Hashes establish unchanged bytes, not the physical accuracy of the original clocks.

## 7 · Lab: implement three contracts that control the result

Open the [student notebook](../labs/0186-production-constraints.ipynb), [rendered executed solution](../labs/html/0186-production-constraints.html), or [solution notebook](../labs/solutions/0186-production-constraints.ipynb). The portable notebook embeds the receipt packet and visible simulator. It runs offline using the Python standard library, with no cloud call or model download.

1. **FIFO schedule:** include time spent waiting behind earlier requests. Return every start and finish time.
2. **Dependency source age:** use the oldest required snapshot; keep missing information unknown.
3. **Freshness alert:** require the full trailing window and the strict threshold.

All three functions are called by the full 81-scenario run. An independent author check reconstructs every response using indexed prefix maxima and a different queue recurrence. Passing notebook checks is preparation for the written defense, not proof of learner mastery.

[[CODE]]

**CHECK:** serve arrivals `[0, 2, 30]` with 10 ms service. Starts are `[0, 10, 30]`, finishes `[10, 20, 40]`, latencies `[10, 18, 10]`. Then change one dependency event from 90 to 60 at response time 100: the source age must increase from 20 to 40 if the other dependency is 80.

## 8 · Exit: write the fallback, not just the threshold

Write a six-sentence serving contract: required relational keys, snapshot/read point, response deadline, freshness limit, fallback action, and monitoring/label policy. Choose whether stale or missing dependencies cause abstention, a baseline response, or an explicitly marked stale answer. The simulation reports breaches; it does not execute or prove the quality of your proposed fallback.

Defend one choice under interrupted payments and another under overloaded request-time service. State the real measurements still needed before deployment. Tomorrow, recall the four clocks without looking. In a week, explain why batch throughput cannot establish request p99.

[[TEACHBACK]]

**Ask the teacher** about any unclear trace, metric or assumption. Bring your written contract for feedback.

[Quick reference](../reference/production-constraints.html) · [Frozen protocol and commands](../labs/l186-reproduction.md) · [Complete report](../labs/evidence/l186/report.json) · [Independent verification](../labs/_verify_l186_results.json) · [Source ledger](../labs/sources/l186/source-ledger.json) · [Curriculum and the next privacy lesson](../reference/curriculum.html)

**Next: [Lesson 187](0187-ethics-privacy-reg.html).** A response can meet its deadline and freshness limit while exposing a person’s information. Extend the serving contract with a protected unit and a precise release rule.
