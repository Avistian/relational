<p class="stream-label">One skill: make temporal queries faster without losing their meaning</p>

A node has interacted a million times. A prediction needs its history, but loading all of it for every query is expensive. **By the end, you will implement a bounded temporal sampler and defend a measured speed–accuracy trade-off.** This is directly useful when a relational model reads linked records as they existed at a particular time.

L107 compressed events into snapshots. Here we retain individual event times and reduce how much history a query reads. The key distinction is between **finding the same records faster** and **choosing different records**. Only the first should preserve predictions exactly.

## First retrieve, without notes

<div id="warmup"></div>

1. At time 8, can an event stamped exactly 8 enter a strict-past prediction?
2. If two systems score the same positives against different negatives, is their accuracy difference a clean model comparison?
3. What information does an hourly count snapshot lose?

<details><summary>Check after writing your answers</summary>

No: strict-past means event time < query time. Different negative candidates change the evaluation problem. A count snapshot loses within-window event order and exact times. Revisit [L104](0104-information-leakage-in-time.html), [L106](0106-temporal-link-prediction.html), and [L107](0107-snapshot-methods.html) if needed.

</details>

## 1 · Find the legal interval before sampling

> **In plain terms.** First decide what this query is allowed to know. Only then decide which allowed records to read.

A **temporal query** is a node together with a cutoff time, written `(node, cutoff)`. A **recency window** is a duration W that limits how old a record may be. Our legal interval is `[cutoff − W, cutoff)`: include the left endpoint and exclude the right. With no age limit, W is infinity. The cutoff remains strict.

**Worked example.** Node A has events at times `[1, 3, 3, 8, 11]`. At cutoff 8 with W = 5, the legal timestamps are `[3, 3]`. Both events at time 3 remain distinct records. The event at 8 is unavailable. Never resolve a timestamp tie by arbitrary file order unless your data supplies a genuine within-time ordering contract.

[[FIG:interval]]

A sorted array makes this interval cheap to locate. **Binary search** repeatedly halves a candidate range. `searchsorted(times, boundary, side='left')` returns the first position whose value is at least the boundary. Two left searches give the start and end of our half-open slice. In the example they return 1 and 3; `times[1:3]` contains two records.

[[CODE:window_bounds]]

**Predict, then change the controls:** move the cutoff from 8 to 9 with W fixed at 5. Does time 3 remain eligible? Does time 8 enter?

[[WIDGET]]

<details><summary>Worked answer, including a no-JavaScript fallback</summary>

The new interval is [4, 9), so both time-3 events expire and time 8 enters. With recent fanout 2, the result has one real event and one padding slot. At the original [3, 8), recent fanout 1 retains the later event ID among the two time-3 records. That deterministic tie rule selects records; it does not permit equal-cutoff events.

</details>

> **Scope check.** Wikipedia records event time, not ingestion time. We assume availability equals event time. Late-arriving database records need L104's additional availability check; an event-time index alone cannot certify that a record was known then. A recency window also does not imply a future prediction horizon.

## 2 · Bound the work, and state the sampling distribution

**Fanout k** is the maximum number of neighbor slots produced per query. Each sampled record carries three aligned values: neighboring node ID, event ID, and event timestamp. Edge features are retrieved by event ID later.

**Uniform with replacement** draws k independent positions from the legal records. If only two records exist and k = 4, the sample might be `[0, 1, 0, 1]`. Duplicates are intentional; k is not the number of distinct neighbors. We sort the selected records by time before returning them.

**Recent sampling** keeps the last min(k, count) eligible records. Missing slots are padded on the left. ID 0 denotes padding and has a separate mask; it is not an observed edge at time zero. The adapter preserves Wikipedia’s integer timestamps exactly in float32; larger absolute timestamps or fractional timestamps need a precision audit before reuse. A model's behavior on an entirely padded row still depends on its attention implementation. The preserved TGAT release uses a finite masking value, and an all-padding row does not automatically imply a zero embedding.

[[CODE:choose_positions]]

The readable reference calls these two student functions directly. The optimized implementation performs the same arithmetic over arrays. The lab requires sample-for-sample agreement, including duplicate choices, tie ordering and padding. A wrong TODO cannot pass simply because the optimized code is correct.

The [TGAT authors describe temporal neighborhood sampling and subgraph batching in §3.4](https://arxiv.org/html/2002.07962v1#S3.SS4). Their [released experiment uses uniform sampling and fanout 20](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/tree/9293d10d1943c4bd4a186337cf38ba98e4c8bb99). Our recent/window variants are explicit course interventions.

## 3 · Store once, search many queries together

A **compressed adjacency index** stores all neighbor records in flat arrays. An offsets array says where each node's row starts and stops. Within a row, records are sorted by `(timestamp, event ID)`. A row can be empty. This lab inserts each interaction in both endpoint rows, matching TGAT's undirected neighborhood view; the prediction question still has a source and destination.

For B queries, two batched binary searches maintain B lower/upper position pairs. At each step, all unfinished searches compare their own midpoint against their own cutoff. A query never searches another node's row. After finding eligible slices, we gather at most B × k records.

[[FIG:pipeline]]

**Shapes.** Nodes and cutoffs have shape `[B]`. Each returned array has shape `[B, k]`. The model gathers `[B, k, F]` edge and node features, where F is feature width. TGAT recursively encodes neighbors, combines those vectors with elapsed-time encodings and edge features, applies masked attention, and merges the result with the source representation. A shared pair-scoring network produces positive and negative probabilities. Training minimizes positive/negative binary cross-entropy; this lesson loads frozen trained weights and changes only their context.

**Work.** For a row of d events, binary search costs O(log d) comparisons: the number grows logarithmically with history length. Gathering costs O(k) per query. Sorting sampled timestamps adds O(k log k). The implementation does not scan the full history on every query. Construction sorts the original records once and stores O(E + N) values, where E counts interactions and N counts nodes. Two endpoint rows mean approximately 2E stored records.

**Memory boundary.** A finite window reduces eligible records but this offline implementation retains the complete index. The window therefore does **not** reduce its persistent memory. Online expiration could reclaim old records, but then later replay of earlier queries would require another store. Report index-array bytes separately from temporary arrays, process memory and GPU tensors.

## 4 · The second hop must inherit the edge time

> **In plain terms.** A neighbor seen yesterday must be represented using what was known yesterday, even when today's root query can see more.

**Worked example.** A query for A at 8 traverses an A–B edge at 3. When encoding B on that path, TGAT passes cutoff **3**, not 8. A B–C edge at 6 is therefore excluded. Reusing the root cutoff would change the path's temporal meaning.

[[FIG:recursion]]

The index is queried once per `(node, cutoff)` pair. Two visits to node B at different cutoffs may have different valid histories. A cache keyed only by node ID is unsafe. Even a `(node, cutoff)` cache needs layer, sampler policy, window, model state and stochastic-sampling semantics considered; reusing a random sample can change the estimator's dependence structure. This lesson does not introduce such a cache.

For neighbor expansion alone, L layers with constant fanout k can expose up to `1 + k + k² + … + kᴸ` tree slots per root before deduplication. With two layers, k = 20 gives 421; k = 5 gives 31. These are **tree slot bounds**, not unique nodes, exact FLOPs, or memory measurements. This TGAT implementation also recursively computes the source representation at each layer, and scores source, positive destination and negative destination. Its actual work exceeds the neighbor-tree sketch; use measured inference timing.

**Check:** two queries request B at times 3 and 8. Can they share the same latest embedding solely because their node IDs match? Explain a counterexample before opening the answer.

<details><summary>Check the cache counterexample</summary>

If B has an edge at 6, the time-8 query may use it and the time-3 query may not. Reusing the later representation leaks information into the earlier one. A node ID identifies an entity, not its information state.

</details>

## 5 · Define the experiment before inspecting accuracy

[[PREDICT]]

**Optimization experiment.** Run the scalar and batched samplers over all 157,474 Wikipedia source-node queries. Fix queries, fanout 20, unlimited history, uniform variates and output order. Check every sampled node, event ID and timestamp. Warm both implementations, alternate their order, and measure three repetitions. Timings exclude random-number generation, index construction and neural inference; construction is reported separately.

**Full checkpoint evaluation.** Load all ten completed L103 TGAT checkpoints. First replay the released Wikipedia all-event and new-node evaluations, preserving the release's boundary omission, negative sampler, random state and batch aggregation. Check every probability against the archived run. Then evaluate four strict-past interventions on the same question IDs, negative candidates and batches:

| Condition | Neighbor slots | Eligible history | Selection |
|---|---:|---|---|
| uniform20 | 20 | All strictly earlier records | Uniform with replacement |
| uniform5 | 5 | All strictly earlier records | Uniform with replacement |
| recent20 | 20 | All strictly earlier records | Most recent records |
| day20 | 20 | Previous 86,400 seconds | Uniform with replacement |

Weights are frozen; no test result selects a configuration. Both all-event and new-node populations are evaluated completely under the archived release question contract. Their populations overlap, so they are not independent datasets. We reset each condition's random seed, but changing fanout changes recursive draw consumption: this is paired **question** evaluation, not exact random-number coupling across all conditions.

**Measure accuracy.** Average precision (AP) summarizes ranking precision at observed positive retrievals. The released paper comparison averages AP over batches. The intervention table pools all predictions within each population, computes one AP per seed, then averages those ten values. These aggregations answer different questions and must not be substituted. The paper's 95.34% and 93.99% targets belong only to the released batch-mean comparison.

**Measure speed.** Synchronize the GPU before and after each measured inference pass. Include temporal sampling, feature gathering, model computation, host transfers and prediction collection. Report positive questions per second, each accompanied by one negative. Rotate condition order across seeds. This is one full timing pass per condition/checkpoint; variation includes checkpoint and hardware scheduling differences. It is not a statistical confidence interval for production latency.

[[RESULTS]]

[[FIG:results]]

**Interpret the result before choosing a policy.** Compare changed-minus-uniform20 AP within each seed. A smaller fanout reduces requested context; a short window removes older records; recent selection changes which interactions receive attention. A higher AP in one frozen model is evidence for that measured intervention, not proof the policy is optimal after retraining. Wikipedia is a moderate-sized graph; synthetic expansion counts do not establish performance on a billion-edge production graph.

## 6 · Reproduction boundary and lab

The [reproduction contract](../labs/l108-reproduction.md) pins source, raw data, processed arrays, splits, ten checkpoints, configuration, runtime and output hashes. It provides local and GPU commands, the full training route inherited from L103, a budget ledger and independent reconstruction of metrics from saved predictions.

> **Scope check.** The selected Wikipedia evaluation replay is complete when all ten outputs pass the recorded checks. L108 does not retrain those checkpoints. Modern-runtime replay and close table values do not establish historical identity or reproduction of the entire TGAT paper. Reddit, node classification, fresh sampler-specific training, live Colab and deployment have separate unrun/unchecked statuses.

Open the [student notebook](../labs/0108-temporal-neighbor-sampling.ipynb), [executed solution](../labs/html/0108-temporal-neighbor-sampling.html), or [downloadable solution](../labs/solutions/0108-temporal-neighbor-sampling.ipynb). The notebook is standalone: portable diagrams, visible sampler, complete TGAT model/trainer, three TODO/CHECK tasks, raw-data authentication and a full-data sampler benchmark. The expensive ten-checkpoint GPU evaluation is a separately executed author experiment, with an explicit rerun command.

**TODO 1:** implement half-open window boundaries, including ties and singleton histories. **TODO 2:** implement uniform replacement and recent padding. **TODO 3:** implement the neighbor-expansion budget and explain why it is not exact model cost. Then run the full-data reference-versus-batched equality check: the first two TODOs are on that executed reference path.

## EXIT · defend one temporal batch

Without opening your notes, submit:

1. Your three implementations and a new boundary fixture that catches a plausible bug.
2. A two-hop trace showing a root cutoff, selected edge time and next-hop cutoff.
3. The measured equality check, throughput ratio, index bytes and a warning about what the timing excludes.
4. A sampling policy recommendation supported by paired AP and measured inference time, plus one reason it may change after retraining.
5. A sentence distinguishing source replay, inference intervention, full-paper reproduction and your own learner evidence.

[[TEACHBACK]]

**Spacing:** tomorrow, derive [cutoff − W, cutoff) and explain the cache counterexample without notes. In one week, implement the boundary test from a fresh example. Prepared material remains **PENDING_WRITTEN_DEFENSE** until you supply learner evidence.

**Primary reading:** [Xu et al., TGAT §3.4 and Appendix A.6](https://arxiv.org/html/2002.07962v1). Read it alongside the [pinned released sampler](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/9293d10d1943c4bd4a186337cf38ba98e4c8bb99/graph.py), watching for differences between intended recency and actual slicing. Ask the teaching agent about any unclear code, boundary or measured result; bring your EXIT defense for critique.

[Quick reference](../reference/temporal-sampling.html) · [Previous: snapshot methods](0107-snapshot-methods.html) · **Next: L109** maps these query cutoffs to database `created_at` and `observed_at` semantics.
