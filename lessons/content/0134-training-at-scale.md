<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Where this lesson fits">
<p class="sequence-eyebrow">From Lesson 133 to this lesson</p>
<p>The layer equation is fixed. Scaling decides which query-owned occurrences reach it and measures the resulting computation.</p>
<details><summary>Quick prerequisite reminder</summary><p>Throughput is completed queries divided by elapsed time. GPU allocated memory tracks live tensors; reserved memory also includes allocator-held space. Fanout gives an expansion bound, not measured memory.</p></details>
</aside>
<!-- sequence-review:end -->

<p class="stream-lead">A graph can have millions of rows while one training step touches thousands. The hard part is controlling which thousands—and measuring the cost without changing the question.</p>

**Your win:** configure a temporal heterogeneous mini-batch, prove that its messages belong to the right prediction query, and defend a measured throughput/memory trade-off.

**Route:** 5 minutes of retrieval → 15 minutes tracing a batch → 20 minutes of live code → 10 minutes defending the measurements. The full reproduction is an author/reference experiment; executing it is optional and does not replace your answers.

[Student notebook](../labs/0134-training-at-scale.ipynb) · [Executed reference](../labs/html/0134-training-at-scale.html) · [Solution notebook](../labs/solutions/0134-training-at-scale.ipynb) · [Quick reference](../reference/training-at-scale.html) · [Reproduction commands](../labs/l134-reproduction.md)

## 1 · The layer is defined. Its input still needs a budget.

In [Lesson 133](0133-hetero-conv-reg.html), you opened the relation-specific transforms and the sum across incoming relations. Applying that layer to the whole database at once is unnecessary for seed-query supervision. We can first select the context required by a small set of predictions, then run the same message-passing computation on the sampled coordinates. This is the bridge from correct layer arithmetic to the mission's practical question: can a relational model deliver evidence within our compute budget?

**Retrieve before reading:** What identifies a prediction besides its entity ID? Why can the same database row occur twice in a batch? Which rows receive the supervised loss?

<details><summary>Check your retrieval</summary>A prediction is keyed by entity and cutoff. Two queries can refer to one entity at different times, requiring different neighborhoods; disjoint temporal sampling keeps their occurrences separate. The head's first B seed outputs receive the B query labels. Context rows provide messages but are not extra labelled examples.</details>

The primary reading is the [RelBench v1 paper, §4 and Appendix B](https://arxiv.org/html/2407.20060v1), followed by the [pinned released loader configuration](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py). This lesson preserves that release's inclusive node-time cutoff and uniform strategy. A stricter deployment rule is a different experiment.

## 2 · Trace one batch from query to optimizer

[[FIG:batch]]

**Diagram trace.** Follow the CPU-to-GPU handoff. Match n_id, batch and input_id to their separate identities before tracing the seed-only loss. On a narrow screen, scroll the figure sideways.

The full graph's adjacency and stored row features live on the host. A seed row contributes an **entity ID, query time, and label**. `NeighborLoader` follows incoming typed edges, filters candidates using that query's cutoff, and samples up to the fanout for each relation at each hop. `n_id` maps sampled coordinates back to database rows. `batch` identifies the query owning each occurrence; it is not a table ID. `input_id` identifies the original task row, so labels remain correct even when seeds repeat or shuffle. PyG's temporal sampler automatically uses disjoint sampling. [PyG 2.6.1 source](https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/sampler/neighbor_sampler.py)

The sampled features move to the device. The model encodes those rows, adds relative-time information, applies two heterogeneous layers, and predicts only the first B rows of the entity table. Backpropagation traverses this sampled computation. Increasing fanout changes the model's information and its training trajectory; it is not merely a speed switch.

**Trace:** Query 0 is `(user 7, day 5)` and query 1 is `(user 7, day 10)`. A post created on day 8 can occur only in the second context. Deduplicating user 7 across queries without preserving context ownership would mix two different prediction problems.

## 3 · Fanout multiplies along paths and across relations

For one relation and two hops, the no-collision expansion is `B × (1 + f₁ + f₁f₂)` node occurrences. A heterogeneous schema can expand through several incoming relations at every hop. You must follow the schema, not multiply a homogeneous formula by the total number of relations.

Consider `orders → users`, `items → orders`, and `users → orders`. Starting with B=2 user queries and fanouts [3,2], the typed frontiers contain 2 users, then 6 orders, then 12 items plus 12 users: **32 occurrences and at most 30 sampled edges**. Actual counts can be smaller because of degree limits, time filtering and shared nodes within a query. Occurrences across different queries remain separate.

[[FIG:frontier]]
[[WIDGET]]

**TODO 1:** implement the schema-aware bound. The function below counts an incoming relation once for each receiving occurrence. It assumes finite nonnegative fanouts and directional neighbor expansion; `-1` means all neighbors and has no finite fanout-only bound.

[[BOUND_CODE]]

A float32 hidden matrix costs `4 × N × width` bytes. That is one tensor, not peak GPU memory. Layer activations, relation outputs, gradients, optimizer state, indices, allocator reservations and transient workspace also count. Host adjacency, sampler CSC conversion and text preprocessing have separate costs. A lower bound on one matrix is not an OOM guarantee.

## 4 · Sampling is an information contract

**TODO 2:** reject a dated node newer than its own root cutoff and any edge joining occurrences owned by different queries. Equality passes because the released experiment uses `node_time ≤ query_time`. A static table with no timestamp is marked explicitly; this does not establish historical availability.

[[AUDIT_CODE]]

Our author runs also compare sampled node times and edges to their original global identities. A batch can satisfy `time ≤ cutoff` while holding the wrong timestamp or foreign-key edge. These checks complement the learner's ownership check rather than replace it. The same learner function runs against the real F1 and scale batches.

**Units are part of the contract.** The archive and task timestamps were probed and both use nanoseconds. The operator nevertheless explicitly converts both to `datetime64[s]` before taking integer counts; it does not assume all future archives use the same raw integer unit. A four-unit regression checks seconds, milliseconds, microseconds and nanoseconds.

**Predict:** if a day-8 post belongs to query 1, should an edge from that occurrence to query 0's user pass? It must fail even though day 8 is earlier than query 1's cutoff. Both endpoints of a message must belong to one prediction context. This extends [Lesson 123's temporal graph](0123-temporal-heterogeneous-graphs.html) contract into the sampled coordinates used by training.

## 5 · Measure the pipeline you actually run

GPU operations are asynchronous. Wall-clock timing around a forward call without synchronization can measure enqueue time. The scale operator synchronizes before and after device transfer and the combined forward/backward/optimizer step. It reports loader construction, warmups and audit time separately. It uses one loader worker, no prefetch workers, and a fixed order; this is a reproducible baseline, not a claim of maximal throughput. [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

**TODO 3:** aggregate throughput as total seed queries divided by total measured seconds. Do not average per-batch rates: small final batches and unequal durations make that biased. Return both a core rate (sample + transfer + step) and an audit-inclusive rate. Peak allocated memory measures live tensor allocations; peak reserved memory includes the CUDA caching allocator's pool. Neither is total process GPU usage.

[[SUMMARY_CODE]]

**Predict:** if sampling takes 0.8 seconds and device work takes 0.2 seconds, making device work twice as fast changes a one-second batch to 0.9 seconds—only about an 11% throughput gain. First measure which stage dominates.

## 6 · Two experiments, two claims

The **F1 lane** freshly executes five complete ten-epoch fits of RelBench v1 Table 7, `rel-f1/driver-position`, using the full released data and pinned model/trainer. It independently rescores every final prediction and checks selected outputs against the original model. The paper target is validation MAE 3.193 and test MAE 4.022; the predeclared descriptive tolerance is 0.2 on each mean. The full experiment's implementation is visible in the notebook appendix, including row encoders, graph construction, trainer and selection.

The **scale lane** retains all `rel-stack` rows through the released test cap and all resolvable foreign-key edges, with real `user-engagement` labels reconstructed by the original SQL on that same pinned archive. It deliberately replaces rich row encoders with constant-plus-log-age features and a width-32 two-layer heterogeneous predictor. It runs BCE training on the same label-blind set of 2,048 eligible task rows (one released training cutoff, roots created by that cutoff, ascending entity ID) under three fixed batch/fanout configurations. Two warmup batches perform updates before each measured pass. These are bounded training workloads, not complete epochs over the task or a reproduced benchmark score. Fixed initialization does not make different minibatch partitions equivalent optimization trajectories.

[[RESULTS]]
[[FIG:measurements]]
[[FIG:scores]]

**Interpret before looking at throughput:** the graph census describes database scale; sampled occurrence counts describe per-step computation. The prefix, small features, warmup updates, single measurement pass and configuration order limit generalization to other queries and full models. We report descriptive timings, not confidence intervals or a winning hyperparameter. No timing-to-accuracy claim follows.

## 7 · Make scale fit a scientific budget

Our total cap is **USD10** including seeds, retries and validation. We reserved USD2 for F1/validation, USD5 for the scale lane and USD3 for overhead/recovery. At the checked base prices, T4 + two CPU cores + 32 GiB is approximately USD0.940464/hour. The runner reserves maximum worker time before dispatch and disables automatic retries. A pilot informs whether to dispatch the full F1 set. [Current price source](https://modal.com/pricing)

A stopped or shortened run retains its actual status. It cannot be turned into a full reproduction by extrapolating its speed or quoting another lesson's scores. The release's statistics use the database through the test cap, historical seeds/runtime are unavailable, and ingestion histories are missing. We preserve those facts in the [protocol ledger](../labs/l134-reproduction.md). Whole-paper parity and historical identity remain NOT_ESTABLISHED.

## EXIT · Defend a batch, not just a score

1. For B=4 and fanouts [2,3] on the widget's schema, derive every typed frontier and the one-matrix float32 storage at width 128. Explain why it is not peak memory.
2. Draw the duplicated user-7 contexts at days 5 and 10. Mark a cross-query edge that the audit must reject, and name one real-world leakage risk the timestamp check cannot resolve.
3. From the measured table, choose a configuration for a stated memory ceiling and explain the dominant time component. Distinguish the measurements from a claim about accuracy or whole-task runtime.
4. State precisely what was fully reproduced, what was a bounded training workload, and what remains unknown. Submit your three passing functions, arithmetic and written defense.

Return after **1, 7 and 30 days**: reconstruct the relation frontier without notes, diagnose a mixed-query edge, and explain why averaging batch rates fails. Lesson 135 will turn these measured costs into a fair tuning budget.

Ask the agent follow-up questions about any step. Prepared lessons and author-run checks leave your mastery **PENDING_WRITTEN_DEFENSE**.

<!-- sequence-next:start -->
**Carry this forward.** Use the measured costs to reserve and enforce a fair tuning budget in Lesson 135. [Continue to Lesson 135](0135-tuning-on-reg.html).
<!-- sequence-next:end -->
