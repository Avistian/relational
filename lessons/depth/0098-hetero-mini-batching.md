## The big picture · a batch is a query-specific computation graph

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 098 / DEPENDENCIES AND TARGETS</p><p><strong>Bring forward:</strong> L096 made table-local identity explicit; L083 expanded graph dependencies from roots. <strong>Follow:</strong> seed query → eligible typed neighbors → local maps → shared layers → seed outputs. <strong>Carry forward:</strong> L100 tests these contracts on a real graph before interpreting a model comparison.</p><p><a href="0096-multi-relational-data.html">L096: database keys</a> · <a href="0083-graphsage.html">L083: outward sampling</a> · <a href="0100-heterogeneous-gnn-checkpoint.html">L100: integrated audit</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:098-batch]]

### Draw the arrows in both directions

To predict for a customer with a two-layer encoder, first discover the customer's incoming neighbors, then the inputs those neighbors need. Dependency discovery moves outward from the seed. Computation moves inward: original features form first-layer support states, which form the seed's second-layer state. Reversing the sampling list is not enough; the actual edge direction and layer dependencies must agree.

Read the [pinned NeighborLoader source](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html) for `input_nodes`, typed fanouts, seed prefix and `n_id`. These are data-routing contracts. They do not select a neural architecture or automatically enforce your task's temporal boundary.

### Count a dependency tree before trusting a batch size

In a single-relation example with B roots and fanouts f₁/f₂, the expanded occurrences are bounded by `B + B f₁ + B f₁ f₂` before deduplication. With B=4, f₁=3, f₂=2, that is 40 occurrences. It is not necessarily 40 unique nodes. Shared neighbors can reduce the count; disjoint temporal queries can retain multiple copies of one global entity. With several relations, each per-relation expansion contributes, so using only one relation's fanout can badly understate the workload.

This count is a planning bound, not a memory benchmark. Hidden widths, edges, attention heads, saved gradients and native sampler buffers also consume memory. The existing experiment reports batch sizes as workload evidence rather than claiming an unmeasured peak-memory reduction.

### Recover identity before recovering a label

There are now three naming systems: database primary key, full-graph type-local row, and batch-local row. The first conversion belongs to L096; the second belongs to the sampler. An edge's source index must use the source type's map, while its destination index uses the destination type's map. Valid integer ranges alone cannot catch swapping those maps.

For temporal queries, include the prediction time in the identity. Customer 10 at day 2 and customer 10 at day 5 can need different context and different target labels. One global entity ID does not identify both prediction examples. Disjoint query components can hold two copies with the same `n_id` while preserving different cutoffs. The fixture's temporal audit checks this separately from ordinary target masking.

### Derive the direct supervision boundary

Suppose a batch contains two customer seeds and three additional customer context nodes. It may compute five logits, but the declared objective averages only the first two seed losses. A context feature can receive gradient through a path into a seed prediction. Its own output logit receives no **direct** supervised gradient when it is outside the seed loss. Those statements are compatible: the graph carries useful context without treating every sampled label as a training target.

Full-neighbor equivalence needs complete receptive fields, identical weights and compatible normalization. A mean over the entire batch would change when the batch changes; node-wise operations do not introduce that particular dependency. Finite fanout is an additional approximation even when all identity and loss checks pass.

<details><summary>Predict first: do two layers always require only one sampled hop?</summary><p>No. A first-hop neighbor's first-layer state generally depends on its own neighbors. A second-hop source can therefore affect the root's second-layer output. One hop is sufficient only under an additional model-specific restriction, such as using fixed precomputed neighbor states with an explicit provenance contract.</p></details>

**Forward connection.** L099 first removes the sampler from the model comparison by using full-graph computation. L100 puts sampled training back and checks equal target exposure and paired sampled edges. Before moving on, draw one seed's complete two-layer dependency graph and mark which labels may contribute. Ask the teacher to perturb one future feature and predict whether an earlier query is allowed to change.
