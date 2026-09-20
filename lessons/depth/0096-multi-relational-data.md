## The big picture · schema fidelity comes before model capacity

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 096 / ROWS, ROLES AND IDENTITIES</p><p><strong>Bring forward:</strong> R-GCN and HGT can preserve relation meaning only if the graph builder provides it. <strong>Follow:</strong> primary keys → typed index maps → foreign-key roles → reversible graph representation → independent query audit. <strong>Carry forward:</strong> L098 introduces a second remapping from full-graph IDs to batch-local rows.</p><p><a href="0091-r-gcn.html">L091: relation transforms</a> · <a href="0093-hgt.html">L093: typed attention</a> · <a href="0098-hetero-mini-batching.html">L098: batch identities</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:096-database]]

### Decide what one node means

A customer node means a customer row. A line-item node means a complete line-item row, identified by its composite key. It does not mean a unique customer–product pair. If two lines mention the same product, their separate identities may carry separate quantities or event times. The relational graph construction described by [Fey et al.](https://arxiv.org/abs/2312.04615) motivates preserving database structure; this lesson tests construction directly, before adding a predictor.

There is no training objective in this diagram. The immediate output is a representation whose paths recover declared database relationships. Adding a GNN before checking that property can make a data error look like an architectural weakness. The independent SQL oracle below supplies a concrete falsification test.

### Trace one fact through three names

Take line key `(100,2)`, whose quantity is 3. In the line table's sorted index map it becomes local row 1. Its order key 100 becomes local order row 0. That order's buyer key 10 becomes local customer row 0. The line's product key 10 separately becomes local product row 0. The two zeros represent different entities because the table types differ.

Read the path backward to recover meaning: line local row 1 → line key `(100,2)` → order key 100 → customer key 10, plus product key 10. If the round trip cannot recover those names, a tensor shape check is not enough. L100 will repeat the same reasoning when batch row 1 maps to a different full-graph node for each type.

### Separate three questions that share the same endpoints

Customer 10 has two line paths to product 10, with quantities 2 and 3. Existence is 1; path count is 2; quantity sum is 5. A binary endpoint graph, a count matrix and an attribute-weighted path sum therefore carry different answers. A mean of quantities is 2.5. All four values can be mathematically correct while answering different questions.

This explains a limit of mean aggregation from L091. A layer that averages identical neighbor states can lose multiplicity. It does not become SQL `SUM(quantity)` because its receptive field spans the same tables. To compare the two, specify both the path structure and the reduction. You might retain explicit count features, use sum aggregation, or preserve event nodes; each choice needs its own test.

### Missing references and isolated rows are different cases

A nullable referrer means there is no referrer edge for that order. It does not delete the order. A valid customer with no orders remains a node because row existence does not depend on appearing in an edge list. A non-null referrer key with no matching customer is an integrity failure under this contract. Treating all three situations as “drop missing edges” hides different semantics. The [SQLite foreign-key reference](https://www.sqlite.org/foreignkeys.html) anchors the NULL and parent-key rules used here.

<details><summary>Try first: can sorting every table make database keys safe tensor indices?</summary><p>Sorting makes an index map deterministic. It does not make a key numerically equal to its row position. Store and use the map for each type, including complete tuples for composite keys. A key permutation should relabel the graph without changing the recovered query answer.</p></details>

**Forward connection.** Referential integrity says a parent exists; temporal availability says a prediction was allowed to use it. Neither implies the other. Before L098, write the identity of one future query as `(entity, cutoff)`, list the legal supporting rows, and explain why a reverse edge inherits the availability of its underlying fact. This prepares the mission's point-in-time relational prediction requirement without claiming that the synthetic SQL audit is a benchmark result.
