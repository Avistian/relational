<div class="package-links"><strong>Your tangible win:</strong> turn four SQL tables into a typed graph and prove that it preserves a declared join and aggregation.<br><a href="../labs/0096-multi-relational-data.ipynb">Student notebook</a> · <a href="../labs/html/0096-multi-relational-data.html">Executed solution</a> · <a href="../reference/schema-graph-contract.html">Reference sheet</a> · <a href="../labs/l096-reproduction.md">Reproduction contract</a></div>

## 1 · Retrieve before reading

Without opening L095, write answers to these questions. Why can customer 10 and product 10 be different nodes? What information disappears when repeated interactions become one binary edge? Does reversing an edge make its information safe to use in a training graph?

<details><summary>Check after you have written</summary><p>Identity includes the node type. A binary edge discards event multiplicity and event attributes unless these are explicitly retained elsewhere. Reversing a held-out link still exposes that link. If any answer was uncertain, revisit <a href="0095-bipartite-graphs.html">L095</a> before building.</p></details>

**The bridge.** In [L034](0034-relational-data-without-rdl.html), joins and grouped features let a flat model use relational information. [L091](0091-r-gcn.html) gave different edge roles different transformations. Here we decide where those roles and entities come from. This prepares the database-to-graph construction in L122. Our mission is to assess relational learning rigorously; preserving the meaning of the input comes before comparing model scores.

**Core route:** read sections 2–6, then implement the three notebook tasks. Allow about 25 minutes for the explanation and 25 for the lab. Sections 7–9 define the audit and the next retrieval session. Ask the agent whenever you cannot explain a step from the data itself.

<!-- depth-walkthrough:start -->
## The big picture · schema fidelity comes before model capacity

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 096 / ROWS, ROLES AND IDENTITIES</p><p><strong>Bring forward:</strong> R-GCN and HGT can preserve relation meaning only if the graph builder provides it. <strong>Follow:</strong> primary keys → typed index maps → foreign-key roles → reversible graph representation → independent query audit. <strong>Carry forward:</strong> L098 introduces a second remapping from full-graph IDs to batch-local rows.</p><p><a href="0091-r-gcn.html">L091: relation transforms</a> · <a href="0093-hgt.html">L093: typed attention</a> · <a href="0098-hetero-mini-batching.html">L098: batch identities</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

<figure class="model-map"><div class="model-map-scroll" tabindex="0" role="region" aria-label="Scrollable architecture: Database → graph · preserve rows before learning"><img src="../assets/architectures/096-database.svg" alt="Foreign-key roles define edges; composite keys preserve distinct events. Construction is not training. A valid foreign key establishes referential integrity; it does not establish prediction-time availability."></div><figcaption>Foreign-key roles define edges; composite keys preserve distinct events. Construction is not training. A valid foreign key establishes referential integrity; it does not establish prediction-time availability.</figcaption></figure>

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

<!-- depth-walkthrough:end -->

## 2 · Start with four tables and one question

Our complete, synthetic database contains customers, products, orders and line items. This is a Tier C construction lab, not a sampled extract from a hidden larger benchmark. The question is: **how many units of each product has each customer bought?** First trace the answer by hand; then make SQL and the graph agree.

| Table | Primary key | Other columns | Rows |
|---|---|---|---:|
| customer | customer_id | none in this fixture | 3 |
| product | product_id | none in this fixture | 3 |
| orders | order_id | buyer_id, nullable referrer_id | 3 |
| line_item | (order_id, line_no) | product_id, quantity | 4 |

Customers are 10, 30 and 90. Products are 10, 50 and 80. Order 100 belongs to customer 10 and was referred by customer 30; order 200 belongs to customer 30; order 300 belongs to customer 10. The last two have no referrer. Order 300 has no lines. Customer 90 and product 80 have no links.

| order_id | line_no | product_id | quantity |
|---:|---:|---:|---:|
| 100 | 1 | 10 | 2 |
| 100 | 2 | 10 | 3 |
| 100 | 3 | 50 | 1 |
| 200 | 1 | 50 | 4 |

**Predict:** customer 10 bought five units of product 10. The two matching rows are distinct because line numbers differ. A primary key can be a tuple; uniqueness applies to the complete tuple, not to each component separately. Foreign keys constrain a child reference to a matching parent key. They do not require every parent to have children. See [PostgreSQL's constraints reference](https://www.postgresql.org/docs/18/ddl-constraints.html#DDL-CONSTRAINTS-PRIMARY-KEYS).

## 3 · Specify the graph before writing tensors

[[SCHEMA_FIG]]

The course mapping is deliberately explicit: one node type per table, one node per row, and one edge role per foreign-key constraint. The relational deep learning paper introduces database graphs as the input to representation learning; here we isolate construction from learning. Read [Fey et al., Relational Deep Learning](https://arxiv.org/abs/2312.04615) for the broader motivation; no result from that paper is a score target in this unit.

| Forward triple: source / role / destination | Number of links | Meaning |
|---|---:|---|
| orders / buyer / customer | 3 | Order identifies its buyer |
| orders / referrer / customer | 1 | Order optionally identifies its referrer |
| line_item / order / orders | 4 | Line belongs to one order |
| line_item / product / product | 4 | Line identifies one product |

**Why preserve the role?** Buyer and referrer have identical source and destination types. One generic orders→customer relation would erase which customer did what. In R-GCN or HGT, that ambiguity would already be present before the first learned transformation.

**Why this direction?** Our forward convention is child→referenced parent, because the child stores the FK. To send messages the other way, create a separate reverse store. SQL does not force an encoder's message direction. A reverse is a view of the same fact, not a new observation. The lab uses names such as `('customer','rev_orders_buyer','orders')`, making both role and originating table explicit.

A non-null FK gives each child exactly one parent for that role. Multiple children can point to the same parent: from the parent's perspective this is one-to-many. A UNIQUE constraint on the child's FK would impose an additional restriction; an FK alone does not. A many-to-many order–product relationship is represented through line rows and two FK roles. In this fixture even the same order–product pair can occur twice.

## 4 · Resolve keys; never confuse them with tensor positions

`order_id=100` is a database key. It is not necessarily tensor row 100. Sort each table's complete primary keys and create its own dictionary. Customer 10 and product 10 both become local index zero, but remain in separate stores. The line keys `(100,1)` and `(100,2)` become local indices zero and one.

```python
customer_index = {(10,): 0, (30,): 1, (90,): 2}
order_index = {(100,): 0, (200,): 1, (300,): 2}
# orders --buyer--> customer: three columns, one per order
edge_index = torch.tensor([[0, 1, 2],
                           [0, 1, 0]], dtype=torch.long)
```

PyG indexes typed edge stores by `(source, relation, destination)`; `edge_index` holds local source and destination positions. Set `num_nodes` explicitly on every node store so rows without edges survive. The graph contains 13 nodes and 12 forward links, or 24 links with the four reverse stores. These are course-fixture counts, not an empirical paper result. [PyG 2.6.1 HeteroData](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.data.HeteroData.html).

**NULL is different from corruption.** Order 200's missing referrer produces no referrer edge. Replacing it with key 999 creates a dangling reference and must fail. The child order node remains in either valid nullable case; no fake customer is invented. For a nullable composite FK, the lab adopts SQLite's MATCH SIMPLE behavior: any NULL component omits the relationship. Required FKs reject NULL. The implementation explicitly enables SQLite enforcement before inserting data. [SQLite foreign-key semantics](https://www.sqlite.org/foreignkeys.html).

**Scope:** the converter accepts the declared schema with integer keys, composite row identity and PK references. It does not parse arbitrary SQL, implement collations or cascades, infer identity for keyless tables, or support references to alternate UNIQUE keys. Those need additional contracts. It stores database identities as audit metadata, not predictive features.

## 5 · A join is a path with identity and multiplicity

Trace line `(100,2)`. Its local node 1 points to order local node 0 and product local node 0. Order local node 0 points to customer local node 0. Recovering the audit keys gives `(customer=10, order=100, line=2, product=10, quantity=3)`.

The complete query has four paths, one for every line. Grouping by customer and product produces three rows:

| customer | product | SUM(quantity) |
|---:|---:|---:|
| 10 | 10 | 5 |
| 10 | 50 | 1 |
| 30 | 50 | 4 |

```sql
SELECT c.customer_id, p.product_id, SUM(l.quantity)
FROM customer c
JOIN orders o ON c.customer_id = o.buyer_id
JOIN line_item l ON o.order_id = l.order_id
JOIN product p ON l.product_id = p.product_id
GROUP BY c.customer_id, p.product_id;
```

Our graph evaluator follows three forward FK lookups from each line and sums its quantity. It does not call SQL or multiply a precomputed answer. The independent oracle creates a fresh SQLite database, inserts the same input rows, and runs the query. We compare full path tuples as well as sums: incorrect paths can accidentally yield the right total.

A plain join retains matching rows unless a query explicitly removes duplicates. Replacing this query with `DISTINCT customer,product` changes its result. For zero-order customers, a LEFT JOIN plus `COUNT(o.order_id)` returns zero; `COUNT(*)` counts the retained outer row. [SQLite SELECT processing](https://www.sqlite.org/lang_select.html).

**Three different answers:** existence of customer 10→product 10 is 1; its number of line paths is 2; its sum of quantities is 5. A binary projection, path count and weighted aggregation answer different questions. A mean message aggregator also does not automatically implement a SQL sum. No generic GNN is promised to reproduce arbitrary SQL semantics.

[[INTERVENTION]]

**Before touching the control:** move line `(100,2)` from product 10 to product 50. Predict both totals. Then explain why the line-node count stays four. The notebook runs this same intervention against the SQL oracle.

## 6 · Implement the three load-bearing steps

All code is visible inline in both notebooks. The student notebook contains three blanks; the solution contains their implementations. Each task has an immediate CHECK, and the final 33-database audit uses those same functions.

1. **TODO: `key_index`.** Reject missing or duplicate composite keys and build stable per-table indices. **CHECK:** `(100,1)` and `(100,2)` are distinct; repeated `(100,1)` fails; input reordering cannot change IDs.
2. **TODO: `fk_edges`.** Resolve each child reference through the parent map. Retain the child's position after skipping NULL; return `[2,0]` for an empty relation. **CHECK:** nullable rows are skipped, orphans fail, and every remaining destination is correct.
3. **TODO: `graph_query`.** Follow line→order→buyer and line→product; preserve every line, sum quantities and include zero-order customers. **CHECK:** all path tuples, all sums and all customer counts equal the independent SQL result.

The supplied graph builder applies the first two steps to all four FK roles, stores quantities, and creates exact reversed edge tensors. The SQL oracle and generated fixtures are also visible. The entire construction runs on CPU; there is no hidden remote trainer or saved model behind the lab.

**Representation is not a prediction task.** If predicting future purchases, you still need an as-of cutoff, allowed features and a label window. Structural validity does not prove availability at prediction time. Removing a direct customer–product target edge would not hide an order/line path that still reveals the same future purchase. L097 next studies negative sampling; L098 introduces mini-batching; L122 returns to full relational entity graph construction. Those future units are not prerequisites for this lab.

## 7 · Reproduce the whole declared experiment

[[RESULTS]]

The full target is the supplied 13-row database plus 32 deterministically generated databases (seeds 0–31). Every generated database has seven customers, six products and twelve orders; all its generated line rows are used. The last customer and product are deliberately unreferenced. There is no subsampling and no unexecuted larger run being substituted with this fixture suite.

Reproducibility here means **the same inputs, mapping, query and expected outputs**, with recorded source/data hashes and environment. Separate tests cover duplicate and NULL keys, dangling references, empty relations, optional composite FKs, reverse stores, 20 row-order permutations and the product reassignment. A finite suite is evidence for the declared implementation; it is not a proof for every SQL database.

| Evidence | Claim boundary |
|---|---|
| Complete course suite | All 33 databases; paths, totals and outer counts checked against SQL |
| Paper-result reproduction | NOT_APPLICABLE_NO_MODEL_PAPER: curriculum specifies a bridge unit |
| Historical predictive-score parity | NOT_ESTABLISHED: no trained model or score target |
| Local standalone notebook | See execution evidence; does not imply live Colab |
| Live Colab / deployed site | NOT_CHECKED |
| Learner mastery | PENDING_WRITTEN_DEFENSE |

See the [exact commands and deviations](../labs/l096-reproduction.md), [behavioral checks](../labs/_verify_l096_results.json), [full query outputs](../labs/_experiment_l096_results.json), and [notebook execution](../labs/_execution_l096_results.json). This respects the request for full reproduction by completing the entire specified construction experiment and naming precisely what has no applicable paper target.

## 8 · Written defense and spaced retrieval

Before opening the solution, submit your graph spec and these answers to the agent:

- List all node types, all forward roles and their reverses. Explain why two orders→customer roles cannot be merged.
- Trace line `(100,2)` from original PKs to local IDs and back. Distinguish existence, path count and quantity sum.
- Explain NULL versus orphan, and why customer 90 and product 80 must remain nodes.
- Show that your query keeps customer 90 with zero orders. Explain why COUNT(*) would differ.
- Report exactly what was reproduced and one limitation of the integer-key converter.
- Describe how a future order could leak a purchase target through an indirect path.

**Pass criterion:** correct identities and roles, four retained paths, totals 5/1/4, zero-order preservation, correct failure behavior, and a bounded reproduction claim. Execution alone is insufficient. Revisit tomorrow with the tables hidden; in one week, add a second nullable FK to product and explain how your spec changes before implementing it.

## 9 · Primary reading

Start with [SQLite foreign keys, sections 1–2](https://www.sqlite.org/foreignkeys.html): reproduce the distinction between a missing parent and a NULL reference. Use [PostgreSQL PK/FK constraints](https://www.postgresql.org/docs/18/ddl-constraints.html) for the schema terminology and [PyG's typed container](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.data.HeteroData.html) for storage. Then read [Fey et al.](https://arxiv.org/abs/2312.04615) with one question: which information must a database graph preserve before learned aggregation begins?
