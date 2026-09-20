<div class="package-links"><strong>Your tangible win:</strong> train a three-type GNN with a real heterogeneous NeighborLoader, then prove which parts of its computation a mini-batch preserves.<br><a href="../labs/0098-hetero-mini-batching.ipynb">Student notebook</a> · <a href="../labs/solutions/0098-hetero-mini-batching.ipynb">Executed solution</a> · <a href="../labs/html/0098-hetero-mini-batching.html">Read the solution</a> · <a href="../labs/l098-reproduction.md">Full reproduction contract</a></div>

## 1 · Retrieve before reading — 5 minutes

[[WARMUP]]

Write answers before opening the feedback. (1) In L096, does local row 0 identify an entity without a table type? (2) For the edge orders → customer, whose representation changes after one message-passing layer? (3) Does L097's negative sampler choose the neighbors needed to compute an embedding? (4) Why should training exclude a future order even if its customer ID is already known?

<details><summary>Check after committing your answers</summary><p>Identity is (table type, row ID). Source orders send messages to destination customers. Negative sampling chooses contrasting targets; neighborhood sampling chooses encoder inputs. A known entity can still acquire information that was unavailable at prediction time. Filtering training labels alone does not remove that information.</p></details>

[Lesson 96](0096-multi-relational-data.html) gave every row and FK a typed identity. [Lesson 97](0097-negative-sampling.html) separated training targets from evaluation candidates. Neither tells us how to compute an embedding when the graph is too large for one forward pass. This lesson supplies that missing execution mechanism. Recall the expanding neighborhoods of [GraphSAGE](0083-graphsage.html) and the partition approach of [Cluster-GCN](0089-sampling-at-scale.html): here the unit of work is a set of typed seed nodes and their incoming dependencies.

**Route:** 25 minutes for sections 2–6; a separate lab session for implementation, execution and written defense. The mission connection is direct: RelBench entity models use a heterogeneous neighbor loader to turn database graphs into trainable batches. Our fixture shares that shape, but is not a RelBench dataset or benchmark reproduction. [Inspect the pinned RelBench entity example](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/examples/gnn_entity.py).

## 2 · Define the query before defining the batch

Our complete generated database has **24 customers, 144 order events and 12 products**. Every customer has six orders, on days 1 through 6. An order points to its customer and product; explicit reverse stores permit information to travel back. Each type has four features. A customer's target is the sign of the average first product feature among its orders on days 1–5. This synthetic historical target makes the useful two-hop route explicit. It is not a future churn label.

| Store | Meaning | Shape before cutoff filtering |
|---|---|---|
| `customer.x` | Static customer covariates | [24, 4] |
| `orders.x`, `orders.time` | Event covariates and availability day | [144, 4], [144] |
| `product.x` | Static product covariates | [12, 4] |
| orders → customer, customer → orders | Buyer FK and reverse view | each [2, 144] |
| orders → product, product → orders | Product FK and reverse view | each [2, 144] |

For ordinary training, remove edges touching day-6 orders in **both** directions. Row IDs stay stable; inaccessible orders remain isolated. There are now 480 directed edges. Customers 0–15 are fitting seeds, 16–19 validation seeds, and 20–23 test seeds. Products are shared static context. This is a transductive feature setting: an input graph need not contain only training entities, but no held-out labels enter the fitting loss. Features are generated without using labels; the target is created afterward.

**Prediction:** how many order neighbors can customer 7 have at cutoff 5? Five. Its six-order total is not the degree the legal encoder may use. Merely calling a sampler on the unfiltered graph would not enforce this boundary.

## 3 · Model architecture — expand backward, compute forward

[[ARCHITECTURE]]

> **In plain terms.** Each destination receives one average message per relationship and keeps a transformed copy of its own features.

For each layer and destination type τ, the visible encoder computes

```text
m_r(v) = mean { h_u : (u, r, v) is present in this batch }
h'_v = ReLU(W_root,τ h_v + b_τ + Σ[r ends at τ] W_r m_r(v))
logit_v = wᵀ h_v^(2) + b, for customer v
```

Here h is a node’s feature vector; W is a learned matrix, b a learned offset, r an edge type and τ a node type. ReLU replaces negative coordinates with zero. A logit is the real-valued score before the sigmoid converts it to a binary probability. BCE is binary cross-entropy: the loss penalizes a confident probability assigned to the wrong label.

An empty relation contributes a zero mean. Each of four edge types has its own message matrix; each of three node types has its own root transform. A root term appears once per destination type, rather than once per relation. Layer widths are 4 → 16 → 16; the customer head returns one logit. The same parameters are reused across batches. This transparent mean encoder is a course model, not an R-GCN or HGT reproduction.

To predict a customer, the first sampling hop follows **incoming** order → customer edges backward to orders. The second reaches products through product → orders and customers through customer → orders. The resulting dependency graph is evaluated in the opposite direction: product features affect order states at layer 1, which affect the customer's layer-2 state. Two sampled hops support two layers. A one-hop loader does not provide the two-hop dependency closure of this model.

At training time, compute BCE on fitting seeds, backpropagate and update parameters. At validation and test time, freeze parameters and use all legal neighbors. The lab retains the checkpoint with lowest validation BCE, then evaluates the four test customers once. This keeps the inference rule fixed across training fanouts.

## 4 · Three IDs that must not be confused

Imagine `batch['orders'].n_id = [42, 44, 43]` and `batch['customer'].n_id = [7]`. A local order → customer edge `(1, 0)` means global order 44 → global customer 7. It does not mean original order 1 → original customer 0. Node IDs are global only **within their type**.

> **Your implementation task:** recover each endpoint through its own type’s `n_id` map, then stack the restored source and destination rows. The CHECK tests unequal local/global IDs so accidental identity mappings cannot pass.

`e_id` identifies the edge in the original relation store, including when the loader internally reorders adjacency. The lab checks recovered endpoints against those edge IDs for every sampled edge. `input_id` instead indexes entries in the input query list; repeated entity IDs at different times can have different input IDs. **Local node index, per-type global node ID and query index are different objects.** [PyG loader API and source](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html).

## 5 · A seed count is a loss boundary, not a node count

```python
loader = NeighborLoader(
    graph,
    input_nodes=('customer', train_ids),
    num_neighbors={edge: [2, 2] for edge in graph.edge_types},
    batch_size=4, shuffle=False, replace=False,
    subgraph_type='directional', num_workers=0,
)
for batch in loader:
    loss = seed_loss(model(batch), batch)  # Your implementation in the lab
```

PyG places the input seeds first in their type's node store. B counts those queries, while the batch also contains context nodes. Our two-hop buyer fixture does not add other customers beyond its seeds; a richer customer graph or deeper sampling can. Always use the seed prefix anyway. The adversarial CHECK gives a context customer an extreme wrong label and verifies that it produces no direct loss gradient.

For an audit that compares accumulated mini-batch gradients with one full-graph **mean** loss, multiply each batch mean by B / total_seeds. Do not take an unweighted average of batch means when the final batch is smaller. Also, do not update parameters between audit batches: multiple optimizer steps are a different optimization trajectory from one full-batch step.

A fanout of 2 is a budget **per destination, per relation, per hop**, not two nodes for the entire graph. Our product and customer reverse relations each supply one predecessor per order. Most approximation therefore comes from sampling only some of the five legal orders per customer. Uniform means without replacement preserve a one-hop mean in expectation, but ReLU and a second layer prevent that observation from proving unbiased final logits or gradients.

[[INTERVENTION]]

**Worked route:** products with scalar values 2, 8, −4 send to three separate orders, which send to a customer. Ignore root terms and nonlinearities for this calculation. All three give (2+8−4)/3 = 2; selecting the first two gives 5. The widget uses a deterministic prefix to make arithmetic inspectable. The actual lab uses random uniform native sampling, not prefix sampling.

## 6 · One customer, two prediction times

A training mask says which targets may contribute to a loss; it does not time-filter context. For a single shared cutoff our explicit snapshot is easy to audit. For queries at different times, the lab also executes the real temporal loader:

```python
input_nodes = ('customer', torch.tensor([0, 0]))
input_time = torch.tensor([2, 5])
# Other settings are the same; all neighbors are requested.
loader = NeighborLoader(graph, input_nodes=input_nodes,
    input_time=input_time, time_attr='time', disjoint=True,
    num_neighbors={e: [-1, -1] for e in graph.edge_types},
    batch_size=2, temporal_strategy='uniform')
```

The native temporal result has two customer seed entries with the same `n_id=0` but different `batch` assignments. Event IDs [0,1] belong to the first query and [0,1,2,3,4] to the second. Each copy remains in its query's component. The lab compares the two logits with independently evaluated snapshots at days 2 and 5, checks endpoint component membership, and perturbs future event features by +10,000 without changing the earlier prediction.

The fixture defines availability as time ≤ cutoff and static customer/product timestamps as zero. In a real task, establish whether equality is allowed and whether a supposedly static feature was already available. Sampling does not undo a leaked aggregate, a future-updated product field or a feature normalization fitted on held-out data. Temporal query labels belong to query rows; our temporal audit checks logits only and does not reuse one entity label as ground truth at both times. [Native CPU sampler source](https://github.com/pyg-team/pyg-lib/blob/edc9e2a88d1c5d0953b5f69c98b8365597c6b699/pyg_lib/csrc/sampler/cpu/neighbor_kernel.cpp).

## 7 · Full reproduction of the declared experiment

[[PREDICT]]

No paper score is assigned to L098. **Full reproduction here means every declared course configuration**, with the actual native sampler, an independent oracle and complete fresh training. It does not establish RelBench benchmark performance.

The correctness track runs 32 generated databases × batch sizes 1, 4 and 24. Every configuration covers all 24 customers exactly once, compares all seed logits with a separately implemented dense adjacency forward, and compares weighted accumulated gradients. Same weights, legal edges, root terms and two-hop depth are held fixed; there is no dropout or batch normalization. Frozen tolerance is 2×10⁻⁶. This establishes full-neighbor computation parity under this contract, not arbitrary sampled-GNN equivalence.

The training track fixes graph seed 98, train/validation/test identities, width 16, 40 epochs, Adam learning rate .02, no weight decay and batch size 4. It runs all neighbors, fanout 2 and fanout 1, each with initialization seeds 0, 1 and 2. Validation BCE chooses the earliest strictly best epoch. Every arm uses full-neighbor validation/test evaluation. Full traces, test logits and the selected model weights are saved. The notebook restores a trained model from JSON arrays and checks its predictions. Batch nodes/edges are workload proxies, not measured peak memory or wall-clock speedups.

[[RESULTS]]

**Interpretation:** full neighbors reproduce the seed computation; limited fanout trades context for smaller batches and changes it. Four test examples cannot support a generalization ranking. Three initializations share one graph and are not three independent datasets. The extra logit-gap diagnostic is one sampled evaluation per run, not an expectation estimated over many neighborhoods.

Source hashes, actual package versions, runtime instructions, all deviations and execution records are in the [protocol](../labs/l098-reproduction.md), [source manifest](../labs/_sources_l098.json) and [complete results](../labs/_experiment_l098_results.json). No cached reference is substituted for execution. Colab and deployment are separate delivery states.

## 8 · Implement, falsify, defend

In the student notebook implement **relation_mean**, **seed_loss** and **global_edges**. Each function is called by the real trainer or complete audit; each has an immediate CHECK. The model, graph generator, native loader, dense oracle and training loop are visible in the notebook. Predict a CHECK's answer before running it. Then execute the entire declared experiment from a clean directory.

Submit the fresh JSON result and explain, without looking up the code:

1. Why product → order → customer requires two message-passing layers even though the order FK originally points toward the product.
2. How to recover original typed endpoints from a local edge and why `input_id` is different.
3. Why seed-only loss and a legal-time input graph enforce separate boundaries.
4. Why full-neighbor **gradient accumulation** can match full-batch gradients while sequential SGD updates need not.
5. Which evidence would be missing if someone renamed the nine toy fits “full RelBench reproduction.”

[[TEACHBACK]]

**Status: PENDING_WRITTEN_DEFENSE.** Prepared material and successful teacher runs are not learner mastery. Tomorrow, reconstruct the two-hop route and seed-loss code from memory. In three days, explain why sampled logits can be biased even when a one-hop mean is unbiased. Ask the agent to quiz you or diagnose any failed CHECK.

Next, curriculum lesson 99 holds the graph and protocol fixed while comparing R-GCN with HGT. Lesson 100 requires you to defend a complete heterogeneous pipeline. Today removes a confound from both: architecture comparisons are only interpretable when batching, target identity and information availability are understood.

**Primary reading:** read the [pinned PyG NeighborLoader source](../labs/sources/l098/neighbor-loader-2.6.1.py) beside the [RelBench entity training example](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/examples/gnn_entity.py). Use the [batching reference sheet](../reference/hetero-batching-contract.html) for review.
