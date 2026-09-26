## 1 · The next bottleneck is computation

In [lesson 112](0112-ogb-gcn-reproduction.html), you followed one full-graph GCN from citation edges to a validation-selected prediction. Every gradient step propagated through all 169,343 papers. That made the experiment easy to define: one graph, one objective, one optimizer step per epoch.

Now ask a practical question: **what must change when the intermediate node representations are too large to train together?** This lesson's tangible win is to construct an induced mini-batch, trace exactly which messages it removes, and defend a compute-and-accuracy report. The larger experiment uses **ogbn-products**, with 2,449,029 nodes. These scaling skills prepare you for relational entity graphs in Year 4, where a row prediction can depend on millions of connected records.

> **In plain terms.** Mini-batching decides which part of a graph participates in one update. It can reduce the amount of work held in memory. It also changes what the model sees during that update.

[[WARMUP]]

**Cold retrieval.** Before reading onward: (1) Does masking the loss remove test nodes from propagation? (2) Why must a GCN checkpoint include batch-normalization buffers? (3) What is cut when a cluster is processed alone? Write one sentence per question. Revisit [L089](0089-cluster-gcn.html), [L108](0108-temporal-neighbor-sampling.html), and L112 after attempting the questions.

<details><summary>Check the retrieval</summary><p>Loss masking restricts label supervision; it does not remove feature paths. Batch-normalization running means and variances affect evaluation, so weights alone are insufficient. A cluster processed alone loses edges to excluded nodes; combining clusters can restore edges between those selected clusters.</p></details>

The primary reading is [Hu et al., OGB §4.1 and Table 4](https://arxiv.org/html/2005.00687v6#S4.SS1). Pair it with [Chiang et al., Cluster-GCN](https://arxiv.org/abs/1905.07953) for the sampling mechanism. Read OGB's footnote about the aggregation architecture before interpreting the result.

## 2 · Separate the sampling rule from the model

**A model** maps features and a graph to predictions. **A sampler** constructs the graph or computation neighborhood used for a training update. They are separate choices. The name “ClusterGCN” in OGB Table 4 refers to a cluster-based training method; its released model uses **GraphSAGE neighbor aggregation**. Replacing L112's normalized GCN with that model changes more than the batch size.

We therefore maintain two explicit experiments:

| Experiment | Fixed elements | Changed elements | What it answers |
|---|---|---|---|
| L112 bridge | Full arxiv data, GCN architecture, initialization, two passes over training labels | Full graph versus four random induced batches; degrees, BN populations, number of updates | What changes when we mini-batch this GCN? |
| Named OGB reproduction | Official products data and split, released SAGE architecture and cluster schedule | Modern recorded runtime and a newly computed fixed METIS partition | Can we replay the selected published experiment within budget? |

The bridge is deliberately short and uses random partitions. It demonstrates the conversion and its confounds. It is not an accuracy comparison between converged samplers. The named experiment is the place for a published score comparison.

[[PREDICT]]

**Prediction.** If both arms see every training label twice, do they take the same number of optimizer steps? Commit before opening the bridge results in section 7.

## 3 · Model architecture: follow one product

[[FIG:architecture]]

**Inputs.** Each product has 100 numeric features. The graph is an undirected co-purchasing graph. The prediction is one of 47 product categories. The official split ranks products by popularity: 196,615 training nodes, 39,323 validation nodes, and 2,213,091 test nodes. We preserve the released IDs; a random split would answer a different generalization question. See the [OGB dataset specification](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products).

**Visibility.** This is transductive training: the full feature graph is eligible for partitioning and messages, including validation and test nodes. Only training labels enter the objective. Validation labels select a checkpoint. Test labels evaluate the selected predictor. A label mask protects supervision; it does not make an experiment inductive or point-in-time valid. Connect this distinction to [L109's timestamp contract](0109-database-timestamp-contracts.html).

**Partition once.** METIS is a provided graph-partitioning library. It groups nodes while trying to reduce edges crossing group boundaries. We form 15,000 disjoint groups and reuse that fixed partition across runs. The original partition was not released. The saved partition and its hash are part of our experiment identity.

**Form a batch.** Shuffle the groups and choose 32. Let B be the union of their nodes. The induced subgraph contains every original edge whose two endpoints both belong to B. It includes edges between selected groups. It excludes edges from B to unselected nodes. The average batch has approximately 2,449,029 × 32 / 15,000 ≈ 5,225 nodes, but actual sizes and edge counts vary.

**Aggregate.** For receiver i, let N_B(i) be its incoming neighbors inside that batch. Define mᵢ as their mean hidden feature vector; use the zero vector if there are no neighbors. A released SAGE layer computes:

**zᵢ = W_neighbor mᵢ + b + W_root hᵢ.**

Here hᵢ is the node's current feature vector. The two W matrices are learned linear maps. The bias b belongs to the neighbor branch; the root branch has no additional bias. Raw self-edges, if present, remain in the neighbor aggregation. We add no new self-loops. Duplicate raw edges retain multiplicity in the mean, matching the released edge-index operator.

This differs from L112's **S H Wᵀ + b**, where S contains symmetric degree normalization and one self-loop per node. SAGE uses separate root and neighbor weights. The products model has no batch normalization and no L2 normalization of the hidden vector.

**Stack layers.** The widths are **100 → 256 → 256 → 47**, for **206,895 trainable parameters**. The first two layers apply ReLU, then dropout with probability 0.5. ReLU clips negative values to zero; dropout randomly zeros activations during training. The last layer produces class logits. Log-softmax converts them to log-probabilities for negative log-likelihood, or NLL. The loss averages only over training nodes present in the batch. Adam uses learning rate 0.001. These choices follow the [pinned publication-era release](https://github.com/snap-stanford/ogb/blob/cf066f93311ab3099cad84d71085d1b0375dcc2e/examples/nodeproppred/products/cluster_gcn.py).

**Read the operator.** In this PyTorch class, `lin_l` is the neighbor transform and `lin_r` is the root transform. Sparse matrix multiplication computes the incoming means before the learned transform. `root` permits inference chunks to use just their own receiver rows while reading a complete previous-layer feature matrix.

[[CODE:MeanSAGE]]

**Task 1 · incoming means.** Implement `mean_adjacency`. A sparse adjacency stores nonzero entries instead of an N-by-N dense matrix. Its rows are receivers, columns are senders. Dividing each incoming entry by its receiver's degree produces a neighbor mean. The CHECK includes a directed graph and an isolated node so an accidental transpose or added self-loop fails.

## 4 · Work the boundary by hand

[[FIG:boundary]]

**Worked example.** Consider the path 0—1—2—3, with scalar features `[2, 4, 8, 16]`. Cluster C0 contains nodes 0 and 1; C1 contains nodes 2 and 3. Set both SAGE weights to 1 and the bias to 0.

On the full graph, node 1 sees neighbors 0 and 2. Its neighbor mean is (2 + 8) / 2 = 5. Its root feature is 4. Its output is **5 + 4 = 9**.

In a C0-only batch, node 2 is absent. The neighbor mean becomes 2, so node 1 outputs **2 + 4 = 6**. Both the missing message and the denominator changed. When C0 and C1 enter the same induced batch, edge 1—2 returns and the output becomes 9 again.

**Predict before moving the control:** will adding a third cluster that is farther down the path change this one-layer output? Why might it change a deeper model?

[[TRACE]]

<details><summary>Check the prediction</summary><p>A third cluster does not change node 1's immediate neighbors, so this one-layer output stays 9. At greater depth, features from farther nodes can travel along additional hops. A complete immediate neighborhood is not necessarily a complete multi-layer computation neighborhood.</p></details>

**Task 2 · preserve identities.** Implement `induced_edges`. If a batch stores global nodes `[2, 1]`, its local row 0 is global node 2, and local row 1 is global node 1. Keeping global edge IDs while slicing the features connects the wrong rows or indexes out of bounds. The CHECK deliberately reverses node order and also tests a batch with no retained edges.

**Loss weighting.** A batch averages NLL over its own training nodes, then takes an optimizer step. Batches can contain different numbers of labeled nodes. This is not one globally averaged gradient accumulated at fixed weights; parameters change between steps. Recall [L100](0100-heterogeneous-gnn-checkpoint.html) when comparing these procedures.

**Why not just process each cluster separately?** Doing so would discard even the edges between clusters that were chosen together. Multi-cluster batching restores those edges. The [pinned PyG ClusterLoader implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/cluster.html) is provided infrastructure; the lab's small induced-subgraph oracle checks the semantic contract independently.

> **Scope check.** The mean of sampled neighbors can estimate a full neighbor mean under particular sampling assumptions. That does not make an entire nonlinear multi-layer loss or gradient unbiased. Cluster inclusion is also not uniform independent neighbor sampling. This lesson makes no unbiased-gradient claim.

## 5 · Budget the whole computation

A **hidden representation** is the feature vector produced by an intermediate layer. With N nodes, H channels, and float32 storage, one dense representation requires **N × H × 4 bytes**. A GiB is 2³⁰ bytes.

**Worked example.** For products at width 256, one hidden buffer is 2,449,029 × 256 × 4 = 2,507,805,696 bytes, or approximately **2.34 GiB**. Two saved hidden layers already approach 4.67 GiB before gradients, graph storage, optimizer state, input features, activations and temporary workspaces. This arithmetic is a lower-level sizing calculation, not a measured peak-memory prediction.

**Neighbor sampling.** Starting from B target nodes, three hops with fanouts f₁, f₂, f₃ can expand to at most B × (1 + f₁ + f₁f₂ + f₁f₂f₃) node occurrences before overlap is removed. A fanout is the maximum number of neighbors drawn per receiver at a hop. Real samplers can deduplicate shared neighbors. Low-degree nodes can contribute fewer neighbors.

[[FIG:memory]]

[[MEMORY]]

**Cluster sampling.** Choose an induced batch first; all layers operate within its nodes. This avoids the same hop-by-hop growth, but cuts boundary context. Neither approach is universally faster: graph structure, implementation, fanouts, partition quality and CPU-to-GPU transfer matter. [GraphSAGE](https://arxiv.org/abs/1706.02216) explains sampled neighborhoods; [Cluster-GCN](https://arxiv.org/abs/1905.07953) motivates dense subgraph batches.

A credible compute note separates **download/preprocessing**, **partitioning**, **training**, and **evaluation**. Record peak allocated GPU memory with its measurement boundary. PyTorch's allocated-memory counter excludes some non-PyTorch allocations and CPU storage. Do not describe it as the whole machine's memory use.

**Budget rule.** This lesson has a $10 aggregate allowance. Data preparation, pilots, all seeds, retries and verification share it. We use a timed pilot and a reserve before launching complete runs. The [budget ledger](../labs/_budget_l113.json) records rates and caps. A credit balance does not raise the allowance. A faster partial run does not count as a completed published schedule.

## 6 · Small batches for training; complete neighborhoods for inference

[[FIG:inference]]

Training on induced subgraphs does not require evaluating on those same cut graphs. At evaluation, we use the complete released graph. We can still limit memory by working **one layer at a time** and splitting the receivers into chunks.

For layer 1, keep the input feature matrix fixed. For each receiver chunk, fetch **all** of its incoming neighbors and compute that chunk's outputs. Write them to their global rows. Only after every node's layer-1 output exists may layer 2 begin. Repeat for all three layers. Disable dropout. Finally choose the largest logit per node.

**Why is this exact?** Within a fixed layer, a receiver's output depends on previous-layer features, not on another receiver's newly computed current-layer output. Chunking receivers therefore changes the execution order without cutting neighbor context. Running all three layers on each isolated chunk would violate that condition and cut dependencies.

The visible implementation keeps the previous feature matrix on the device and streams sparse adjacency rows into receiver chunks. It retains output features on the CPU between layers. This still requires one complete previous-layer feature matrix to fit on the GPU. It is sufficient for this products experiment; a larger graph may require gathering neighbor features from CPU memory or storage. Small training batches alone do not establish end-to-end scalability. This is a modern execution strategy; it is not a claim of historical runtime identity. The CHECK compares three chunk sizes, including one receiver at a time, against a dense full-graph calculation.

**Read the inference loop.** `rowptr` marks the start and end of each receiver’s entries in CSR storage. Each sparse block has rows for the current receivers and columns for all previous-layer nodes. The `out` matrix is completed before replacing `x`.

[[CODE:layerwise_inference]]

**Select consistently.** The released schedule trains for 50 epochs and evaluates at epochs **20, 25, 30, 35, 40, 45, 50**. Select the **first** maximum validation accuracy among those candidates. Store that model and its predictions. Report its test score. Evaluating every epoch would create additional selection opportunities and change the protocol.

**Task 3 · checkpoint selection.** Implement `selected_epoch`. The CHECK gives the best test score to a worse-validation candidate and includes a validation tie. Your function must ignore test scores and preserve the first tie.

## 7 · Evidence: what actually ran?

### The GCN conversion

[[BRIDGE]]

These arms share initialization, data, architecture, learning rate and two label passes. The mini-batch arm takes eight optimizer steps instead of two, uses smaller batch-normalization populations and changes normalized neighborhoods. Its different accuracy cannot be attributed to memory reduction alone. At two epochs neither arm is a converged baseline. The result teaches the implementation change and identifies variables a rigorous comparison would need to control or study separately.

### The named products experiment

[[RESULTS]]

[[FIG:results]]

The published target is **validation 92.12 ± 0.09% and test 78.97 ± 0.33%**, reported over ten runs. Our predeclared numerical rule is a mean within **0.5 percentage point** of the corresponding published mean. This is a descriptive tolerance, not an equivalence test. We report sample seed standard deviation without forcing it to match. A seed spread measures variability on this fixed dataset and split, not uncertainty across all possible deployment datasets.

> **Scope check.** A source-parity check establishes a computation comparison under recorded conditions. A pilot establishes execution and timing. Only completed full-schedule runs support a selected-experiment score report. Even ten numerically close runs do not establish the original random seeds, historical METIS partition, bitwise identity or reproduction of the entire OGB paper.

**Interpret the numerical miss.** The completed test mean is 0.604 percentage points below the published mean, exceeding our 0.5-point tolerance. We therefore record `OUTSIDE_TOLERANCE`. Independent replay through the released model agrees on all 24,490,290 selected-checkpoint class predictions. That rules out a discrepancy in those replayed classes; it does not establish equality of the training trajectories. A new METIS partition, unspecified original random seeds and a modern sparse/runtime stack remain potential differences, but this experiment does not isolate their causal effects. We do not retune against the observed test result.

The [protocol ledger](../labs/l113-reproduction.md) separates these claims, including the released logging bug, modern runtime, partition provenance, initialization and inference execution differences. The primary source remains authoritative for its own result; local evidence is reported under its actual scope.

## 8 · Lab, defense and spaced return

Open the [student notebook](../labs/0113-scaling-ogb.ipynb), [prepared notebook](../labs/html/0113-scaling-ogb.html), or [teacher solution](../labs/solutions/0113-scaling-ogb.ipynb). The notebook contains the complete visible SAGE model and trainer, independent CHECKs, the GCN bridge, and a gated full-data reproduction path. See the [quick reference](../reference/scaling-ogb.html) and [exact commands](../labs/l113-reproduction.md).

**EXIT.** Submit your three functions, your bridge output, and a short compute note. Defend these four points:

1. Trace node 1's output before and after cutting its boundary edge. Explain the denominator as well as the missing message.
2. Explain why converting L112 to mini-batches changes more than GPU memory. Name the changes to degrees, batch normalization and update count.
3. Explain why receiver-chunked, layerwise inference preserves complete-graph computation, while independent subgraph inference does not.
4. State exactly which evidence supports the products result. If the run is partial, explain why you cannot report a ten-seed reproduction verdict.

[[TEACHBACK]]

**Teach back.** In 120 words, explain why the word “GCN” in ClusterGCN does not identify the products aggregation operator. Then connect that distinction to a future relational model: which choices determine the row encoder, which determine message passing, and which determine the sampled computation graph?

**Return tomorrow.** Reconstruct the 2.34 GiB calculation without notes. In one week, draw the two-stage inference dependency and identify where a premature layer transition breaks it. Ask the teaching agent for feedback on any unclear step and for grading of your written defense. Author execution does not establish learner mastery: **PENDING_WRITTEN_DEFENSE**.
