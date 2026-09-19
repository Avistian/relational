## Start with retrieval

[[WARMUP]]

Before reading, write three answers from memory: What makes a message-passing layer invariant to the order of a node's neighbors? What does the `batch` vector identify in a disjoint union? Why can a validation score used to select a model be optimistic?

**Your tangible win.** Implement a GIN update and whole-graph readout, then explain a pair of graphs the model cannot separate. The core reading takes about 25 minutes. The notebook adds three live implementation tasks; the full reproduction is a separate, much longer run.

[Lesson 87](0087-link-prediction.html) ended by turning a candidate link into an enclosing subgraph. Now the prediction target is the entire graph. One molecule yields one class label, even though molecules have different numbers of atoms. Our mission needs this step: relational models must compress variable-size neighborhoods without silently discarding the distinctions a task needs.

## 1 · From node rows to one graph prediction

> **In plain terms.** A graph classifier must ignore how we number the nodes while preserving the information that distinguishes graphs.

A graph is a set of nodes and their edges. A node feature is a vector describing a node. **Graph classification** assigns one category to each graph; it is different from assigning a category to each node. In MUTAG, nodes represent atoms, the input provides categorical atom tags, and the binary target describes a molecular mutagenicity assay. The released GIN input does not use bond labels. Read the dataset description in [Xu et al., Appendix I](https://arxiv.org/html/1810.00826v3#A9).

A **readout** reduces all node representations of one graph to a fixed-length vector. If node representations have width *d*, sum, mean, and coordinate-wise maximum each return *d* numbers, regardless of graph size. A final classifier maps those numbers to class scores, called **logits**. Softmax converts logits into probabilities; cross-entropy trains the correct class to receive higher probability.

**Two different reductions.** Neighbor aggregation combines adjacent nodes to update one node. Graph readout combines all nodes to predict one graph. Changing graph readout does not change the neighbor aggregator. The experiment later changes readout only.

In a mini-batch, we concatenate node rows from several graphs and offset their edge indices. The `batch` vector records which graph owns each row. Edges never cross graph boundaries. `graph_readout(h, batch, B)` returns a `[B, d]` matrix for *B* graphs. A sum over every row without grouping would merge unrelated molecules.

## 2 · Counting is information

**Predict first.** Duplicate every node representation in a readout input. Which reduction must retain that change when the sum is nonzero?

[[PREDICT]]

**Worked example.** Encode atom type A as `[1,0]` and type B as `[0,1]`. The multiset `{A,A,B}` has sum `[2,1]`; `{A,A,A,A,B,B}` has sum `[4,2]`. A **multiset** retains repetition. A set would retain only which distinct types occur.

[[POOL_FIG]]

Mean divides by the number of rows. Both multisets become `[2/3,1/3]`, so this representation describes proportions and loses total multiplicity. Coordinate-wise max returns `[1,1]` for both, describing presence. These collisions cannot be repaired by a classifier that sees only the collided vector. But mean can be a useful inductive bias when size should not matter. Greater expressive capacity is not a guarantee of better generalization.

**Sum is not magically injective.** An **injective** function maps distinct inputs to distinct outputs. Raw scalar sums collide too: `{1,3}` and `{2,2}` both sum to 4. The paper's result concerns sums of suitable encodings of elements from a countable domain with bounded multiset size. It establishes that an injective encoding exists; it does not prove arbitrary features or learned finite-width networks never collide. See [Lemma 5 and Corollary 6](https://arxiv.org/html/1810.00826v3#S4.SS1).

## 3 · WL: refine colors, then compare counts

> **In plain terms.** Two nodes should keep the same structural description only while their own descriptions and their neighborhoods remain indistinguishable.

The **one-dimensional Weisfeiler–Lehman test**, abbreviated **1-WL**, starts with node labels, represented here by colors. One round builds a signature for each node: its current color and the sorted multiset of neighbor colors. Each distinct signature gets a distinct new color. All nodes update simultaneously from the previous round. When comparing graphs, they must share the signature-to-color dictionary; independent arbitrary color numbers cannot be compared.

Count each color within each graph after every round. Different color histograms certify that the graphs are not isomorphic. **Isomorphic** means equal after a relabeling of nodes that preserves edges and initial labels. Equal histograms are inconclusive: WL is not a complete graph-isomorphism test. See [§2 and Theorem 3](https://arxiv.org/html/1810.00826v3#S3).

**Worked example.** Give every node the same initial color. A four-node path has degrees `[1,2,2,1]`; a four-node star has degrees `[3,1,1,1]`. The first WL round encodes these different neighborhood counts, so the graph histograms separate immediately. With identical scalar features, neighbor means would instead give 1 wherever a neighbor exists.

[[WL_WIDGET]]

[[WL_FIG]]

**A permanent collision.** Compare a six-cycle with two disjoint triangles. Both have six nodes, every node has two neighbors, and all initial labels match. Every node therefore receives the same signature within each round, in both graphs, forever. A standard message-passing GNN with the same initial features cannot separate them either. Giving nodes informative extra features or using a stronger architecture changes the problem; extra depth alone does not.

## 4 · Model architecture: GIN turns multiset counts into learned features

The **Graph Isomorphism Network (GIN)** uses a sum and a multilayer perceptron, or **MLP**, to learn a neighborhood update. An MLP composes affine maps with nonlinear functions. The nonlinear transformation is needed because one linear map cannot represent every useful multiset distinction. The precise argument and architecture appear in [§4–5 of the primary paper](https://arxiv.org/html/1810.00826v3#S4).

For node *v* at depth *k*:

```text
h_v^(k) = MLP_k((1 + ε_k) h_v^(k−1) + Σ_{u in N(v)} h_u^(k−1))
```

Here `h` is a node vector, `N(v)` is the set of its neighbors, and `ε` weights the center relative to those neighbors. Every node at a given depth uses the same MLP; different depths have separate parameters. **GIN-ε** learns epsilon. **GIN-0**, our paper target, fixes it to zero. At zero, center and neighbors have equal weight. GIN-0 is a practical variant: do not transfer every sufficient injectivity condition for the center–multiset pair to it automatically.

**Worked example.** A center `[1,0]` receives neighbors `[1,0]` and `[0,1]`. Their sum is `[1,1]`; adding the center produces `[2,1]`. This `[2,1]`, not the mean `[2/3,1/3]`, enters the MLP. With epsilon .5 the input would be `[2.5,1]`.

[[ARCH]]

The released model contains **four message-passing updates plus the input representation**, described as five layers in the paper. Each update uses a two-layer MLP: linear → batch normalization → ReLU → linear, followed by another batch normalization and ReLU. **Batch normalization** standardizes hidden coordinates using training mini-batch statistics and maintains running statistics for evaluation. **ReLU** keeps positive values and replaces negative values with zero.

GIN keeps information from every depth. It sums node representations separately at depths 0 through 4. The paper describes concatenating these five graph vectors. The release applies a separate affine classification head to each and adds the resulting logits. Without dropout these are equivalent to a single affine map on the concatenation: partition that map by depth and sum its bias terms.

**Release detail.** During training, the code drops individual class logits independently at each depth before summing them. This is not dropout on node embeddings. The dropout rate is tuned over `{0,.5}`. At inference dropout is off and batch normalization uses stored statistics. The same trained weights serve new graphs; graph labels never enter the forward pass. [Pinned model source](https://github.com/weihua916/powerful-gnns/blob/9a2ce8ac3e99278307093a464a95caf0fb04b602/models/graphcnn.py).

```python
# The notebook exposes the complete model, not just this fragment.
h = neighbor_sum(h, edges) + (1 + eps) * h
h = relu(outer_batch_norm(mlp(h)))
pooled = graph_readout(h, batch, num_graphs, mode="sum")
# Repeat at each depth; include the input-depth prediction head.
```

**Capacity, not certainty.** Under injective aggregation and readout conditions, a sufficiently expressive GNN can match the distinctions 1-WL makes. Ordinary message-passing GNNs cannot exceed that bound under matching initial labels. Training may learn less discriminative functions. Neither the theorem nor a training accuracy proves better performance on new molecules.

## 5 · Reproduce a named published experiment

**Target:** Xu et al. (2019), Table 1, MUTAG, Sum–MLP (GIN-0): **89.4 ± 5.6%**. This is one specific cell, not all nine datasets or all model families. Primary reading: [paper §7 and Table 1](https://arxiv.org/html/1810.00826v3#S7). The [authors' README](https://github.com/weihua916/powerful-gnns/tree/9a2ce8ac3e99278307093a464a95caf0fb04b602#cross-validation-strategy-in-the-paper) clarifies the selection procedure.

| Contract | Executable reconstruction |
|---|---|
| Data | All 188 released MUTAG graphs; SHA-256 verified bytes; seven one-hot node tags |
| Model | Four GIN-0 updates, two-layer MLPs, inner/outer batch normalization; sum readouts at five depths |
| Grid | Hidden width `{16,32}` × batch size `{32,128}` × dropout `{0,.5}`: eight configurations |
| Optimization | Adam, initial learning rate .01; halve after each 50 epochs; no weight decay |
| Epoch | 50 updates; each uses a fresh permutation's first batch, not a full pass through the dataset |
| Budget | 350 epochs, the release default; 17,500 updates per fold/configuration |
| Splits | Ten stratified graph folds, split seed 0; training RNG reset to 0 for each fold, as in the release |
| Selection | Average ten validation curves, choose one common epoch; then choose the highest-scoring configuration |
| Reporting | Equal-fold mean and fold SD at that configuration/epoch; report both SD denominator conventions |

**Why not take each fold's best epoch?** That would choose ten different models using ten separate maxima and then average the maxima. The release instead selects one epoch after averaging the curves. With fold curves `[.9,.8]` and `[.5,.8]`, its choice is epoch 2 and mean .8. Averaging separate fold maxima gives .85, a different and more optimistic statistic.

**Why no test accuracy claim?** Although Table 1 uses that heading, the clarified procedure has training and validation folds, with no untouched test set. We reproduce that historical selection rule and call our measurements selected cross-validation accuracy. A prospective performance estimate would require an outer test split or nested cross-validation; that would be a separate experiment.

[[RESULTS]]

[[RESULT_FIG]]

> **Scope check.** Modern PyTorch, NumPy and scikit-learn replace the historical runtime. Exact historical fold indices and selected configurations are unavailable. We publish current fold IDs and all curves. The release supplies 350 epochs as a default, not a per-dataset search log. Graph edge ordering and floating-point reductions can change optimization. Only completed candidates are reported above; the full search remains incomplete. Historical numerical parity is **INCOMPARABLE**, even when a score lies near the paper target. Other datasets, GIN-ε training, and a historical-runtime replay are **NOT_RUN**.

Use the [reproduction guide](../labs/l088-reproduction.md) for exact commands, environment pins, source hashes, protocol differences and independent checks. Download [recorded paper-track results](../labs/_paper_l088_results.json) and [verification evidence](../labs/_verify_l088_results.json). Do not silently substitute a 30-epoch teaching run for this search.

## 6 · Lab: which graph readout wins here?

**Held fixed:** the full MUTAG dataset, ten fold assignments, seed, GIN neighbor sum, width 16, batch size 32, no dropout, and a small budget of 30 epochs × 10 updates. **Varied:** only graph readout — sum, mean or max — at every depth. **Measured:** cross-validation accuracy, selecting one common epoch for each readout. This isolates a particular readout intervention under a limited budget, not an intrinsic ranking of architectures.

[[TEACHING]]

A max-readout win under this small budget would not contradict the counting argument. The theorem describes which distinctions a representation can preserve; this experiment measures what a particular optimizer learns from a small labeled dataset. Fold differences and model selection both matter.

The published Sum/Mean/Max–MLP rows vary neighborhood aggregation; this lab varies graph readout. They are not the same comparison. A sum readout can exploit graph size, including at the input layer; a size-only baseline would be a useful next diagnostic. A selected CV winner here needs a fresh outer evaluation before a generalization claim.

Open the [student notebook](../labs/0088-graph-classification.ipynb). Its three TODOs are live in the training path:

1. **TODO / CHECK — neighbor sum.** Route each sender to the correct receiver. Check exact sums and gradients; add the center once in the GIN update.
2. **TODO / CHECK — graph readout.** Return one vector per graph for sum, mean and max. Check unequal graph sizes and ensure graphs remain separate.
3. **TODO / CHECK — common epoch.** Select the earliest maximum of the mean validation curve. The counterexample above must pass.

The complete loader, disjoint-union batcher, GIN model, trainer and paper grid are visible in both notebooks. The solution's short execution demonstrates that the learner-facing code runs; the command-line paper track executes the full search. Embedded figures work without repository-relative image paths.

**EXIT artifact.** Save `l088-exit.json` with your three readout results, selected epochs, fold values, and answers to: Why can raw sums collide? Why does a six-cycle collide with two triangles? What makes the selected CV score optimistic? Explain a result that would make you choose mean readout despite its weaker counting ability. Send the artifact and your explanation to the agent for feedback.

[[TEACHBACK]]

**Spaced return.** Tomorrow, draw the four-node path/star refinement without looking. In a week, implement grouped sum and recover the common-epoch counterexample from memory. [Lesson 89's planned topic](../reference/graph-classification.html#next) asks how to preserve useful graph computation when the whole graph cannot fit in memory.

Ask the agent about any unclear step; especially challenge the distinction between an expressiveness theorem and a measured predictive advantage. Prepared material does not establish your mastery.
