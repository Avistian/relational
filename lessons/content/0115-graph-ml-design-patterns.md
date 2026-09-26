<p class="stream-kicker">Synthesis · one reusable design skill · three prediction units</p>

**Your win:** given a graph prediction problem, write a defensible encoder → processor → head design, name the prediction unit and information boundary, and show whether a refactor preserves the original computation.

The core reading takes about 25 minutes. Then spend 25–35 minutes on the three design rows and live notebook tasks. The ten-run author experiment is separate from that study session.

[Student notebook](../labs/0115-graph-ml-design-patterns.ipynb) · [Executed solution](../labs/html/0115-graph-ml-design-patterns.html) · [Quick reference](../reference/graph-ml-design-patterns.html) · [Reproduction ledger](../labs/l115-reproduction.md)

[[WARMUP]]

## 1 · Start with the prediction unit

[Lesson 114](0114-ogb-error-analysis.html) asked where a trained GCN fails. Before choosing a replacement, we need a way to describe exactly what would change. An architecture name is insufficient: replacing a decoder, dropping an edge type, changing pooling, or exposing future events can each change the problem.

Write this sentence first: **“For each ___, predict ___ using information available at ___.”** A paper-topic classifier predicts a class per node. A citation model scores a directed candidate pair. A molecule classifier predicts once per graph. The three cases may share message passing, but their output rows, labels and losses mean different things. [OGB §3 and task definitions](https://arxiv.org/html/2005.00687v6#S3).

The organizing pattern is:

1. **Encoder:** turn available raw attributes into vectors with explicit type and missing-value semantics.
2. **Processor:** move and transform information along permitted graph connections.
3. **Head/readout:** select nodes, combine candidate endpoints, or pool within each graph, then produce scores with the required shape.
4. **Training contract:** define the supervised examples, loss, split, selection metric and saved state.

These are responsibilities, not a rule that every model must have exactly three learned layers. An encoder can be the identity. A head may itself aggregate neighbors. A decoder can be an MLP or another graph network. This lesson's decomposition is an engineering choice informed by the compositional view in [Battaglia et al., §4.3 and Figure 6b](https://arxiv.org/html/1806.01261v3#S4.SS3). It does not claim to reproduce that entire paper.

**Mission connection.** In [Lesson 076](0076-encoder-predictor-stack.html), a database row became an embedding and joined a graph. Here we make the contract reusable: table encoders → typed relational propagation → target-row head. That becomes the explicit RDL stack in planned Lesson 131.

## 2 · Trace the complete GCN before naming its modules

We use the same named experiment as Lesson 112: the released three-layer GCN on ogbn-arxiv. There are N=169,343 papers, 128 input coordinates and 40 classes. The official split has 90,941 training, 29,799 validation and 48,603 test papers. The publication supplies the benchmark; the pinned release supplies the executable model and training defaults. [OGB §4.3/Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3), [released GCN and training loop](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py).

[[FIG:architecture]]

**Step A — make the graph operator explicit.** Deduplicate the citation graph after making it undirected, add one self-loop per node, then form S=D⁻¹ᐟ²(A+I)D⁻¹ᐟ². A matrix row identifies the receiving node; its nonzero columns identify senders. Degrees are measured after deduplication and loop addition. The operator stays fixed during this experiment.

**Step B — preserve the encoder.** Here E(X)=X. The input is already a numeric feature matrix. Adding a learned 128→256 projection merely to fill an “encoder” box would add parameters and change the experiment.

**Step C — process two hidden layers.** For each layer compute U=S(HWᵀ)+b, apply batch normalization, ReLU, then dropout with probability 0.5. Shapes are N×128 → N×256 → N×256. Batch normalization sees all nodes in the transductive training forward pass. Only the loss is restricted to training labels. Dropout is disabled at evaluation; saved BN statistics are restored with the selected weights.

**Step D — keep propagation inside the head.** The final layer computes N×40 class scores as S(HW₃ᵀ)+b₃, then log-softmax. An ordinary Linear(256,40) head would omit the third graph operation. Both return N×40, so shape checking alone cannot detect the mistake.

**Step E — train and select.** Mean negative log-likelihood uses training rows only. Adam uses learning rate 0.01 for 500 epochs. Each epoch is evaluated; the first maximum validation accuracy selects the state, including BN buffers. Test accuracy is reported from that selected state. Ten seeds measure variation under this fixed dataset and schedule. No test-driven tuning is performed.

```python
class GCN(nn.Module):
    # Full definitions are visible in the notebook and canonical source.
    def forward(self, x, adj):
        z = self.encoder(x)         # identity: [N, 128]
        h = self.processor(z, adj)  # two hidden GCN blocks: [N, 256]
        return self.head(h, adj)    # final GCN + log-softmax: [N, 40]
```

**Predict:** if the last layer has no activation before log-softmax, can we remove its graph multiplication?

<details><summary>Check the dependency, not just the activation</summary><p>No. Linearity does not make the graph operator the identity. The final class score still depends on neighboring hidden vectors. Removing S reduces propagation depth and changes the scores. Even bias placement matters: S(HWᵀ)+b is generally different from S(HWᵀ+b), because symmetric normalization need not have row sums of one.</p></details>

**Task 1:** implement `graph_forward`. Apply the encoder once, then processor once, then head once. Supply the same adjacency to both processor and head. The checker uses noncommuting operations; the reproduction additionally compares outputs, input and parameter gradients, BN buffers and an Adam update.

## 3 · Change the head when the question changes

Suppose the processor has produced node vectors Z. For the following arithmetic, take z₀=[1,2], z₁=[3,5], z₂=[7,11]. These are fixed synthetic vectors, not learned explanations of the arxiv model.

[[FIG:readouts]]

**Node prediction.** Select the target node's vector and map its d coordinates to C class scores. For ordinary nodewise Linear heads, output shape is N×C; a seed-node mini-batch loss uses only seed outputs. Our reproduction uses a propagating node head, so its contract additionally requires adjacency. Selection does not mean deleting context nodes before message passing.

**Undirected link prediction.** Construct a vector from each candidate pair. Elementwise product gives z₀⊙z₁=[3,10], unchanged when endpoints swap. A learned linear score on that product is therefore symmetric. This is a valid deliberate constraint for an undirected relation; it is not a universal link decoder.

**Directed link prediction.** Ordered concatenation gives [z₀,z₁]=[1,2,3,5]; reversing endpoints yields [3,5,1,2]. A learned decoder can distinguish those orders. Concatenation permits asymmetry but does not guarantee it: symmetric learned weights could still yield equal scores. Heterogeneous links additionally need endpoint-type and relation semantics. A same-width tensor does not establish compatible meaning.

**Graph prediction.** For batch IDs [0,1,0], graph 0 contains nodes 0 and 2, so its mean is [4,6.5]; graph 1 contains node 1, so its mean is [3,5]. Pooling all three rows produces one vector [11/3,6] and loses the distinction between examples. Apply the graph head after per-graph pooling to produce B×C scores, where B is the number of graphs. [Gilmer et al., §2: message passing and readout](https://proceedings.mlr.press/v70/gilmer17a.html).

Mean and sum answer different questions. Duplicating every vector in a graph leaves its mean unchanged but doubles its sum. If graph size is predictive, mean alone can discard useful information; sum or an explicit size feature is an alternative to test. This is a modeling decision, not a universally superior pooling rule.

[[EXPLORER]]

**Task 2:** implement `pair_features(z,pairs,directed)`. Use elementwise product for undirected candidates and ordered concatenation for directed candidates. Explain the invariance being imposed, not just the resulting width.

**Task 3:** implement `graph_mean(z,batch,num_graphs)`. Accumulate per-graph sums and counts; divide within each graph. Our explicit empty-graph convention is a zero vector. A real benchmark may exclude empty graphs or require a different convention; record it instead of letting division by zero choose for you.

<details><summary>What must a link training protocol add?</summary><p>Candidate pairs and labels, an allowed message graph, a negative-sampling rule, an objective, and a ranking/evaluation population. Avoid letting held-out positive target edges enter the training message graph. A binary loss on arbitrary negatives is not automatically equivalent to a benchmark's ranking protocol. The small link exercise below supplies its pairs explicitly and makes no link-benchmark claim.</p></details>

## 4 · Protect the information boundary across modules

[[FIG:boundaries]]

A module interface needs more than a shape. Write down **row identity, type, graph membership, edge direction, and time visibility** wherever they matter.

| Situation | Encoder responsibility | Processor responsibility | Head and supervision responsibility |
|---|---|---|---|
| Static homogeneous nodes | Use the specified feature representation | Respect declared graph visibility and normalization | Preserve target IDs; mask loss to authorized labels |
| Heterogeneous entities | Encode each type with its feature schema | Route each typed relation in its defined direction | Select the target type; decode endpoints with their relation semantics |
| Independent graphs | Encode attributes consistently across examples | Keep adjacency block-diagonal across graph IDs | Pool within each graph; one label per graph |
| Temporal events or rows | Use attribute versions available at the cutoff | Apply event and availability cutoffs at every hop | Predict before updating with the target event; train only on mature labels |

The temporal row is a point-in-time contract from [Lesson 109](0109-database-timestamp-contracts.html), not the arxiv reproduction protocol. A dataset split by publication year does not by itself turn a transductive full-graph GCN into a historical snapshot system. For this OGB baseline, later nodes' features and graph structure are visible according to the released transductive setup, while held-out labels never enter the loss.

The reusable [Lesson 076 cutoff/reduction widget](0076-encoder-predictor-stack.html) makes the availability boundary concrete. Moving the cutoff changes which rows reach the processor; changing mean to sum changes the meaning of their aggregation.

[[CUTOFF]]

**A debugging order:** check IDs and visibility → check shapes and invariances → check forward values and gradients → check optimization and selection → compare the final metric. A close metric alone can hide offsetting errors. Lesson 116 will turn this order into a symptom-driven debugging practice.

## 5 · Refactoring is an empirical claim

Our claim is narrow: the modular GCN retains the released model's computation, under matched inputs, weights, buffers and random masks. Tests cover training and evaluation, including dropout. The flat-versus-modular comparison checks an Adam update in training mode. The independent PyG source comparison checks forward and gradients in both modes; its optimizer comparison uses evaluation-mode BN to avoid amplifying tiny near-zero pre-BN bias-gradient differences. That check does not establish identical historical training trajectories.

Fresh full-data training then checks the selected named experiment. The two checks answer different questions: operator parity asks whether the implementations agree under controlled conditions; benchmark reproduction asks what the complete declared protocol produces.

[[RESULTS]]

Read [the protocol and deviation ledger](../labs/l115-reproduction.md) before calling this “full reproduction.” The completed scope is ten fresh fits for **one published GCN experiment**, with independent scoring and original-code checkpoint replay. Historical randomness, historical software identity and the rest of the OGB paper are not established. The link/graph runs in the notebook are synthetic course examples, not extra paper benchmarks.

The student notebook includes all load-bearing code: graph construction, modular model, loss, selection, trainer, pair composition and graph pooling. Its default run executes the three live tasks on small examples; a separate full-reproduction switch uses the full dataset and ten-seed schedule. The author results do not imply that your notebook trained those models.

## 6 · Your design table and written defense

Fill all columns **before** opening the solution. The three task specs are deliberately different:

| Task spec | Encoder | Processor + visibility | Head + output shape | Loss and selection | One invariant / failure probe |
|---|---|---|---|---|---|
| A: classify each paper into 40 topics under the released arxiv protocol | Your choice | Your choice | Your choice | Your choice | Your choice |
| B: score whether customer u buys product v in the next 7 days using records available at cutoff t | Your choice | Your choice | Your choice | Your choice | Your choice |
| C: classify each independent molecular graph, using atom and bond attributes | Your choice | Your choice | Your choice | Your choice | Your choice |

<details><summary>Design feedback — open after writing</summary><p>A: identity numeric encoder; complete released undirected normalized graph; two hidden GCN blocks and a third propagating class head; N×40 log probabilities; train-label NLL and first best validation accuracy; compare outputs/gradients and preserve BN population. B: customer/product-specific encoders; directed typed historical interactions with event and availability times no later than t at every hop; ordered pair or relation-aware decoder, one score per candidate pair; binary loss with an explicit negative population or an explicitly defined ranking objective; validate on future windows with mature labels and matching candidate populations. Probe reversed endpoints and a late-arriving event. C: atom encoder with bond-aware messages; disconnected graphs retain batch IDs; invariant per-graph readout then graph classifier; one label per graph, split/metric chosen for the actual benchmark. Probe node permutation and cross-graph contamination. B and C are defensible starting designs, not uniquely correct architectures or executed benchmark protocols.</p></details>

**EXIT:** submit the three functions, your table, and a short defense explaining (1) why the GCN head needs adjacency, (2) what information each task may use, (3) why product and concatenation impose different link symmetries, (4) why pooling needs graph IDs, and (5) what the reproduced experiment does and does not establish. Passing automated checks alone leaves **PENDING_WRITTEN_DEFENSE**.

[[TEACHBACK]]

**Spacing:** tomorrow, redraw the complete GCN without looking and mark all three graph multiplications. In a week, design task B again with a late-arriving purchase and explain exactly where it must be excluded. Interleave a node task and a graph task to practice identifying the unit of supervision.

**Primary reading:** Battaglia et al., [Relational inductive biases, deep learning, and graph networks, §4.3](https://arxiv.org/html/1806.01261v3#S4.SS3). Pair the conceptual composition with the [pinned OGB implementation](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py). Ask your tutor follow-up questions wherever a tensor boundary, source claim or experimental conclusion is unclear.
