<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Reading guide">
<p class="sequence-eyebrow">From contextual scores to contextual row tokens</p>
<p><strong>Reading route.</strong> Build five row descriptions → follow local and global attention → audit token ownership → interpret the reproduction boundary.</p>
<details><summary>Quick prerequisite reminder</summary><p>A token is one sampled row occurrence represented as a vector. A projection is a learned mapping to another vector space. Local attention compares tokens belonging to the same query. Global attention here reads a stored centroid memory; it does not fetch every database row at prediction time.</p></details>
</aside>
<!-- sequence-review:end -->

## Your tangible win

Trace one relational query through **RelGT**, then implement three contracts that the actual model, data audit and trainer use. You will be able to explain both what attention changes and what a successful reproduction must establish. This advances our mission: make relational-model claims defensible through visible computation and controlled evidence.



## 1 · What the previous methods leave unresolved

[RelGNN](0143-relgnn-reproduction.html) changes how messages travel along foreign-key paths. A foreign key identifies a row in another table. RelGNN's composite route moves information across a bridge without first collapsing every relationship into one shared state. [ContextGNN](0144-contextgnn.html) uses a shared query graph to rank nearby candidates contextually and distant candidates through a separate item representation.

Neither mechanism asks every pair of sampled rows to exchange information directly. That is the new question here. Suppose two results belong to different races but share a driver. A graph layer normally follows the database's links. A Transformer can compare the two sampled result tokens directly. It must still know which table each row came from, when it occurred, and how it relates to the query. Otherwise an unordered bag of row attributes loses relational meaning.

> **In plain terms.** RelGT widens communication inside a sampled context, while enriching each row's representation so the context retains relational clues. Attention cannot use rows that were never sampled, and a larger communication range is not a guarantee of better predictions.

The primary reading is [Dwivedi et al., Relational Graph Transformer, §§3–4](https://arxiv.org/html/2505.10960v1). The paper evaluates entity classification and regression; it excludes recommendation tasks. We return to F1 driver-position, so Lesson 144's full-catalog sponsor ranking objective does not carry over.

## 2 · Model architecture: a row becomes five vectors

[[FIG:architecture]]

A **query** is a driver ID and prediction cutoff. The target is the mean finishing position over the next 60 days. The released pipeline first builds a heterogeneous graph: rows are nodes, table names identify node types, and foreign keys define edges. It gathers neighbors up to two hops from each root and selects a fixed context. The release uses **300 slots including the root**, rather than 300 neighbors plus a root. Sampling can repeat rows when the neighborhood is small.

**Shapes.** Let `B` be the number of queries in a batch, `K=300` the slots per query, and `d=512` the hidden width. Each of the following components produces a `B × K × d` array. One slot is one sampled row occurrence owned by one query.

1. **Table type.** A learned embedding lookup maps a table ID to a vector. An embedding is a trainable row of numbers. It distinguishes, for example, a driver from a result.
2. **Hop distance.** Another embedding represents distance from the root: zero for the root, one or two for graph neighbors. The source reserves hop three for global fallback samples; it is not a measured third-hop path.
3. **Relative time.** The release computes age in days as `(cutoff − row_time) / 86400`. Sinusoidal positional features, followed by a learned linear transformation, encode that number. Undated rows receive zero. Negative ages receive a learned mask vector. **Masking an age does not remove the row's attributes.**
4. **Row attributes.** Each table has a typed feature encoder and a four-layer residual network. Numerical, categorical and other supported columns become a common vector. A residual connection adds an earlier hidden state to a later transformation, helping information and gradients travel through the network. The release replaces nonfinite input tensor values before this encoder.
5. **Local structure.** Each sampled occurrence gets a fresh random scalar. A projection and four residual GIN layers propagate these values along sampled edges. A GIN layer combines a node's own representation with a sum of neighbor representations, then transforms it with a neural network. The resulting vector carries connectivity information that type and hop alone cannot describe.

Each component is normalized separately. **Layer normalization** rescales coordinates within a vector. The release concatenates the components in the order **type, hop, time, features, structure**, producing `5d=2560` coordinates. Its mixing network maps `2560 → 1024 → 512`, with a ReLU between the two linear layers. ReLU replaces negative values with zero.

[[FIG:token]]

**Worked example.** Use scalar components `[1,2,3,4,5]` and weights `[1,0,0,2,−1]`. The projection is `1 + 8 − 5 = 4`. Setting only the structural component to zero changes the output to 9. Summing the five inputs first would lose which weight belongs to which component. This example isolates concatenation; it is smaller than the released nonlinear mixer.

[[TOKEN_WIDGET]]

**TODO 1 — `mix_five`.** Concatenate five equally shaped components and apply the supplied projection. Preserve gradients to every component. Your function is called inside the complete model, not only by an exercise checker.

> **Source distinction.** Paper §3 writes a linear relative-time encoder and a single token projection. The pinned release uses sinusoidal time features and a two-layer mixer. Its random structure features are resampled on every forward call, including evaluation. We expose these differences rather than implying that the equations and release are identical. [Encoder source](https://github.com/snap-stanford/relgt/blob/19e423ca3e7cac761130aba790857f2dc3a46ef7/encoders.py).

## 3 · Local attention changes who can communicate

**In plain terms.** Every slot asks which other slots in its own query context contain useful information. It does not have to traverse the FK path again at every Transformer layer.

Each local layer projects tokens into **queries, keys and values**, written `Q`, `K` and `V`. A query-key dot product measures compatibility. Dividing by the square root of the per-head width controls its scale. **Softmax** exponentiates these scores and divides by their sum, producing nonnegative weights that sum to one. The layer computes `softmax(QKᵀ / √128)V`. Four heads perform separate comparisons; each head has 128 coordinates. This use of `K` for keys is distinct from the context-size symbol.

The attention matrix has `300 × 300` entries per query and head. Different queries do not attend to one another. Residual additions, normalization and a feed-forward network complete each layer. More layers repeat that computation; the released sweep uses one, four or eight.

**Readout.** After the final normalization, the source keeps the root vector and forms a learned weighted average of the other slots. It scores each neighbor jointly with the root, normalizes those scores over neighbors, then adds the weighted neighbor vector to the root. This produces one local `B × 512` representation.

A repeated sampled row occupies multiple slots. That can change its total attention weight. Moreover, the released adjacency map keeps only the last slot for a duplicate row as an edge destination. Neither repetition nor arbitrary slot order should be described as a faithful copy of a unique-node induced graph.

## 4 · Global attention reads a training-time memory

[[FIG:attention]]

A **centroid** summarizes a cluster of root representations. RelGT stores 4,096 centroids and updates them using **exponential moving averages (EMA)**: each update retains most old state and adds a small contribution from the current training batch. The decay is 0.99. These are buffers updated by clustering, not ordinary embedding parameters trained directly by gradient descent.

The root attends to centroid keys and values. The released global branch scales scores by `√512`, and adds `log(centroid occupancy)`. Occupancy counts the assignments in its node-to-centroid buffer. A zero count gives a negative-infinite logit, removing that centroid's attention weight. This buffer is initialized randomly; it must not be presented as an exact census of historically observed database rows.

**Equal similarity does not mean equal global weight.** Use two centroid values, 2 and 10, with equal query–key scores and no dropout. The `log(count)` term makes softmax weights proportional to occupancy:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:.4em .25em"><thead><tr><th>Counts</th><th>Weights</th><th>Global value</th></tr></thead><tbody><tr><td>1,3</td><td>1/4,3/4</td><td>8</td></tr><tr><td>3,1</td><td>3/4,1/4</td><td>4</td></tr><tr><td>4,0</td><td>1,0</td><td>2</td></tr></tbody></table>

The first result is `(1×2 + 3×10)/4`. This is the **released global attention calculation** in a scalar diagnostic, before the later normalization and prediction head. Changing only the assignment-count buffer changes the result even though the query, centroids and learned weights are unchanged. That buffer must travel with the checkpoint. Its randomly initialized assignments still do not become a census of observed historical rows. [Released global forward](../labs/sources/l145/model.py)

**Try it after the example.** Double both positive counts in the first row. Then instead move all four assignments to the second centroid. What are the two outputs? <details><summary>Check your reasoning</summary>Doubling all counts preserves their proportions and gives 8. Counts [0,4] remove the first centroid and give 10. The zero-count case is a mask, not a finite additive penalty.</details>

The local and global vectors are separately normalized, concatenated to `B × 1024`, and passed through a feed-forward network and scalar prediction head. Training minimizes **mean absolute error (MAE)**: the mean of `|prediction − target|`. Evaluation clips predictions to the training targets' second and ninety-eighth percentiles, following the source.

**State boundary.** Centroid updates happen during training; evaluation freezes centroid and normalization state. Save those buffers with a checkpoint. However, the released scaled-dot-product attention receives its dropout probability unconditionally. **Dropout** randomly removes contributions; it remains active in this local attention operation during evaluation. Along with resampled structural features, this means repeated predictions can differ even after `model.eval()`.

## 5 · A correct encoder cannot repair the wrong query context

[[PREDICT]]

[[FIG:ownership]]

**Worked example.** Driver 7 has queries at times 5 and 10. An event occurs at time 9. The event is eligible for the later query but unavailable to the earlier one. If a cache stores only `driver 7 → context`, its later entry overwrites the earlier context. Both queries can then receive the time-9 event. The cached age is `10−9=1`, a positive number, even when the actual owner cutoff is 5.

The original `local_nodes_hetero` returns a dictionary keyed by node type and entity ID, omitting cutoff. Its HDF5 writer reuses that entry for every repeated entity in the chunk. This is a source behavior, not merely a hypothetical bug in our adaptation. A separate fallback branch samples globally when a root has no neighbors, without filtering the fallback rows by cutoff.

[[OWNER_WIDGET]]

**TODO 2 — `audit_tokens`.** Check raw timestamps against the cutoff of their actual query row. Reject duplicate `(entity, cutoff)` keys. Return violating `(query row, token slot)` pairs. Missing timestamps remain an explicit availability limitation. Positive cached ages are insufficient evidence.

A corrected cache would use full query identity and a fingerprint of the sampler and graph. A corrected fallback would filter candidates before sampling. Those changes are scientifically useful, but they create a different pipeline. This lesson preserves the original behavior for the audit and does not pass off a repaired score as the published experiment.

## 6 · Reproduce the experiment, then defend the boundary

The named target is **RelGT v1 Table 1, `rel-f1/driver-position`**, published test MAE **3.9170**. The paper's RDL GNN comparator is **4.022**. Both are published numbers, not fresh measurements from this lesson. They correspond to a descriptive relative reduction of about 2.61%; one task does not establish general superiority. [Table 1](https://arxiv.org/html/2505.10960v1#S4).

**Held fixed:** full task population, released graph preparation provenance, width 512, 300 slots, 4,096 centroids, batch 256, Adam learning rate `0.0001`, weight decay `0.00001`, and seed 0. **Varied:** three depths `{1,4,8}` crossed with three dropout values `{0.3,0.4,0.5}`: nine configurations. Each chosen dropout value is used for both feed-forward and attention dropout. Each configuration requests 100 epochs. The release parses warmup steps but does not apply a warmup schedule in this training loop. Gradient norms are clipped to one.

**TODO 3 — `last_validation_min`.** Select the last epoch attaining the smallest validation MAE. The source uses `<=`, so `[4,3,3,5]` selects epoch 3, not epoch 2. This differs from earlier lessons' first-minimum contract. Test never chooses a checkpoint or configuration. The full runner declares its additional cross-configuration selection rule because a historical aggregation script is unavailable.

[[RESULTS]]

The default notebook runs a small complete-network forward/backward fixture, source parity checks, original-cache failure probes, and independent audits of saved author evidence. Its small synthetic network is mechanism evidence. The complete released-scale trainer is visible after the exercises, with a separate guarded execution gate. **Running the default notebook is not running nine full experiments.**

The US$10 total cap includes preparation, pilots, validation, failed attempts and overhead. We reserve before launching workers and retain failed attempts. If either temporal validity or remaining budget fails, the complete reproduction stays `INCOMPLETE`; the runner and evidence still ship. See the [exact protocol and commands](../labs/l145-reproduction.md).

> **Scope check.** Source parity tests software agreement. A numerically close MAE tests only a score difference. Historical reproduction additionally needs matching data, preprocessing, cache, selection and environment. No result here establishes real-world attribute availability, whole-paper reproduction, deployment behavior or learner mastery.

## Lab and exit defense

Open the [student notebook](../labs/0145-relational-graph-transformer.ipynb), [executed solution](../labs/solutions/0145-relational-graph-transformer.ipynb), or [prepared notebook HTML](../labs/html/0145-relational-graph-transformer.html). Use the [one-page reference](../reference/relgt.html) after attempting retrieval.

1. Complete all three TODOs and run their CHECK cells. Explain why each contract lies on a real execution path.
2. Trace one `B × 300 × 512` token array through local readout, global attention and the head. Distinguish EMA buffers from gradient-trained parameters.
3. Explain the cache overwrite with actual timestamps. Propose a repair and state why its result needs a new experiment label.
4. Defend the reproduction verdict using the complete schedule, temporal audit and cost ledger. Identify which evidence was executed by the author and which work you personally completed.

[[TEACHBACK]]

**Spaced return.** Tomorrow, reconstruct the five components without notes. In one week, explain why `eval()` and a positive age do not each prove what their names might suggest. Ask the teaching agent to review your code, tensor trace and written defense; follow-up questions are part of this lesson.


<!-- sequence-next:start -->
**Carry this forward.** Use the access and selection audits in Lesson 146. A GNN–transformer comparison is meaningful only after specifying what each model saw and how its result was chosen. [Continue to Lesson 146](0146-gnn-vs-graph-transformer.html).
<!-- sequence-next:end -->
