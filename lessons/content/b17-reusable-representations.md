# B17 · FlexTab and GTAlign: what can a representation reuse?

<p class="subtitle">Research bridge · elective · trace the adaptation boundary.</p>

[B16](b16-autograble-graph-selection.html) asked which graph to construct for a task. Once we have chosen useful inputs, another question appears: **can we encode them once and reuse the result for another task?** This matters to the mission because a relational foundation model must justify both its predictive benefit and the work required to adapt it.

Your tangible win: trace two target tasks through one encoder, identify exactly when its cache becomes stale, and explain which GTAlign weights change on a new domain. Recall [B07’s semantic features](b07-semantic-transfer.html), [B11’s structural encoders](b11-supervised-relational-baselines.html), and [B12’s adaptation boundary](b12-adaptation-mechanisms.html). FlexTab is a different paper from B04a’s **TabFlex**.

**Route:** 25–35 minutes for the lesson, then 35–50 minutes for the lab. Prerequisites are vectors, weighted averages and support/query separation, restated below. [Student notebook](../labs/b17-reusable-representations.ipynb) · [Executed solution](../labs/html/b17-reusable-representations.html) · [Printable reference](../reference/b17-reusable-representations.html)

## Recall before reading

<div id="b17-warmup"></div>
<noscript><p>Recall without notes: can fixed weights produce changing predictions? Why can validation choose a graph while test labels cannot? What information is lost when rows are averaged?</p></noscript>

## 1 · Reusable does not mean independent of everything

An **embedding** is a vector representing an input. An **encoder** computes that vector. A **decoder** combines it with task information to produce a prediction. **Support rows** are examples whose permitted labels help predict **query rows**, whose labels must remain hidden during prediction. A support/query episode is one such prediction problem.

> **In plain terms.** Keep the description of the rows separate from the question you ask about them.

Let X denote the feature table and S the support-row membership. An encoder with weights θ produces Z = Eθ(X, S). A decoder with weights φ computes query probabilities Dφ(Z, yS), where yS contains only support labels. Changing yS cannot change Z if it never enters E. It can still change the answer from D. This is an **inference information boundary**, not a claim that the encoder was trained without prediction targets.

**Worked example.** Four support rows have features (−1,0), (0,1), (1,0), (0,−1). Two query rows have (.5,.5) and (−.5,−.5). Task A labels the support [0,0,1,1]; task B labels it [0,1,0,1]. The feature input is identical for both tasks. The same encoder output is therefore reusable. If task B instead promotes one of those feature columns to the target, that column must be removed from X. You no longer have the same encoder input.

<div id="b17-predict"></div>
<noscript><p>Predict: complement all four support labels with X fixed. Embeddings stay fixed; decoder probabilities may change.</p></noscript>

## 2 · Model architecture: FlexTab separates the streams

FlexTab places feature processing in an encoder and task conditioning in decoders. Its row tokens collect information from cells; projected row outputs from successive encoder layers are combined. The target stream reads those outputs separately. [FlexTab §2.2 and Appendix A](https://arxiv.org/html/2606.30336v2#S2.SS2)

{{FLEX_ARCH}}

### Follow one row

**Tokenize.** A cell token is a vector encoding a cell’s value. With R rows, C columns and hidden width D, the cell tensor has shape R×C×D; a tensor is simply an array with named axes. Add one learned **[ROW]** token per row to collect its features: R×(C+1)×D. Our six-row, two-column, width 16 example starts as 6×3×16. The full paper also has a batch axis, omitted here for one table.

**Across columns.** Each token issues a query vector Q to compare with key vectors K. Similarity scores become nonnegative weights summing to one through **softmax**, then mix the associated value vectors V. The [ROW] token reads feature keys. Feature tokens cannot read [ROW] as a key in the course implementation.

**Across rows.** Each column position can read only support rows as keys/values. A query still retains its own features through its residual path: a **residual** adds the block input back to its update. But it cannot read a different query row. This makes query batching irrelevant to each other query under this particular mask.

**Across layers.** A layer applies these operations plus a feed-forward transformation. **Layer normalization** rescales a vector using its coordinate mean and variance; the feed-forward network transforms each vector independently. Different depths can encode different information. We collect each depth’s row vector rℓ, apply its own learned linear map Pℓ, sum, then apply a final map W:

`z = W( P₁(r₁) + P₂(r₂) + … + Pₗ(rₗ) )`

The code you read in the notebook implements each operation explicitly. The paper’s larger model uses richer tokenizers and multiple attention heads. The lab’s single-head numeric network isolates the information path; it is not the released model.

### Why one-key attention is a special case

**Worked example.** Give a target token exactly one row-value vector V=[2,−1]. Whatever its query/key similarity score, softmax over a single number equals [1]. The attention output is therefore [2,−1]. The learned value and output projections can transform that vector, and the residual preserves the target token. With two row values [2,−1] and [0,3], scores [0, log 3] instead give weights [.25,.75] and output [.5,2]. Now the query can select among different rows.

{{ATTENTION}}

For a single table, the decoder can inject a row embedding through this one-key case and then mix support information along its target stream. For relational prediction, several related row embeddings can become the keys. Their relative weights can matter. [FlexTab Appendix A.2](https://arxiv.org/html/2606.30336v2#A1.SS2)

**Training boundary.** FlexTab first trains its encoder with the classification/regression decoder, then freezes the encoder while training additional task decoders. Its relational decoder learns from partitioned single tables. At inference, related database rows supply its inputs. A feature encoder that excludes the current labels can still have learned from many earlier prediction tasks. [FlexTab §3](https://arxiv.org/html/2606.30336v2#S3)

> **Scope check.** No such training occurs in B17. We inspect random networks at seeds 0/1/2. Their tiny probability changes reveal dependencies, not predictive skill, transfer, anomaly-detection quality or source parity.

## 3 · Predict the intervention, then inspect the measured path

Everything in this control comes from saved float64 forward passes. **Held fixed:** model weights, numeric preprocessing, support size and architecture. **Varied:** exactly the selected labels, features or row order. **Measured:** maximum absolute embedding change and probability change, after aligning rows by ID. A maximum is the largest change across coordinates or query probabilities, not a mean accuracy.

{{BOUNDARY}}

Changing support labels can reuse the feature cache. Changing support features can alter all contextual row representations. Editing q1 alone may change q1; q0 stays unchanged. Reordering support rows together with their labels preserves outputs up to floating-point rounding. Our conservative cache key still changes because it includes order; a cache miss can be harmless extra work.

{{RESULTS}}

These are per-seed mechanism measurements, with no confidence interval: the seeds initialize random wiring, rather than sampling trained model performance. Independent NumPy replay checks all 36 forward records,3456 embedding coordinates and 72 keyed query probabilities. All four possible externally held query-label assignments are checked for each seed/task. Query-label exclusion is enforced structurally by the inference interfaces; those external labels are never arguments to either encoder or decoder.

### A cache is a claim about unchanged dependencies

A **cache key** names the inputs under which a saved result remains valid. For this encoder include feature bytes, ordered row IDs, support membership, preprocessing identity and encoder weights. Include implementation/mask version and precision too. Exclude support labels only because this specific encoder does not read them. A decoder cache would need those labels.

The lab’s `cache_identity` function builds this contract. It hashes complete feature and state bytes rather than only tensor shapes. A **hash** is a compact fingerprint; matching fingerprints are an integrity check, not evidence of good predictions. Changing preprocessing, encoder weights or the support population must invalidate the cached embeddings. Frozen weights alone are insufficient.

For timestamped database predictions, audit **all** information paths. Restricting BFS neighbors to earlier timestamps is insufficient if the table encoder or preprocessing still sees future records. **BFS**, breadth-first search, visits graph neighbors by increasing hop distance. Every cutoff needs a legal feature-context population as well as legal direct neighbors.

## 4 · Model architecture: GTAlign adapts a different boundary

GTAlign converts structural graph representations into inputs for a tabular foundation model. **PCA** projects differing feature widths into a common space; a **GCN** repeatedly mixes neighboring node features. A **community** is a graph partition found from connectivity. Community IDs act as **pseudo-labels**—training targets inferred from structure rather than supplied class annotations. [GTAlign §2](https://arxiv.org/html/2607.11374v1#S2)

{{GT_ARCH}}

First, contrastive pretraining encourages related graph views to have similar representations. Next, community-based episodes jointly adapt the graph encoder and tabular predictor. Finally, labeled target-domain examples adapt the graph encoder while the tabular predictor stays frozen. The adapted encoder must recompute the embeddings before prediction. [GTAlign §§2.2–2.4](https://arxiv.org/html/2607.11374v1#S2.SS4)

**A prototype** is the mean embedding of a class’s labeled support examples. The target-stage objective encourages embeddings to agree with their class prototypes. This target-label use is adaptation, even though the final predictor uses in-context inference. In-context inference does not imply the entire preceding pipeline was zero-shot.

| Boundary | FlexTab reuse question | GTAlign adaptation question |
|---|---|---|
| Representation input | Table features and encoder context | Node features and graph adjacency |
| Reusable object | Target-free contextual row embeddings | Graph embeddings under a particular encoder |
| Target labels | Enter decoder at inference | Also update graph encoder during adaptation |
| Frozen part | Encoder during specialized decoder training | Tabular predictor during target adaptation |
| Required cache action | Rebuild if features/context/encoder change | Rebuild after encoder adaptation |

**Predict before checking.** If a GTAlign implementation freezes the tabular model but updates the graph encoder, can it reuse embeddings computed before adaptation?

<details><summary>Check your explanation</summary><p>No. The encoder is part of the cached function. Updating its weights changes that function, even when raw node features stay fixed. Recompute support and query embeddings using the adapted encoder.</p></details>

A fair future community-supervision ablation would hold real support labels, source graphs, episode budget and target adaptation fixed while changing only community supervision. B17 does not execute that study. A citation graph and a foreign-key database graph also encode different relationships; success on one cannot establish the other.

## 5 · Full reproduction: a named target and a real stop

The selected target is **B17-FLEXTAB-MULTI-TABLE5-F1-DNF**: the complete `rel-f1/driver-dnf` test task, published AUROC **74.6%**. AUROC measures how often a randomly chosen positive receives a higher score than a negative, with half credit for ties. This is a cited target, not a B17 measurement. [FlexTab Table 5](https://arxiv.org/html/2606.30336v2#S4.T5)

The archived author-linked repository probe returned404; the FlexTab model search returned no matches. We have not authenticated original weights, preprocessing, sampling/seeds, the complete evaluation population or its temporal encoding policy. The gate checks saved source hashes, then refuses a paper run. It cannot create a missing evaluator. GTAlign’s promised release is likewise not authenticated by the name-search results. [Source receipts](../labs/sources/b17/manifest.json) · [Full reproduction contract](../labs/b17-reproduction.md)

| Evidence lane | B17 status |
|---|---|
| Reduced mechanism: three seeds × two tasks × six interventions | COMPLETE_MECHANISM_DIAGNOSTIC |
| Full NumPy forward replay | PASS; independent arithmetic, same saved weights |
| Released-source or historical checkpoint identity | NOT_ESTABLISHED |
| Selected Table 5 F1-DNF inference | INCOMPLETE_SOURCE_PROTOCOL_GATE; NOT_RUN |
| Full pretraining / other paper experiments / GTAlign experiments | NOT_RUN |
| Live Colab / deployment | NOT_CHECKED / NOT_REQUESTED |
| Learner mastery | PENDING_WRITTEN_DEFENSE |

The original context and sampling settings remain in the contract; no smaller model, shorter context or synthetic result replaces the paper run. Approved paid execution is $0. The local numerical budget is 3600 aggregate seconds including retries and checks, under the standing $10 ceiling.

## Lab · make the boundaries executable

Open the [portable student notebook](../labs/b17-reusable-representations.ipynb). It contains the full reduced forward path, immediate checks and three live tasks:

1. Compute scaled attention and handle the one-key case correctly.
2. Aggregate row outputs with a different projection for each depth.
3. Build a cache identity that rejects changes to encoder dependencies.

The notebook executes your functions inside the complete grid. Predict the label-edit result before running it. Then compare the measured cache hits and deltas with your explanation. The [executed solution](../labs/html/b17-reusable-representations.html) is author evidence, not your completed work.

<div id="b17-teachback"></div>
<noscript><p>Explain in 150–200 words why target-agnostic embeddings may still be context-dependent. Identify two cache invalidators and the trainable GTAlign component on a new domain. Bring your answer to the teacher.</p></noscript>

**Exit ticket.** Include one measured intervention, the single-key attention derivation, and why B17 has not reproduced the paper score. Ask the teacher about any unclear operation; paste your code and explanation for feedback. Revisit from memory after 1, 7 and 30 days. Next, B18 asks whether the available context contains enough evidence at all.

**Primary reading:** [FlexTab §2.2 and Appendix A](https://arxiv.org/html/2606.30336v2#S2.SS2), then [GTAlign §2.4](https://arxiv.org/html/2607.11374v1#S2.SS4). Treat their benchmark claims as paper evidence until the corresponding experiments are independently reproduced.
