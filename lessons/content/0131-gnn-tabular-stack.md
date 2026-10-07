<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Where this lesson fits">
<p class="sequence-eyebrow">From Lesson 130 to this lesson</p>
<p>A working pipeline still leaves its internal computation hidden. Follow one real batch through row encoding, time addition, messages, root readout and gradients.</p>
<details><summary>Quick prerequisite reminder</summary><p>A tensor shape counts rows and coordinates. Autograd records differentiable operations. Detaching a tensor preserves its current values but cuts its gradient path.</p></details>
</aside>
<!-- sequence-review:end -->

## The question this lesson answers

**Can you follow one prediction through the actual network, and show how its loss changes the row encoders?** Lesson 130 established ownership of the experiment. This lesson opens the computation inside that experiment. Your deliverable is an annotated forward/backward trace with real tensor shapes, a checked gradient path and a defensible reproduction verdict.

[Student notebook](../labs/0131-gnn-tabular-stack.ipynb) · [Executed solution](../labs/html/0131-gnn-tabular-stack.html) · [Reference card](../reference/gnn-tabular-stack.html) · [Reproduction contract](../labs/l131-reproduction.md)

Read [Fey et al., §5, relational deep learning blueprint](https://proceedings.mlr.press/v235/fey24a.html) for the decomposition. Use [Robinson et al., RelBench v1 §3, Appendix B and Table 7](https://arxiv.org/html/2407.20060v1) for the implemented baseline and numerical target. These are different sources with different jobs: the blueprint motivates the stack; the benchmark specifies the selected experiment.

**Route.** First predict shapes on paper. Then trace the measured batch and complete the three functions. Run the short standalone notebook to inspect the author evidence and a small neural parity fixture. Finally use the explicit full-training gate for five fresh fits. The default replay, small fixture and full reproduction have separate evidence labels.

## 1 · Model architecture: a batch is a collection of questions

A query is `(driverId, cutoff)`. Its label is mean finishing position in the next 60 days. The full experiment contains 7,453 training, 499 validation and 760 test queries. A context node is a permitted database row that helps answer a query. It is not another supervised example.

The loader builds disjoint contexts for 512 queries at a time. A database row reused by two queries becomes two **node occurrences**, each with its own query owner and time cutoff. `n_id` maps an occurrence back to its database row; `batch` maps it to a query; `input_id` maps a supervised query back to the task table. These indices answer different questions.

[[FIG:architecture]]

**Diagram trace.** Trace the forward values and backward gradient path separately. Where would detaching preserve the current prediction but stop encoder learning? On a narrow screen, scroll the figure sideways.

The graph has nine table types. Each table owns a separate four-block Frame ResNet. Its output width is 128 regardless of how many raw columns that table has. Parameters are shared across occurrences of rows from the same table, but not across different table encoders. A row encoder can therefore process drivers and results with different input schemas while delivering compatible vectors to the graph network.

Time encodings are added to dated rows. Two heterogeneous GraphSAGE layers exchange information along foreign-key and reverse edges. Each relation uses neighbor **sum** aggregation in this experiment; relation outputs are also summed for each destination type. Node-wise LayerNorm and ReLU follow each layer. The scalar head reads only the first B driver vectors.

**Predict.** If the sampled batch contains 512 query roots and thousands of context occurrences, how many predictions and loss terms should it produce? Write the answer before inspecting the trace.

[[TRACE_TABLE]]

These are observed shapes from the first real training minibatch before any optimizer update. They are not the 99-node exhaustive single-query neighborhood used in Lesson 130. Sampling, query multiplicity and node type all affect the counts.

## 2 · From typed columns to row vectors

A TensorFrame groups columns by semantic type. Numerical values receive learned linear encodings; categorical values use embedding lookups; timestamps have their own encoding; precomputed text embeddings receive a learned projection. The ResNet combines those column representations into a row vector. Primary and foreign keys define identity and connectivity and are removed from feature columns by the released graph builder.

[Lesson 125](0125-pytorch-frame-deep-dive.html) opened the row encoder. Here its output becomes the GNN's input. The question changes from “how is this row represented?” to “how can a root's loss train a representation of a neighboring row?”

**TODO 1 — `activation_summary`.** Report shape, mean, maximum absolute value, zero fraction and a small coordinate sample. Detach only the copy used for logging. Replacing the live activation with its detached copy changes training. Reject nonfinite forward activations explicitly; an empty tensor has no numerical mean.

The canonical code is visible in the notebook, including the typed encoders and ResNets. Read-only primitive source is included alongside its license and pinned provenance. No opaque model import is the sole explanation of the computation.

## 3 · A node occurrence carries its owner's clock

For node occurrence i owned by query q, its age in days is `(cutoff[q] − node_time[i]) / 86400`. The index q is `batch_index[i]`, not the table row ID. Looking up the first query's cutoff for all rows silently misrepresents a disjoint batch.

[[FIG:time]]

**TODO 2 — `relative_days`.** Validate the owner indices and tensor lengths, gather the correct root cutoff, reject a future row and convert seconds to days. The real trace replay calls your function on recorded timestamps. Static tables without timestamps receive no invented age.

**Predict before changing the control.** In the synthetic example, query cutoffs are days 10 and 20; node times are days 9 and 18; owners are 0 and 1. Correct ages are 1 and 2. What happens if both occurrences use query 0's clock?

[[TIME_WIDGET]]

The model applies a positional encoding to each age, followed by a table-specific linear layer, and adds the resulting 128-vector to the row representation. This addition does not enforce eligibility; the sampler and temporal audit do. A learned time vector cannot repair a future row that was incorrectly admitted.

[[FIG:activations]]

Read the first row as an arithmetic check: the same occurrence's encoded features plus its time vector must equal the vector passed into message passing. Later channels mix through learned matrices, so channel 0 is not a conserved semantic feature.

## 4 · Message passing changes representations; the head selects roots

For relation r from source type s to destination type t, the released sum-GraphSAGE operator computes a learned transform of the summed neighbor features plus a learned transform of the destination's own features. HeteroConv sums these relation outputs for destination t. Because each relation has its own root transform, the root contribution is also repeated across incoming relation types. This is the pinned implementation, not an assumed generic GNN equation.

After relation summation, node-wise LayerNorm normalizes each row across its channels and ReLU sets negative coordinates to zero. The next layer repeats this process. Two rounds make two-hop input information available to a root. The released wrapper accepts sampled-size metadata but its forward implementation processes the supplied node and edge dictionaries without trimming them layer by layer.

[[FORWARD_CODE]]

**TODO 3 — `seed_readout`.** Return `hidden[:B]`, reject invalid B, and preserve autograd. This works because this loader places the query roots first in their entity type. It is a loader contract, not a rule that any arbitrary graph tensor satisfies. Context driver rows after the roots must not receive the task's root labels.

In the recorded first batch, all 512 driver occurrences are roots, so an incorrect “all driver rows” readout could pass that example accidentally. The CHECK deliberately supplies five driver rows but only two roots; it verifies both output count and which rows receive gradients.

The head maps `[B,128]` to `[B,1]`. Training uses mean L1 over B query labels. Evaluation additionally clips predictions to training-label percentiles. Do not put evaluation clipping into the training-loss trace: it would describe a different gradient.

## 5 · The backward pass explains what is learned

Backpropagation applies the chain rule from the root loss through the head, graph layers, time addition and row encoders. A context row can receive a gradient even though it has no direct task label: its representation contributed to a supervised root. Shared table parameters accumulate contributions from all their occurrences.

The trace uses three isolated copies of the same model on the same real batch and random state: the original forward, the explicitly instrumented forward, and a version that detaches encoded rows. It compares outputs, loss, gradient presence, nonfinite masks and finite gradient values. The original fit's parameters, buffers and random state are preserved.

[[FIG:gradients]]

**Predict.** Detach the row vectors immediately after encoding. Will the current prediction change? Will the row encoder receive gradients? Will the GNN still receive gradients?

**A numerical backward trace.** Reduce the path to one scalar row encoder and one scalar head: input `x = 3`, encoder weight `w = 2`, row vector `h = wx = 6`, head weight `v = 4`, prediction `vh = 24`, target 20. The L1 loss is 4. Its derivative with respect to the prediction is +1, so the chain rule gives `dL/dv = h = 6` and `dL/dw = v × x = 12`.

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:.4em .25em"><thead><tr><th>Path into head</th><th>Output</th><th>Encoder<br>gradient</th><th>Head<br>gradient</th></tr></thead><tbody><tr><td>Live h</td><td>24</td><td>12</td><td>6</td></tr><tr><td>Detached h</td><td>24</td><td>None</td><td>6</td></tr></tbody></table>

Detaching preserves the current number 6 and removes its recorded connection to w. It does not detach the head's own multiplication by v. `None` here means no gradient was produced for w, not a computed numerical zero. This linear fixture isolates the gradient path; it is not the full benchmark model.

**One-step exercise.** Use plain SGD with learning rate 0.01 and no momentum or weight decay. The live branch changes w to 1.88 and v to 3.94, giving a next prediction of **22.2216**. The detached branch keeps w at 2 and changes v to 3.94, giving **23.64**. Derive both before running autograd; explain why equal predictions before the update did not imply equal learning afterward. These optimizer settings are for the fixture, not the benchmark's Adam configuration.

[[GRAD_WIDGET]]

[[GRADIENT_FINDING]]

A gradient norm describes local sensitivity under this batch and parameter state. It is not a causal feature attribution or a universal importance ranking. A missing gradient, a zero finite gradient and a nonfinite gradient are different observations; the trace keeps them separate.

**Failure diagnosis.** Explain why each bug is caught by a different check: (1) all context rows reach the head; (2) all nodes use the first root cutoff; (3) an activation logger replaces live tensors with detached values. A passing shape check alone does not catch all three.

## 6 · Reproduce the complete selected experiment

The named target is RelBench v1 Table 7, basic RDL on `rel-f1/driver-position`: five fresh seeds, ten complete epochs each, full graph and official query splits. The pinned model uses width128, two layers, fanouts `[128,64]`, uniform temporal sampling, batch512, Adam0.005 and mean L1. Choose the first strictly improved validation MAE checkpoint. Evaluate the frozen checkpoint and independently rescore every prediction by exact query identity.

[[FIG:scores]]

[[RESULTS]]

The reported closeness uses a predeclared absolute mean tolerance of 0.2 MAE. Seed SD is descriptive variability, not a confidence interval or a proof of equivalence. Final validation scores can differ from checkpoint-selection scores because evaluation resamples neighborhoods.

**What “full” covers here.** Every prescribed run, epoch, database row and final task query in this selected released-protocol experiment. It does not cover the paper's other 29 tasks. Exact historical training commit, seeds and runtime are not established. Released `[128,64]` fanouts differ from the paper table's `128`; feature statistics use the database through the test cap; original arrival and mutable-feature histories are unavailable. Numerical agreement does not erase these gaps.

The [protocol ledger](../labs/l131-reproduction.md) contains exact commands, archive hashes, source revision, budget reservations, the failed pilot and successful recovery, and the distinction between paper, local fixture, instrumented batch and full-data evidence. Instrumentation findings do not silently change the released algorithm.

## 7 · Your annotated trace and written defense

Submit the three working functions and an annotated trace that:

1. Maps a database row ID, occurrence index and query owner to their separate meanings.
2. Derives one recorded age and one coordinate of the time-vector addition.
3. Explains the observed `[B,128] → [B,1]` readout and which rows receive loss.
4. Predicts and explains the detach intervention, including the finite/nonfinite gradient distinction.
5. States the strongest reproduction claim supported by the seed artifacts and lists unresolved protocol gaps.

Return tomorrow and redraw the stack without opening the notebook. Explain why gradients can reach an unlabeled neighbor. Then compare with [Lesson 076](0076-encoder-predictor-stack.html) and [Lesson 115](0115-graph-ml-design-patterns.html); Lesson 132 will ask what identity-aware message passing adds to this baseline.

Author execution does not establish learner mastery: **PENDING_WRITTEN_DEFENSE**. Ask your tutor about any tensor, index, gradient or evidence claim that you cannot yet explain. Live Colab and deployment remain **NOT_CHECKED** unless separately verified.

<!-- sequence-next:start -->
**Carry this forward.** Ask how representations can depend on which entity is the current query root. [Continue to Lesson 132](0132-identity-aware-message-passing.html).
<!-- sequence-next:end -->
