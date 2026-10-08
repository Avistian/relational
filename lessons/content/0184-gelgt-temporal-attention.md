## One skill: separate access, selection and attention

A larger neighborhood helps only when it supplies useful information that was available for the query. By the end, you should be able to trace one GelGT prediction through those three decisions and reject a reproduction that confuses them.

This serves our mission: make relational-model gains defensible through visible computation and a valid experiment. [Lesson 145](0145-relational-graph-transformer.html) introduced the relational Transformer family; use the [course map](../reference/curriculum.html) if that lesson's title has changed. [Lesson 183](0183-graph-transformer-pretraining.html)'s proposed pretraining experiment asks a different question: GelGT is trained directly on its downstream task. You do not need an unfinished previous lesson to follow this one.

**Recall first.** A primary key identifies a row. A foreign key connects it to a row in another table. A query cutoff is the time at which a prediction must be possible. An attention weight tells us how much an already-admitted value contributes; it does not grant access to that value.

> **Scope check.** [[STATUS]]

## Why reach alone is insufficient

**In plain terms.** Imagine predicting a driver's next finishing position. One nearby race may connect the driver to a constructor, then to other drivers. A message-passing network transports information through these edges in successive layers. Compressing many paths into fixed-size vectors can lose distinctions. Attention can connect sampled tokens directly, but cannot use a useful row that the sampler discarded.

A **token** is a vector representing a sampled row for this query. A **hop** is one graph edge. GelGT addresses three bottlenecks: retain the local relational structure, choose useful distant nodes, then learn which time differences matter. It does not prove that every long-range problem disappears. Read the [primary paper §§3.1–3.2](https://arxiv.org/html/2605.15575v2#S3).

## Model architecture: database to one number

[[FIG:architecture]]

**Start with a query.** For regression, the published sampling budgets are 500 candidates and 300 retained tokens. `B` denotes a batch of queries, `K` the token count, and 512 the hidden width. Each token receives representations of its table type, hop distance, relative time, table fields and structural position. Normalization puts each channel on a controlled scale; concatenation joins channels side by side; a small multilayer perceptron mixes them into one vector.

**Choose context after encoding.** The seed is the query's own row. Nearby nodes are protected; more distant candidates are ranked using their dot product with the seed vector. A dot product multiplies matching coordinates and sums them. It is a learned compatibility score, not a guarantee of relevance.

**Compute two summaries.** Three GraphSAGE layers aggregate along retained edges, keeping local topology visible. In parallel, multi-head attention compares retained tokens and adds a time-dependent bias to each comparison. A head is a separate learned comparison subspace. The attention branch combines the query representation with a weighted neighbor summary.

**Fuse and predict.** The release learns a sigmoid gate: `g = 1/(1+exp(-w))`, between zero and one. It mixes `g × GNN + (1−g) × attention`; a two-layer head predicts one finishing-position value. The full released model and trainer appear in the notebook appendix. The paper/source depth discrepancy remains explicit in the [protocol ledger](../labs/l184-reproduction.md).

## Mechanism 1: preserve a legal relational route

**Worked example.** Driver 0 connects to races on days 5 and 15. At cutoff day 10, only day 5 is available under a strict prior-time rule. At day 20, both are available. These are different queries, even though the driver is unchanged.

[[FIG:cache]]

**Breadth-first sampling** visits all one-hop neighbors before expanding two-hop neighbors. With a finite budget, it favors broad local coverage. The retained **induced graph** includes the original edges whose two endpoints survived. An invalid future node must be rejected before traversal; merely hiding its target is insufficient.

> **Scope check.** The teaching sampler uses sorted ties, strict `< cutoff`, and excludes untimed rows. The release uses `<=` and admits untimed rows. The paper contains both strict and non-strict descriptions. These policies are recorded separately, not silently equated. The course policy still needs ingestion histories before it establishes real deployment availability.

## Mechanism 2: rank distant candidates without losing the seed

**Worked example.** Let the seed embedding be `[1,0]`. One-hop node A has embedding `[0,1]`; distant nodes B and C have `[0.1,0]` and `[2,0]`. Their seed dot products are 0, 0.1 and 2. With room for three tokens, protect seed and A, then retain C. B loses its slot.

| Token | Hop | Score | Three-token outcome |
|---|---:|---:|---|
| Seed | 0 | 1 | Protected |
| A | 1 | 0 | Protected |
| B | 2 | 0.1 | Removed |
| C | 2 | 2 | Retained |

This isolates semantic ranking: candidate graph, embeddings and budget stay fixed; only the distant score determines the last slot. If there are more protected nodes than slots, the course implementation raises an error. The release uses large sentinel scores and top-k; ties require care. After selection, remap edge indices to the new token positions.

**Try before the lab.** Reverse B and C's embeddings. Predict which edge disappears. Then decrease the budget to one: explain why quietly removing the seed would corrupt the prediction readout.

## Mechanism 3: learn a preferred temporal band

**In plain terms.** Useful evidence need not be the newest evidence. A model may prefer a band of lags, such as roughly two days ago. The center says where that band sits; the width says how quickly preference falls away.

For a pair of tokens, `Δt` is their absolute time difference. A Gaussian feature is:

`r(Δt) = exp(−((Δt − μ)/σ)²)`

Here `μ` is the center and `σ` is a positive width. The release uses `abs(raw_sigma)+0.00001`. There is no factor of one-half in this kernel. Several such features are passed through a learned linear projection to obtain one bias per attention head. Projection weights may be negative, so a large Gaussian feature need not increase attention.

**Worked example.** With lags `[0,2,4]`, center 2 and width 2, the squared normalized distances are `[1,0,1]`. Exponentiation gives `[0.368,1,0.368]`. For an illustrative identity projection and equal content scores, softmax converts these biases into weights that sum to one. Softmax exponentiates each score and divides by the sum of the exponentials.

[[FIG:attention]]

**The complete operation.** Learned query and key vectors produce content scores `QKᵀ/√d_head`; `d_head` is the width of one head. Add the temporal bias, apply softmax, and take the weighted sum of the value vectors `V`. This changes contribution, not temporal legality. The source-executed lab checks this entire attention layer against explicit matrix arithmetic and compares input gradients.

[[EXPLORER]]

**Predict an intervention.** Set projection to zero. The temporal contribution should disappear. Set it to minus one: the preference reverses. This is why a special-case scalar kernel calculation cannot establish an unconditional performance gain or a universal attention-ratio theorem for every trained projection.

**Follow the weights through to the output.** Keep lags `[0,2,4]`, center 2, width 2, equal content scores and values `[1,3,9]` fixed. Change only the scalar projection of the Gaussian feature:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:3px"><thead><tr><th>Projection</th><th>Middle weight</th><th>Output</th></tr></thead><tbody><tr><td>+1</td><td>0.48475</td><td>4.03050</td></tr><tr><td>0</td><td>1/3</td><td>4.33333</td></tr><tr><td>−1</td><td>0.20994</td><td>4.58011</td></tr></tbody></table>

For +1, both outer weights are 0.25763, so the output is `0.25763 × 1 + 0.48475 × 3 + 0.25763 × 9`. Favoring the middle value pulls the answer below the uniform average. A negative projection reverses that preference. These are controlled single-head outputs, not fitted model predictions.

**Transfer the trace.** Replace all three values by 7 while preserving the biases. Every output becomes 7 because the weights sum to one. Attention can change substantially without changing the prediction when the retained values agree. Check both interventions with `attention_trace` and explicit weighted sums.


## Reproduction: a failed prerequisite is evidence

[[PREDICT]]

**Named target.** GelGT v2 Table 2, `rel-f1/driver-position`: reported test MAE **3.7345 ± 0.1200**. Mean absolute error averages the absolute distance between predictions and labels; lower is better. We have not produced a new model MAE. The [official source](https://github.com/USTC-DataDarknessLab/GelGT/tree/1997b2c2f480ce5d3cbdb48d46f33cc303f5feb4) is pinned, rather than following a moving branch.

**What ran.** All nine cached database tables and all three task splits are authenticated. SQL reconstructs every target from raw race-result rows. Original sampling functions execute on a temporal counterexample. Original attention executes on controlled tensors; independent arithmetic checks outputs and gradients. These checks have different evidence boundaries.

[[RESULTS]]

**Why the gate fails.** The release's context dictionary uses `S[table][entity]`, then retrieves that entry for every matching entity in a 10,000-row chunk. Each complete F1 split fits within one chunk. The later query overwrites the earlier entry. The synthetic source execution demonstrates that this can pass a future race to an earlier query. The stored lag is relative to the later cutoff, so it may still look nonnegative.

**What the counts mean.** Cache collisions count lost distinct query entries, not observed future-row occurrences across the complete database. We did not run the full graph materialization, training, or full-population context audit after this blocker. Correct labels do not imply correct features.

**What would permit a future experiment.** Preserve full query keys through sampling and caching; authenticate per-query raw timestamps; settle padding, fitting scope, evaluation dropout, source/paper configuration and seed aggregation; then measure the full cost before dispatch. A repaired release defines a new declared protocol. It cannot be described as an unchanged historical reproduction.

> **Scope check.** The aggregate ceiling is $10 including retries and verification, with a planned stop at $8 and $2 reserve. No paid pilot ran because the scientific gate failed earlier. Full reproduction remains **INCOMPLETE**. Nothing here changes the learner's mastery record or claims deployment.

## Lab: implement, intervene, defend

Download the [student notebook](../labs/0184-gelgt-temporal-attention.ipynb), inspect the [executed solution](../labs/html/0184-gelgt-temporal-attention.html), or use the [quick reference](../reference/gelgt-temporal-attention.html). The portable notebook embeds the data, source and figures. It needs Python, NumPy, pandas, PyArrow, DuckDB and PyTorch; live Colab is **NOT_CHECKED**.

1. Implement strict two-hop temporal sampling; test earlier and equal cutoffs.
2. Implement the Gaussian feature; compare it against the actual release's bias and attention arithmetic.
3. Implement complete-key MAE; reject duplicate, missing and nonfinite predictions.
4. Run the source counterexample and all 8,712 label reconstructions. Change one assumption and explain the boundary it affects.

The synthetic attention output is not a prediction benchmark. The notebook deliberately leaves fresh training gated off. Inspect the [full protocol and exact commands](../labs/l184-reproduction.md) and [machine-readable evidence](../labs/evidence/l184/report.json).

[[TEACHBACK]]

**Bridge forward.** A temporally valid prediction is still not a causal intervention. [Lesson 185](0185-causal-relational-data.html) asks whether acting on a prediction changes an outcome. First make the information contract correct; then ask what decision the model supports.

Ask the teaching agent about any unclear step, or paste your written defense for feedback. Preparing this lesson does not count as completing it.
