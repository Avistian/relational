<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Reading guide">
<p class="sequence-eyebrow">From one target to a ranked candidate set</p>
<p><strong>Reading route.</strong> Understand two towers → trace the local and distant branches → replace owner-specific scores → evaluate the full catalog.</p>
<details><summary>Quick prerequisite reminder</summary><p>A bipartite prediction problem links two entity roles, here facilities and sponsors. A dot product multiplies corresponding vector coordinates and sums them. B × N means a score for every one of B queries against every one of N candidates. “Local” means present in this query’s sampled graph, not geographically nearby.</p></details>
</aside>
<!-- sequence-review:end -->

**The win:** implement a ranking rule that uses a different representation for each nearby facility–sponsor pair while still scoring sponsors outside that facility’s sampled graph. Start with the guided trace; use the notebook for the three implementation contracts and your written defense.

[Lesson 143](0143-relgnn-reproduction.html) asked what makes a reproduction claim defensible. Its model predicted one number for one entity. Recommendation adds a different problem: **which of many candidate entities should come first?** Better message passing alone does not decide how to represent each candidate relative to the query. This lesson connects the mission’s relational-model thesis to a concrete sponsor recommendation task.

Prerequisites: [bipartite graphs](0095-bipartite-graphs.html), query-owned temporal sampling, and validation-only selection. The same facility at two dates defines two different queries.

## 1 · Why two towers leave something unresolved

A **query** here is a facility and a cutoff time. A **candidate** is a sponsor. The task asks which sponsors will conduct trials at that facility in the next 365 days. The graph connects facilities, studies, sponsors, and other tables through foreign keys. At prediction time, temporal sampling admits timestamped rows only through the query cutoff.

A **two-tower model** represents the query as a vector and each candidate as another vector. Their dot product—multiply matching coordinates and sum—gives a score. Candidates with larger scores are ranked first. An item vector can be reused across queries, making catalog-wide scoring efficient. But that sponsor’s vector does not itself change to describe its relationship with this particular facility.

**Worked example.** Facility vector `[1, 0]` and sponsor vector `[2, 3]` yield score 2. A second facility vector `[0, 1]` yields score 3 for the same sponsor. Two towers can therefore produce personalized scores. Their limitation is narrower: the sponsor representation remains pair-agnostic. Do not confuse a shared item embedding with identical rankings for every user.

A **pair-wise representation** instead encodes a candidate inside the query’s context. This can capture a facility→study→sponsor connection and its surrounding facts. Computing a separate graph for every possible pair would be expensive. Restricting prediction to the facility’s local graph is cheaper, but any correct sponsor outside that graph becomes unreachable. These complementary limitations motivate [ContextGNN, §§3–4](https://arxiv.org/html/2411.19513v1#S4).

## 2 · Model architecture: share the graph, split the scores

**In plain terms.** Run one graph computation around the facility. Read both the facility state and its contextual sponsor states from that computation. Use a separate shallow representation to score the rest of the sponsor catalog.

[[FIG:architecture]]

The released RelBench benchmark uses four graph layers for `rel-trial`. It samples at most 128, 64, 32, then 16 neighbors per hop with the `last` temporal strategy, which favors recent eligible neighbors. Sampling is **query-owned**: each copied node belongs to a particular root query and must respect that root’s cutoff.

**Encode rows.** Each table has a ResNet row encoder. A ResNet is a feed-forward network with residual additions that help carry information through multiple blocks. Categorical, numerical, text-embedding, and timestamp columns become a common hidden width `h`. Relative-age encodings express time relative to each root. A learned identity vector marks the root so the computation can distinguish the querying facility from other nodes.

**Propagate.** Heterogeneous GraphSAGE aggregates neighbors separately for each foreign-key relation, then combines relation outputs. “Heterogeneous” means table and relation types retain distinct operations. This release uses sum aggregation, node-wise layer normalization, and ReLU between layers. A sampled sponsor state `h_ui` has `h` coordinates and depends on query `u`; the root state is `h_u`.

**Score outside the graph.** Project the root into `d` coordinates: `z_u = W h_u + b`. The shallow sponsor vector `e_i` also has `d` coordinates. “Shallow” describes the candidate encoder, not the query encoder: the query still uses the GNN. The source supports learned ID lookups, feature-based embeddings, or their sum (`FUSION`). Feature-based embeddings pass sponsor columns through a small residual network; ID lookups are learned vectors indexed by sponsor ID.

> **Scope check.** With learned sponsor-ID lookups, entirely new sponsors have no trained lookup vectors. New query facilities and unseen candidate sponsors are different generalization questions. `FUSION` in the source’s item encoder is also different from the final fusion of local and distant scores.

**Score inside the graph.** The paper explains local pair scores plus a personalized offset. The pinned release makes this more specific:

```
tower[u, i] = dot(z_u, e_i) + tower_offset(z_u)
local[u, i] = head(h_ui) + dot(h_u, h_ui) + local_offset(z_u)
```

A **head** maps a hidden vector to the scalar score. An **offset** shifts all scores from one branch for a particular query. Their relative shift lets the model favor contextual candidates or exploration. Both branches share the graph computation. The full visible notebook includes the encoders, typed graph layers, item encoder, score construction, and trainer—not just a call to a packaged ContextGNN.

> **Source distinction.** The release does not inject shallow sponsor embeddings into graph inputs as described in the paper’s two-tower discussion. Its contextual head also adds a root–candidate dot product and uses two learned offsets. We teach and execute this pinned release explicitly; historical paper identity remains a separate claim. [Source](https://github.com/kumo-ai/ContextGNN/tree/ca4a96985b7ef73c36a40da32e70710ad9b59a1e/contextgnn/nn/models).

## 3 · Fusion is replacement, with ownership

Initialize a matrix of tower scores with shape `B × N`: `B` query rows and `N` sponsor candidates. For every sampled sponsor, replace precisely its `(query owner, sponsor ID)` entry with its local score. A sponsor can appear under several query owners, with different contextual vectors and different cutoffs.

[[FIG:ownership]]

**Predict first.** If a local sponsor’s tower score is 3 and its contextual score is 1, should the final score be 1, 3, or 4? The released operator chooses 1. It does not add the two branches and does not take their maximum. Other sponsors keep their tower scores.

[[FIG:ranking]]

[[RANKING_WIDGET]]

The illustration holds learned vectors fixed, so changing only the offset isolates score calibration. Changing graph membership is a second intervention. A sponsor outside the sampled graph is not necessarily new or never previously encountered; sampling may simply omit it.

**Only the relative branch offset changes the ranking.** Fix one local sponsor A and two distant sponsors B and C. After replacement, let their logits be `[1,2,0]`. The following trace uses a single positive label A, a valid special case of the training loss:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:.4em .25em"><thead><tr><th>Offsets changed</th><th>Final A,B,C</th><th>Top</th><th>Loss</th></tr></thead><tbody><tr><td>Neither</td><td>1,2,0</td><td>B</td><td>1.408</td></tr><tr><td>Both +7</td><td>8,9,7</td><td>B</td><td>1.408</td></tr><tr><td>Local +2</td><td>3,2,0</td><td>A</td><td>0.349</td></tr></tbody></table>

For this one-positive case, loss is `−log(exp(score_A) / Σ exp(score))`. Adding 7 to every score multiplies numerator and denominator by the same factor, so probabilities, loss and rank are unchanged. Raising just the local branch changes its competition with the distant branch. The toy holds graph states, membership and every other parameter fixed; it isolates offset behavior rather than predicting a trained MAP improvement. [Released score construction](../labs/sources/l144/contextgnn/nn/models/contextgnn.py)

**Try it after the example.** Increase both branch offsets by another 100. Which entries change, and which ranking and loss values remain? <details><summary>Check your reasoning</summary>Every final logit rises by 100, while each row retains the same ranking, probabilities and loss. Stable log-softmax subtracts a shared maximum instead of exponentiating these large raw scores directly.</details>

**TODO 1 — `fuse_scores`.** Implement the owner-specific replacement without mutating the supplied tower matrix. The full model calls your function. The CHECK verifies two queries that share a candidate ID and checks gradients: overwritten tower entries must receive no gradient through the final score, while local values and offsets must receive it.

## 4 · Train on future links, constrain access to the past

The label for `(facility, t)` is the distinct set of sponsors linked through studies whose facility-study date lies in `(t, t + 365 days]`. The left boundary is open; the right is closed. The source joins sponsor-study records by study ID. Preserve that exact label rule instead of adding an undocumented sponsor-date filter. Query rows without eligible targets and dangling entities follow RelBench’s filtering rules. [Task source](https://github.com/stanford-star/relbench/blob/v1.1.0/relbench/tasks/trial.py).

The training objective is **sparse multi-positive cross-entropy**. A query can have several correct sponsors. The implementation consumes the listed positive `(query, sponsor)` pairs and trains scores against the full sponsor vocabulary. “Sparse” refers to storing positive pairs compactly; it does not mean this benchmark samples only a few negative candidates. The release has a separate sampled-softmax path, but the selected benchmark does not call it.

**TODO 2 — `audit_cutoffs`.** Check every timestamp against its own query owner. With cutoffs `[5, 10]`, a timestamp 6 owned by query 0 is a violation even though it is below the batch maximum 10. Your function runs on the actual sampled batches in the full trainer.

> **Scope check.** Correct event-time filtering does not prove when a non-timestamped attribute became available in the real world. We audit the released temporal contract and state the unresolved availability-time boundary.

## 5 · Rank and score the complete query population

**Average precision (AP)** rewards putting relevant items early. At every relevant rank, compute the fraction of recommendations so far that are correct. Sum those fractions and divide by the smaller of the number of relevant items and the cutoff `k`. **Mean average precision (MAP)** averages AP across query rows. Here `k = 10`.

[[FIG:metric]]

**Worked example.** Relevant sponsors are `{A, C}` and the top-three list is `[A, B, C]`. Precision is 1 at the first hit and 2/3 at the second hit. AP@3 is `(1 + 2/3) / 2 = 5/6`. A second query with AP 1/2 gives MAP `(5/6 + 1/2) / 2 = 2/3`. One repeated facility at two dates still contributes two distinct query rows.

**TODO 3 — `keyed_map`.** Align predictions by `(facility ID, cutoff)` before scoring. Reject missing or duplicate query keys and repeated candidate IDs. The CHECK shuffles prediction order so a positional shortcut fails. The full trainer compares your metric with RelBench’s evaluator on every validation pass.

The paper reports MAP as percentages: 28.02% means 0.2802, not 28.02 in the Python evaluator. A ranking over a small handpicked candidate set cannot establish the full-catalog MAP target.

## 6 · Execute the named reproduction and defend its boundary

The selected target is **Table 2, `rel-trial/site-sponsor-run`: ContextGNN versus ShallowItem**, with published test MAP **28.02% versus 10.66%**. This task is especially informative because both nearby context and distant candidates can matter. It does not establish the paper’s cross-task average improvement. [Paper results](https://arxiv.org/html/2411.19513v1#S5).

The complete released search requests 50 Optuna trials for each architecture, then five repeat fits of each selected configuration. Optuna proposes hyperparameters using validation results; its median pruner can end unpromising trials. Each fit runs up to 20 epochs and stops when validation MAP falls more than 0.001 below its best value. The source also caps training at 2,001 batches per epoch because its stopping comparison is `steps > 2000`. We preserve and report that detail.

The paper describes a narrower search than the source. The release additionally searches encoder depth, item width, embedding mode, normalization, and learning-rate decay. Its Optuna sampler is unseeded, and repeated fits consume a continuing random stream. Our executable reconstruction seeds the search and each fit explicitly; these are recorded deviations, not recovered historical seeds. Validation chooses configurations and checkpoints; test scores never choose settings.

[[RESULTS]]

**A source-matching failure mode.** The RHS item encoder caches its embeddings in evaluation mode. The release does not clear that cache when training resumes. Our mechanism probe tests whether changing item weights leaves evaluation output stale until the cache is cleared. The reproduction preserves the released behavior. A repaired-cache comparison would be a separate experiment and must not silently replace these results.

The reproduction budget is $10 total, including retries, preparation, and validation. The measured pilot is a cost and correctness probe, not a published-score reproduction. If the complete search cannot fit, preserve the runnable full experiment and mark it incomplete. Never turn a partial search into “full reproduction” by renaming it.

[Protocol and execution commands](../labs/l144-reproduction.md) · [Student notebook](../labs/0144-contextgnn.ipynb) · [Executed reference notebook](../labs/html/0144-contextgnn.html) · [Quick reference](../reference/contextgnn.html).

## 7 · Exit: explain the score and defend the claim

Without looking back, explain why the same sponsor can have different contextual embeddings under two facilities. Trace which final score changes when its owner-specific membership changes. Then defend the evidence status of a one-epoch pilot that uses all training rows but skips the 50-trial search.

[[TEACHBACK]]

**Written defense:** identify a failure that output parity cannot rule out, explain the cached-embedding test, and state what additional evidence would support a full selected-task reproduction. The author’s runs do not establish your mastery. Ask the teaching agent to review your three functions and your explanation; follow-up questions are welcome.

For the primary reading, use [Yuan et al., ContextGNN, §§3–5](https://arxiv.org/abs/2411.19513), then compare the [pinned implementation](https://github.com/kumo-ai/ContextGNN/tree/ca4a96985b7ef73c36a40da32e70710ad9b59a1e). This prepares the next lesson’s question: how can a relational transformer encode the query’s context without this particular message-passing design?


<!-- sequence-next:start -->
**Carry this forward.** ContextGNN changes how query–candidate scores are formed. Lesson 145 returns to a scalar target and changes communication among sampled rows with RelGT. [Continue to Lesson 145](0145-relational-graph-transformer.html).
<!-- sequence-next:end -->
