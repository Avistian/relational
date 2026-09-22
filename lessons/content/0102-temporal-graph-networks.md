<div class="package-links"><strong>Your deliverable:</strong> implement the message → memory path, explain its timing, and defend a complete Wikipedia TGN-attn replay.<br><a href="../labs/0102-temporal-graph-networks.ipynb">Student notebook</a> · <a href="../labs/solutions/0102-temporal-graph-networks.ipynb">Executed solution</a> · <a href="../labs/html/0102-temporal-graph-networks.html">Read the lab</a> · <a href="../labs/l102-reproduction.md">Reproduction contract</a></div>

## 1 · Retrieve before reading

[[WARMUP]]

Close the notes. What makes a neighbor legal at a historical prediction time in [L101](0101-static-vs-temporal.html)? Why does [L097](0097-negative-sampling.html) distinguish sampled negatives from full-catalog ranking? In [L098](0098-hetero-mini-batching.html), what is shared across a mini-batch and what depends on each seed query?

<details><summary>Feedback — open after making an attempt</summary><p>Inputs must have been observable by the query cutoff, including their feature versions. Candidate sampling changes the evaluation question. Model weights are shared, while the seed identities and sampled dependencies specify each computation. Today, a fourth object matters: a mutable per-node state that must correspond to the correct historical point.</p></details>

**The tangible win.** Given an event stream, you can say exactly which event first affects a node’s memory, which prediction can read it, and which loss trains the memory updater. Work through sections 2–6 first. Then implement the three notebook tasks. Treat the full benchmark as a separate, longer experiment.

**Why this belongs in the mission.** A relational database records interactions: purchases, messages, edits, appointments. L101 established which records are legal. It did not show how a model can compress a long stream into a useful state. TGN supplies that mechanism. Whether it improves a real relational task remains an empirical question with a declared evaluation protocol.

Primary reading: [Rossi et al., 2020, §§3.1–3.2, Figure 2 and Appendix A.2](https://arxiv.org/html/2006.10637v3). Read those sections alongside the computation below. The implementation reference is the [publication-era release](https://github.com/twitter-research/tgn/tree/e38cdf85998c6ca077167610dc4e769a688efa95).

## 2 · An event changes history; a query asks about history

**An event** is an observed interaction `(u, v, t, e)`: source node, destination node, timestamp and event feature vector. Wikipedia gives user–page edits. User 7 and page 7 need distinct node IDs, just as the two sides of the bipartite graph did in [L095](0095-bipartite-graphs.html). Repeated edits are separate events, not duplicate edges to discard. Each prediction query supplies its timestamp; the decoder scores a candidate interaction at that time.

**Memory** `sᵢ` is a vector summarizing previously consumed events for node `i`. **Last-update time** `τᵢ` says when its most recently incorporated event occurred. A new node starts with zero memory. Neither object is a learned ID embedding: the updater’s shared weights are learned, while each node’s memory is computed from its history.

**An embedding** `zᵢ(t)` is the representation used for a particular query. Memory and embedding are different. A node may not interact for an hour, so its stored memory stays unchanged. Its neighbors can still change. A query-time graph operator lets its embedding consult those neighbors and the elapsed times.

> **In plain terms.** Memory is the running notebook; the embedding is the answer you construct from that notebook and the legal neighborhood when someone asks a question.

**Worked example.** A edits B at time 1; A edits C at time 2; B interacts with C at time 3. Before the first event, all memories are zero. After observing the first event, A and B have information that C lacks. But if the first two events are predicted together, their memory inputs must share a declared batch boundary. Do not quietly turn batch inference into sequential event inference.

## 3 · Model architecture: trace the two paths

[[FIG:architecture]]

**Prediction path.** A batch of `B` real interactions creates `3B` node queries: sources, true destinations and sampled destinations. For each query, retrieve up to ten prior interactions. The same encoder weights process every node. The same decoder scores positive and negative pairs.

**State path.** Pending messages from older batches produce a differentiable candidate memory. The candidate serves the current prediction. Only after that prediction do the current real events create new pending messages. Random negative pairs do not update the event history.

The pictured variant is **TGN-attn** from Table 1: last-message aggregation, identity message function, a GRU updater and one two-head graph-attention layer. All three feature widths are 172 in the Wikipedia release. “Identity message” means concatenate the inputs without an additional learned MLP; it does not mean copy the old state unchanged.

**The dimensions are a useful checksum.** A message contains two 172-wide states, a 172-wide event vector and a 172-wide time encoding: `688` numbers. A query contains state and zero-lag time encoding: `344`. Each key/value input contains neighbor state, historical edge features and relative-time encoding: `516`. Attention returns `344` numbers; a merge MLP brings this back to a 172-wide embedding. See the released [temporal attention layer](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/model/temporal_attention.py).

## 4 · Build one message, aggregate repeated nodes, update memory

### 4.1 What information enters a message?

The message for A after event `(A,B,t,e)` uses A’s state, B’s state, the observed event features and the elapsed time since A’s previous update. The reverse message swaps the endpoint roles and uses B’s own elapsed time. Different last-update times mean the two time encodings need not agree.

<div class="tgn-equation">m_A = concat(s_A, s_B, e, φ(t − τ_A))<br>m_B = concat(s_B, s_A, e, φ(t − τ_B))</div>

The time encoder is a learned cosine bank: coordinate `k` is `cos(wₖ Δt + bₖ)`. `Δt` is the elapsed duration; `wₖ` controls how quickly that coordinate oscillates; `bₖ` is its phase. It converts one duration into multiple smooth numerical features. These weights receive gradients where the encoding participates in the prediction graph. The [released initializer](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/model/time_encoding.py) spreads frequencies from 1 to 10⁻⁹ and initializes phases to zero. L103 will examine time encoding in more detail.

### 4.2 Which message survives a batch?

**Worked example.** The ordered recipient IDs are `[B,A,B,A]`. Their message positions are `[0,1,2,3]`. Last-message aggregation returns position 3 for A and position 2 for B: `[3,2]` in sorted-node order. Selecting the first occurrence or averaging the messages changes the model.

This rule assumes chronological input. For timestamp ties, the release keeps the last record in stream order. It is not a permutation-invariant set reduction. Our real-data replay retains the input order and exact split IDs.

**Task 1** implements this index selection. **Task 2** constructs the two directed messages. Both functions are called by the model you train; they are not disconnected worksheet exercises.

### 4.3 Why a GRU instead of replacing the state?

A **gated recurrent unit (GRU)** learns how much old information to retain. Let `m` be the aggregated message and `s` the old memory. Every gate below is a vector with the same width as `s`. `σ` is the sigmoid function, mapping a real number into `(0,1)`; `⊙` means coordinate-wise multiplication.

<div class="tgn-equation">r = σ(Wᵣm + bᵣ + Uᵣs + cᵣ)<br>z = σ(W_zm + b_z + U_zs + c_z)<br>n = tanh(W_nm + b_n + r ⊙ (U_ns + c_n))<br>s_new = (1 − z) ⊙ n + z ⊙ s</div>

`W` matrices transform the message; `U` matrices transform the old state; `b` and `c` are learned biases. The reset gate `r` controls the old-state contribution inside the candidate `n`. The update gate `z` controls retention in the final mixture. This is the [PyTorch GRUCell convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html): a larger `z` retains more old memory. Other texts sometimes name the complementary gate `z`.

[[FIG:gru-trace]]

**Worked example.** Hold both gates at `0.5`, old memory at `0.4`, input contribution at `1`, hidden candidate weight at `0.5`, and hidden bias at zero. The reset-weighted hidden contribution is `0.5 × 0.5 × 0.4 = 0.1`. The candidate is `tanh(1.1) = 0.8005`. Updated memory is `0.5 × 0.4 + 0.5 × 0.8005 = 0.6002` after rounding. This is a one-coordinate illustration, not a measured Wikipedia state.

**Task 3** applies the GRU only to selected nodes, records their update times and returns a candidate state without mutating its input. Its CHECK checks both values and gradients. Unchanged nodes must retain their old values.

## 5 · The training puzzle: learn from history without predicting yourself

[[PREDICT]]

A naive loop predicts an event, updates memory, then detaches the state. Its loss was computed before that update, so the updater has no path into that loss. Moving the current event’s update before the prediction gives a gradient but leaks the answer. The [paper’s Figure 2](https://arxiv.org/html/2006.10637v3#S3.F2) resolves both problems with delayed messages.

[[FIG:batch-timeline]]

**Step 1 — read pending history.** Take messages from events that were observed in earlier batches. They are ordinary stored data at this point.

**Step 2 — recompute the update.** Apply the current GRU weights to those messages and the stored memory. This produces candidate state `S*` inside the current autograd graph. Autograd records the operations needed to differentiate the loss.

**Step 3 — predict and train.** Build embeddings from `S*`, score the real and sampled pairs, compute the loss, and update the model weights. The chain `loss → embedding → S* → GRU weights` is intact.

**Step 4 — preserve the temporal boundary.** Consume old messages for current real endpoints and queue messages describing the newly observed real events. Detach memory and pending messages between optimizer steps. Detaching preserves values while cutting the gradient graph, preventing backpropagation through the entire event history.

**Read the actual training call.** The notebook exposes this loop and the `probabilities` method it calls. `p` and `n` are the positive and negative scores; the loss below is the sum of their two mean binary cross-entropies.

```python
optimizer.zero_grad()
p, n = model.probabilities(u, v, sampled_destinations, times, edge_ids)
loss = nn.functional.binary_cross_entropy(p, torch.ones_like(p))
loss = loss + nn.functional.binary_cross_entropy(n, torch.zeros_like(n))
loss.backward()
optimizer.step()
model.detach_state()
```

`probabilities` first builds the candidate state and embeddings, then commits consumed messages and queues new ones. The returned score tensors retain the candidate’s gradient path. Detaching earlier, before `backward`, would require care not to sever a needed graph; skipping the detach retains history across optimizer steps.

The released implementation computes scores from embeddings already made before it queues the current messages. The ordering inside a Python function is less important than the actual data dependency. A test that changes the current event’s features must leave its current score unchanged, while changing what future queries can read.

**A release detail about gradients.** The stored message vector already contains a time encoding. The release detaches that vector between batches, so a later loss does not backpropagate through its old encoding calculation. The shared time encoder still learns through query-time attention. The recomputed GRU update is the differentiable path we test here.

**Do not confuse memory with adjacency.** All queries in a batch share the same candidate memory. Their temporal neighbor cutoffs still differ. The query at time 2 can retrieve the time-1 edge even when both events belong to the same batch. The memory channel is stale within the batch; the neighborhood channel is filtered per query. This distinction explains why batch size is part of the scientific protocol, not just a performance setting.

## 6 · Intervene on the event stream

**Predict first.** For the second event, will changing batch size from two to one change A’s memory? Will allowing the current event to write memory change the current score even when the model weights stay fixed?

[[TIMELINE]]

This widget uses the scalar GRU from section 4 with zero initial state and a fixed illustrative decoder `sigmoid(s_source + s_destination)`. It displays prediction-time **candidate memory**, after applying previous-batch messages. The release physically commits only active endpoints; the widget’s fixed weights let us show the equivalent candidate values directly. It isolates the memory channel; there is no graph-attention contribution. Keep the event and weights fixed while switching batch size or the illegal update. The legal score remains visible beside the intervention.

<details><summary>Feedback</summary><p>At event 2 with batch size 2, A still has zero memory, so the legal scalar score is 0.5. With batch size 1, event 1 has updated A to 0.5 × tanh(1) ≈ 0.3808; C remains zero. The score becomes about 0.5941. Including event 2 before scoring changes inputs using the very event being predicted. That is leakage, not a better historical summary.</p></details>

## 7 · From memory to an actual prediction

**Temporal neighbors.** Find the ten most recent interactions strictly before each query timestamp. Node ID zero is padding. The original arrays are searched with a left insertion boundary, so same-time interactions are excluded from adjacency. Timestamp ties that cross memory batches need a separate ordering convention; a strict adjacency filter alone cannot guarantee that every state contribution has strictly earlier time.

**Attention.** Project the query and historical key/value inputs into two heads. Each head computes query–key dot products, scales them by the square root of its width, masks padding, and softmax-normalizes across neighbors. The weighted value sum selects relevant historical context. Concatenate the attention output with the node’s own state and pass it through a two-layer MLP. An MLP is a sequence of learned linear maps with a nonlinear activation between them.

An empty neighborhood needs explicit handling. Softmax over only masked positions is undefined. Temporarily allow one padding slot, compute attention, then zero the attention output for that query. Retain the own-state merge path. “No neighbors” is not the same as “no embedding.”

**Decoder and objective.** Concatenate a source embedding with a candidate destination embedding. A `344 → 172 → 1` MLP returns a logit; sigmoid turns it into a score between zero and one. For each observed interaction, sample one destination as a negative. Training minimizes the mean positive binary cross-entropy plus the mean negative binary cross-entropy:

<div class="tgn-equation">L = mean[−log p(real pair)] + mean[−log(1 − p(sampled pair))]</div>

The CSV also contains a node-classification label. This link-prediction replay ignores that column: its binary targets come from observed pairs and sampled pairs.

A sampled destination can coincide with a real or previously observed pair: the release does not reject those collisions. This is a property of its candidate protocol, not a guarantee that every sampled pair is an impossible interaction. L097’s candidate-set warning applies directly.

**Inference.** Freeze learned weights, not historical state. After a real interaction is observed, later queries may use it. Score all hypothetical candidate destinations against the same pre-observation state. Do not call a mutating event-consumption API repeatedly to rank hypothetical pairs; that would invent history. Back up and restore state for independent evaluation branches.

## 8 · Reproduce a named result, not a resemblance

The target is **Rossi et al. v3, Table 2, TGN-attn on Wikipedia**: AP `98.46 ± 0.1%` in the transductive column and `97.81 ± 0.1%` in the inductive column. The requested complete run covers all ten initializations, the full data and both evaluation branches. [Paper Table 2](https://arxiv.org/html/2006.10637v3#S5.T2).

**Held fixed.** One attention layer, two heads, ten recent neighbors, width 172, dropout 0.1, Adam at 0.0001, chronological batches of 200, up to 50 epochs and patience five. Select by all-event validation AP, never test AP. A fixed tolerance of 0.5 percentage point for each mean was recorded before results; it is a course closeness criterion, not a statistical equivalence test.

**Complete input.** The hash-pinned JODIE Wikipedia file has 157,474 events and 9,227 nodes. Zero padding adds one state row. The release’s 70% and 85% time quantiles define the cutoffs. It samples 10% of all nodes from nodes appearing after the first cutoff and removes their early interactions from training. The resulting partitions here have 81,029 training, 23,621 validation and 23,621 test events. New-node subsets contain 12,016 validation and 11,715 test events. New nodes mean nodes absent from the training interactions, including naturally late arrivals.

**What the metrics actually average.** AP, or average precision, summarizes precision as positive examples are recovered down a score ranking. Here the evaluator computes it separately for each chronological batch containing real and sampled pairs, then averages batch APs equally. It is not pooled AP over all events, and it is not full-catalog retrieval. Even the smaller final batch has equal weight. **Worked example:** sort scores 0.9, 0.8, 0.7, 0.1 with labels 1, 0, 1, 0. The positive ranks are 1 and 3, with precisions 1 and 2/3. AP is their mean, 5/6 ≈ 0.8333. For tied scores, evaluate precision after the whole tied group. The saved predictions permit an independent reconstruction of every batch metric.

**Names versus populations.** The release evaluates its “transductive” result on all test edges, including edges involving new nodes. The new-node branch evaluates only events involving at least one node absent from training. Its negative sampler draws from destination IDs appearing in that subset; that does not guarantee every sampled destination is itself unseen. We report the actual populations rather than silently rewriting the release to fit a label. The all-event sampler uses the full dataset’s destination-ID catalog, including future arrivals. That is a benchmark convention, not an unknown-catalog deployment simulation. The full evaluation neighbor index also restores pre-cutoff interactions of nodes withheld from training; they can serve as historical context at inference. Event timestamps are treated as observation times; delayed ingestion and historical feature versions from L101 need additional handling in a real database.

**Two different kinds of reproducibility.** The modern port uses explicit independent seeds 0–9. The original script seeds once before ten consecutive runs, so its later random streams depend on earlier runs and stopping times. Python 3.12 also requires converting a set to a tuple before `random.sample`; the archived split IDs make that choice auditable. We reproduce the released computation and documented modern protocol, while historical execution identity remains **INCOMPARABLE**.

**The checkpoint trap.** The release’s `state_dict` includes persistent memory and last-update times but excludes pending messages. Early stopping therefore restores the best weights and persistent memory while retaining stopping-epoch pending messages. The named replay preserves that choice and records it. A checked warm-state example changes its next probability by about 0.00164 when pending messages alone are omitted. A robust deployment checkpoint should save parameters, memory, clocks, pending messages, stream position and relevant RNG/optimizer state together. Fixing the historical behavior would define a separate experiment.

[[PAPER_RESULTS]]

[[FIG:results]]

> **Scope check.** These are author-reference results, not proof of learner mastery. Numerical closeness cannot establish historical identity. Reddit, Twitter, dynamic node classification, competing models and the paper’s ablations are outside this selected replay. Full-paper reproduction remains **NOT_ESTABLISHED**.

**A separate short run.** The notebook’s default training exercise uses 600 real training events, 200 events per evaluation branch and two epochs, retaining the full feature widths and architecture. Neighbor retrieval still uses the complete strictly historical interaction index; optimization and memory replay are shortened. It is deliberately too small to claim the paper’s score. Its purpose is to prove that the learner’s three functions participate in a working train/evaluation loop.

[[SMOKE_RESULTS]]

## 9 · Lab, checks and written defense

In the [student notebook](../labs/0102-temporal-graph-networks.ipynb), implement last-message selection, raw-message concatenation and the differentiable GRU state update. Run each CHECK before continuing. All data processing, attention, decoder, optimizer loop and evaluation are visible in the notebook. The full experiment is gated explicitly because it takes much longer than the teaching run.

**Verified mechanism boundaries.** The checker compares a repeated-node stream against the pinned original implementation, including probabilities, memory, pending messages and parameter gradients. Additional tests cover current-event feature leakage, strict neighbor cutoffs, padding, mutation safety and consistent state restoration. These are implementation tests, not paper-score evidence.

**EXIT — submit five answers with your run artifacts.**

1. Trace A’s pending message and committed memory across the first two batches. Explain why the time-2 query can see a time-1 edge but not its memory update.
2. Draw the gradient path that trains the GRU. State exactly where detaching breaks that path across batches.
3. Explain why hypothetical negatives never write memory, but real test events can change later test predictions.
4. Recompute one batch AP and the final unweighted batch mean from saved scores. Contrast it with pooled AP.
5. Separate completed runs, numerical tolerance, historical differences and the experiments still unrun. Explain the checkpoint caveat without calling it a harmless implementation detail.

Status stays **PENDING_WRITTEN_DEFENSE** until you supply that reasoning. Ask the teacher follow-up questions about any state transition or equation you cannot yet explain cold.

**Next connection.** L103 studies TGAT’s time-encoded attention without TGN’s node memory. L104 will audit leakage through the whole temporal pipeline. In relational modeling, carry both ideas forward: legal historical access from L101 and explicit state-transition boundaries from L102.

Keep the [TGN reference card](../reference/tgn-memory.html) for later retrieval. Revisit the memory/embedding distinction tomorrow, then explain the delayed-gradient trick again a week later without opening this lesson.
