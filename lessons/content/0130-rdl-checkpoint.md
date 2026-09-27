<!-- sequence-review:start -->
<aside class="sequence-context" aria-label="Where this lesson fits">
<p class="sequence-eyebrow">From Lesson 129 to this lesson</p>
<p>The baseline and competitor are available. The checkpoint tests ownership of the complete chain from a database query to an independently verified result.</p>
<details><summary>Quick prerequisite reminder</summary><p>MAE is the mean absolute prediction error, in target units. A prediction packet must preserve entity and cutoff keys so independently saved outputs can be aligned.</p></details>
</aside>
<!-- sequence-review:end -->

> **The win.** Produce a defensible experiment packet for one complete RelBench task. You must explain how a database becomes a prediction, run the pipeline, and show why its reported score belongs to the intended queries.

[Student lab](../labs/0130-rdl-checkpoint.ipynb) · [Executed solution](../labs/html/0130-rdl-checkpoint.html) · [Checkpoint reference](../reference/rdl-checkpoint.html) · [Exact commands and protocol](../labs/l130-reproduction.md) · [Written defense template](../labs/l130-defense.md)

## 1 · The checkpoint joins the quarter together

[Lesson 122](0122-reg-construction.html) made the database into a graph. [Lesson 123](0123-temporal-heterogeneous-graphs.html) restricted context by time. [Lesson 124](0124-entity-task-tables.html) separated prediction queries from database rows. [Lesson 125](0125-pytorch-frame-deep-dive.html) encoded typed columns. [Lesson 127](0127-relbench-v1.html) established the benchmark protocol, [Lesson 128](0128-task-taxonomy.html) matched the prediction head to the target, and [Lesson 129](0129-manual-feature-engineering.html) built the relational-feature competitor.

The missing skill is ownership of the chain. A passing training script is insufficient if you cannot identify the query, reconstruct its permitted information, explain its output, and independently verify the experiment report. This is the Q1 deliverable from the [year-four plan](../plan/year-4.md), and prepares the instrumented forward pass of Lesson 131.

[[WARMUP]]

**Recall before opening code.** Why can one driver have several labels? Which rows receive loss? Which split chooses the checkpoint? Why does a future scheduled race differ from a future race result?

Read [Fey et al., RDL blueprint](https://proceedings.mlr.press/v235/fey24a.html) for the database-to-graph-to-prediction argument. Read [Robinson et al., RelBench v1, §3, Table 7 and Appendix B](https://arxiv.org/html/2407.20060v1) for this experiment. The former motivates the computation; the latter supplies the named numerical target.

**Suggested route.** First explain the task and trace one query without running code. Then implement the three checkpoint functions, run their checks and replay the author packet. Finally run the explicit full-training gate in the pinned environment and submit the written defense. The short replay and the longer fresh-training lane have separate completion records.

## 2 · Freeze the question before building the graph

> **In plain terms.** A prediction is a question asked about a driver at a particular time. It is not a permanent property stored on that driver's node.

A **query** is the pair `(driverId, cutoff)`. The **cutoff** is the time when a prediction is requested. The target is the driver's mean finishing position during the next 60 days, with outcomes in **(cutoff, cutoff + 60 days]**. Lower mean absolute error, or **MAE**, is better.

**Worked example.** Driver 10 at 2004-07-05 has an outcome window ending 2004-09-03. The same driver queried at another cutoff has another permitted history and another target. Store both key components at full timestamp precision; collapsing dates or grouping only by driver can silently combine different questions.

| Contract | Frozen value |
|---|---|
| Database / task | `rel-f1` / `driver-position` |
| Complete query splits | 7,453 train / 499 validation / 760 test |
| Database through the test cap | 9 tables, 74,063 rows, 338,842 directed edges |
| Task loss and score | Training mean L1; evaluation MAE in finishing-position units |
| Fits | Five fresh seeds 0–4; ten full epochs per seed |
| Selection | First strictly improved validation MAE |
| Numerical comparison | Table 7 validation 3.193 / test 4.022; absolute mean tolerance 0.2 |

**Freeze** means deciding these values before inspecting the new test results. Hashes identify exact source and archive bytes. A changed setting creates another experiment; it does not silently replace this one.

> **Scope check.** The released cohort uses future participation when defining eligible queries. Reproducing that cohort is not proving that the task can be deployed without knowing future participation. Event timestamps are available; real ingestion times and historical versions of mutable/static features are not. Keep these limits in the packet.

**TODO 1 — query identity.** Implement `validate_queries`: require the complete expected row count, unique `(entity, time)` keys and finite targets. Return keys in the task's order. The check deliberately repeats a driver at two times and rejects duplicate questions.

## 3 · Trace a real query through the model

A **relational entity graph** represents each table row as a typed node. A foreign key creates a directed link to its parent row; a reverse edge lets information flow back. Node types retain table identity. The graph is shared infrastructure, while each query specifies its own root and cutoff.

**Sampling.** Start at the driver root and gather incoming neighbors for two hops, with maximum per-relation fanouts `[128, 64]`. A fanout is a neighbor-count limit. The released loader samples uniformly among eligible neighbors. It constructs disjoint contexts for different query roots, so the same database row may occur more than once in a batch.

**Time.** Every dated row in a sampled context must satisfy `row_time <= root_cutoff`. The cutoff stays attached to the root at every hop. Labels describe later outcomes and are attached to supervised queries separately; they are not a contextual node feature.

[[FIG:trace]]

The figure uses the real driver-10 query and its **exhaustive** two-hop context: 99 node occurrences and 245 directed edges. These are structural counts, not recorded hidden activations. Bounded stochastic training neighborhoods can differ. The notebook reconstructs this example from the complete archived graph and checks the original row and edge identities.

### Model architecture · keep context count separate from output count

A **TensorFrame** stores typed columns for sampled rows. A separate four-block residual encoder for each table converts its numerical, categorical, timestamp and text features into 128-dimensional row vectors. A residual block combines a learned update with its input. It allows deeper transformations while retaining a direct path for the previous representation.

The relative-time encoder represents the age of a row with respect to its query cutoff. This time vector is added to that row's encoded features. Rows without recorded time do not gain a fabricated timestamp.

[[FIG:architecture]]

**Diagram trace.** Keep context count separate from query count. Which of the sampled rows become supervised root outputs? On a narrow screen, scroll the figure sideways.

**Message passing.** For each directed relation, GraphSAGE sums transformed information from neighbors. Relation-specific outputs are summed for each destination type, then normalized and passed through a rectified linear unit (ReLU), which replaces negative coordinates with zero. Repeat for two layers. The **head** maps the first B driver vectors to B scalar predictions; B is the number of supervised queries, not the number of context nodes.

[[MODEL_CODE]]

**Predict the shapes.** If one query gathers 99 nodes in total, should the loss contain 99 terms or one? Explain why a context row can influence a prediction without receiving its own direct supervised loss.

The code and diagram are inherited computation made explicit for this checkpoint. The notebook includes the complete model, graph constructor, trainer and underlying library primitives. In Lesson 131 you will instrument the actual intermediate activations rather than infer their shapes from the graph.

## 4 · Train, choose, and freeze

An **epoch** visits every training query once. Each mini-batch predicts only its root queries. Mean L1 loss averages `abs(prediction - target)` over those roots. Backpropagation computes gradients; Adam uses them to update the row encoders, temporal encoder, message-passing layers and prediction head together.

The released configuration uses Adam with learning rate 0.005 and batches of 512 queries. Every epoch must account for all 7,453 training queries. The final short batch still counts; a loop that silently skips it has changed the experiment.

[[TRAIN_CODE]]

Validation is evaluated after each epoch. A **checkpoint** is a saved set of model parameters. Keep the first checkpoint whose validation MAE strictly improves on previous epochs. Equal scores retain the earlier checkpoint. The test score does not participate in this choice.

[[FIG:selection]]

**Predict before interacting.** Hold the validation trace fixed. Can changing the reported test MAE legitimately change the selected checkpoint?

[[CHECKPOINT_WIDGET]]

At inference, the released pipeline clips predictions to the training-label 2nd and 98th percentiles. Clipping limits extreme predictions; its bounds come from training labels. The selected model is then evaluated on all 499 validation and 760 test queries using the official task evaluator.

> **Scope check.** A fresh sampled validation pass can differ from the validation value that selected the checkpoint. Preserve both values. Released feature statistics are fitted on the database through the test cap, rather than exclusively on training-era rows. Reproduce and disclose that rule; changing it is a separately named intervention.

## 5 · A plausible score can belong to the wrong rows

> **In plain terms.** Before subtracting predictions from labels, prove that each prediction answers the same question as its label.

**Worked example.** Driver 7 has target 2 at time 10 and target 5 at time 20. Its predictions are 2 and 4 respectively. The absolute errors are 0 and 1, so MAE is `(0 + 1) / 2 = 0.5`. Reverse only the prediction-row order. Positional subtraction now gives `(2 + 3) / 2 = 2.5`, even though neither prediction changed.

[[FIG:alignment]]

Use a one-to-one join on both entity and cutoff. Reject missing, extra or duplicate keys **before** averaging. Never silently take an inner join: dropping difficult unmatched queries can make the remaining score look better.

[[KEYED_WIDGET]]

**TODO 2 — keyed MAE.** Implement `keyed_mae`. Validate the task and prediction keys, require exact coverage, align predictions and compute the mean absolute error. The full fresh worker deliberately reverses prediction records before calling the canonical function; the notebook calls your implementation on the same complete author evidence.

The official evaluator and independent scorer answer complementary questions. The official evaluator defines the metric. The independent scorer checks both arithmetic and identity. An archive audit separately confirms that the stored labels, IDs and times equal the released task table.

## 6 · Decide what five runs establish

One successful fit can be lucky or incomplete. The selected experiment requires all five expected seeds. A seed controls the random initialization and stochastic sampling. A run UUID identifies an execution: two different seed labels attached to the same execution are not two fresh runs.

**TODO 3 — reproduction verdict.** Implement `reproduction_verdict`. Reject missing, unexpected or duplicate seeds, repeated run identities, incomplete epochs, nonfinite scores and negative errors. Then compute the mean and **sample standard deviation**, which divides squared deviations by `n - 1`. Compare each split's mean with its predeclared paper target.

`CLOSE` means `abs(measured_mean - paper_mean) <= 0.2`. `OUTSIDE_TOLERANCE` is an equally valid reported outcome. Neither status grants permission to change the tolerance after seeing the result. Seed standard deviation describes variation between fits; it is not a confidence interval and does not account for new databases or alternate time splits.

**Predict before reading.** Would complete original-model output agreement plus a close mean establish the exact historical training environment? Name one missing piece of evidence.

[[RESULTS]]

[[FIG:scores]]

The measured packet records source and archive hashes, all epoch traces, checkpoint hashes, fresh execution identities, original-model output checks and full-query predictions. Every actual batch also checks node identity, edge identity and its own root's temporal cutoff.

> **Scope check.** This is a complete selected released-protocol replay. The exact historical training commit, package/random-state identity and whole-paper reproduction are **NOT_ESTABLISHED**. The other 29 RelBench tasks are **NOT_RUN** here. Live Colab and deployment remain **NOT_CHECKED**. Author execution leaves learner mastery **PENDING_WRITTEN_DEFENSE**.

## 7 · Compare with the manual-feature competitor carefully

Lesson 129 gives a stronger relational baseline than fitting trees to a driver's raw row. Its SQL aggregates relational histories into features, then tunes LightGBM. Before comparing, verify identical query keys and targets. Both methods should answer the same question even though their information processing and training budgets differ.

[[COMPARISON]]

This comparison reuses the explicitly identified Lesson 129 runs; it does not call them fresh L130 fits. Basic RDL and tuned manual features have different preprocessing scopes and compute budgets. Figure 3's human-study comparison can also use a boosted regression head, whereas Table 7's basic RDL is the present target. Matching task keys does not erase those distinctions.

**Interpretation task.** Explain what the observed difference says about this task. Then explain why it cannot establish that RDL universally beats manual feature engineering, or reproduce the human-effort saving of the original study. A close or unfavorable result belongs in the research record too.

## 8 · Diagnose before repairing

For each fault, first predict the artifact that should expose it. Then run the indicated check. Repair the cause and rerun; do not merely delete the check.

| Fault | First evidence to inspect | Why the printed MAE may mislead |
|---|---|---|
| Join predictions by driver only | Full `(entity, time)` key coverage | Repeated drivers answer different questions |
| Choose the best test epoch | Epoch-selection trace | Test information influenced model selection |
| Train on context labels | Root count and target attachment | Context rows are not independent supervised queries |
| Permit a future neighbor | Per-root node-time audit | The model saw information unavailable at its cutoff |
| Average four completed seeds | Exact expected seed set | A failed or unfavorable run may be missing |
| Rename an old run as fresh | UUID, dispatch time and source hashes | A copied result does not demonstrate new execution |

The behavioral tests also reject deliberately corrupted implementations. This tests whether a guard can detect a plausible mistake, rather than merely accepting the reference solution.

## 9 · Submit a reproducible packet and defend it

Use the [defense template](../labs/l130-defense.md). Attach the task/source/environment hashes, configuration, full traces, selected checkpoint hashes, query-keyed predictions and per-seed summary. Include the exact commands and the deviation ledger so another person can regenerate the result.

**EXIT rubric — five required defenses.**

1. **Question:** state the query key, target window, eligibility rule, split sizes and the boundary of availability evidence.
2. **Computation:** trace the real query from rows and foreign keys through typed encoders, two message-passing layers and one output. Distinguish context from supervision.
3. **Decisions:** explain what trains parameters, what selects the checkpoint and when test labels enter evaluation. Account for every epoch and query.
4. **Evidence:** show how source parity, temporal audits, keyed metrics and complete fresh seeds support different claims. Interpret the predeclared tolerance honestly.
5. **Research judgment:** compare manual FE with basic RDL on this task, preserve protocol differences, and identify what another experiment would need to resolve.

All five need a correct explanation supported by your artifacts. A benchmark score alone is not a pass. A well-diagnosed result outside tolerance is more useful than an unexplained close score. Your agent will assess the written defense; automated preparation does not mark this checkpoint mastered.

[[TEACHBACK]]

Ask follow-up questions about any query, intermediate tensor, failed check or evidence claim. After a delay, explain the chain again without the diagram; that retrieval tests whether you can reconstruct it rather than recognize it.

<!-- sequence-next:start -->
**Carry this forward.** Instrument a real forward and backward pass to explain what is learned. [Continue to Lesson 131](0131-gnn-tabular-stack.html).
<!-- sequence-next:end -->
