<p class="route-lead">A tabular foundation model can predict from relationally constructed features. Your job is to locate where the relationships enter—and what the predictor still cannot recover.</p>

**Today’s win:** build a defensible transfer map: what carries over, what needs new machinery, and which experiment supports each claim. Read and trace in **15 minutes**; allow **30–45 minutes** for the notebook and written defense.

[Student notebook](../labs/0167-tabular-to-relational-fm-transfer.ipynb) · [Executed solution](../labs/html/0167-tabular-to-relational-fm-transfer.html) · [Quick reference](../reference/tabular-relational-transfer.html) · [Reproduction contract](../labs/l167-reproduction.md)

## 1 · The gap left by Lesson 166

[Lesson 166](0166-rdb-pfn-synthetic-relational-priors.html) showed how synthetic relational data can shape a pretrained predictor. But its comparator, TabICL, also received the same relationally constructed matrix. If both methods see database information, what exactly makes their learning different?

Retrieve [TabPFN v2](0064-tabpfn-v2.html) and [TabICL](0066-tabicl-column-row-attention.html): **support** means labeled examples supplied during inference; a **query** is a row whose target must remain hidden. In-context learning changes these inputs while the pretrained weights stay fixed. Pretraining previously changed the weights across many tasks. The distinction is explained in the [TabICL paper](https://proceedings.mlr.press/v267/qu25d.html); Molnar’s [introduction and reading map](https://tabularfoundationmodels.com/introduction) provide a gentle recap.

[[WARMUP]]

Use three separate questions: **What information reaches the model? What prior shaped its weights? How does it adapt to this task?** A model name alone answers none of them completely. This directly serves our mission: before claiming that relational models unlock extra value, make the tabular baseline competitive and identify the actual source of any gain.

## 2 · Model architecture: trace the handoff

> **In plain terms.** Keep three boxes separate: prepare the database inputs, load the learned weights, and supply this task's examples. Two methods can receive exactly the same relational features yet have learned different prediction rules before this task.

[[FIG:architecture]]

The shared interface is `predict(X_support, y_support, X_query)` with frozen weights. The implementations differ. TabPFN v2 alternates attention across features and across examples. TabICL first builds row representations, then performs dataset-level ICL. The released RDB-PFN numeric model alternates feature attention and support-only row attention over a DFS matrix. A shared API does not make their internal operations or pretrained weights interchangeable. See [TabPFN’s architecture](https://www.nature.com/articles/s41586-024-08328-6), [TabICL](https://proceedings.mlr.press/v267/qu25d.html), and [RDB-PFN §§4–5](https://arxiv.org/html/2603.03805v5).

**DFS means Deep Feature Synthesis:** follow relationships and aggregate records into target-row features. It can supply relational information to a tabular FM. RDB-PFN additionally changes the synthetic pretraining distribution. Neither fact means its inference input is the original database graph. In contrast, [Griffin](0164-griffin-graph-centric-rdb-fm.html) makes graph structure part of the model’s computation. This distinction explains why “relational” can describe the data preparation, the pretraining prior, or the predictor architecture.

| Component | What carries over | What must be supplied or checked |
|---|---|---|
| Row/cell representation | Encoders can process a constructed numeric table | Keys define joins; arbitrary ID codes do not explain relationships |
| Context inference | Labeled support guides frozen-weight predictions | Every support label must be available at the query time |
| Feature construction | A tabular FM accepts engineered relational summaries | Join path, aggregation and owner cutoff must be explicit |
| Pretraining | Learn a reusable prediction procedure across tasks | A relational synthetic prior changes task distribution; same API does not imply same prior |
| Evaluation | AUROC compares binary rankings | Align complete entity/cutoff keys, label orientation and support draws |

**What cannot transfer automatically:** information discarded by the representation. Child histories `[2,8]` and `[5,5]` both become count 2, mean 5. Any predictor receiving only those two summaries sees identical inputs. It cannot distinguish their spread without another feature or representation. Better pretraining cannot reconstruct which history actually occurred.

[[PREDICT]]

## 3 · Worked trace: the attention mask is only half the contract

Suppose customer 7 has purchases `(day 2, value 4)`, `(day 8, value 8)`, `(day 10, value 100)`, `(day 14, value 12)`. For a prediction at day 10, a strict-past join includes days 2 and 8: count **2**, mean **6**. At day 20, all four are eligible: count **4**, mean **31**. One batch-wide cutoff would corrupt the earlier query.

Now consider three candidate support rows, queried on days 2, 5 and 10. Their labels become available on days 9, 12 and 10. At cutoff 10, this lesson’s strict-before policy allows only the first row. The second row’s features are old enough but its label is not. The third sits exactly on the boundary. In a real task, label availability comes from the target horizon and reporting process; it is not generally the task timestamp.

**Boundary convention changed deliberately.** Lesson 165 allowed a matured label arriving exactly at the current query cutoff (`arrival ≤ cutoff`), while still requiring an earlier context anchor. This fixture uses strict-before availability (`arrival < cutoff`). Neither convention should be inferred from a variable named “time”: write it into the task contract. A context anchored on day 5 whose label arrives on day 10 passes L165's arrival test at day 10 and fails this fixture's test. Its input graph still uses its own day-5 cutoff in both cases.

[[FIG:cutoff]]

[[EXPLORER]]

The explorer recomputes **features and eligible support identities**, not model probabilities. Its baseline is cutoff 10, final purchase value 12. Changing that later purchase must leave the baseline features unchanged. At cutoff 20 it should change the mean. This fixture assumes events are available immediately; it does not certify historical availability in RelBench.

[[CODE]]

The relational preparation enforces time eligibility; the predictor’s attention mask enforces which supplied rows can act as context. A correct mask still leaks if you feed it labels that were unavailable. The notebook makes you implement both eligibility functions, then tests future-value changes, repeated entities and exact-boundary cases.

## 4 · Reproduce the evidence before interpreting it

The named experiment is [RDB-PFN v5, Table 9](https://arxiv.org/html/2603.03805v5): **rel-f1 / driver-dnf, 512 support rows, seeds 0–9**, three fixed arms. All receive 72 released DFS features and the full 702-query test population. TabICLv1.1 uses **32 estimators**. TabPFN v2 is discussed above but is **not a measured arm here**.

Lesson 166 ran the 30 checkpoint evaluations. Lesson 167 independently verifies **41 input/source files**, regenerates all support draws, aligns `(driverId,date)` identities and recomputes all **21,060 predictions’** AUROCs using average ranks with half-credit ties. No new benchmark inference or cloud job runs in this lesson.

[[RESULTS]]

[[FIG:paired]]

The paired RDB-PFN minus single-table checkpoint difference is **+0.05795 AUROC**, positive in all ten draws. Against TabICL it is **+0.00437**, positive in six draws and negative in four. These are ten support draws on **one task**, with overlapping support and the same test population—not ten independent databases. Their variability does not establish cross-database generalization or a causal effect of the prior alone. The single-table checkpoint is the closer comparison, but this replay does not independently recreate its full training history.

**Keep the boundaries visible.** All three means match the paper at four decimals; the inherited descriptive tolerance is 0.02 AUROC, not a statistical equivalence test. Released labels complement the current raw DNF reconstruction documented in L166. Preserve that orientation when replaying: silently calling class 1 “DNF risk” would be wrong. Complement both labels and probabilities to preserve AUROC under the alternate orientation. This lesson audits saved evidence; it does not rerun the SQL reconstruction or original DFS. Historical identity remains `NOT_ESTABLISHED`; fresh pretraining and whole-paper reproduction remain `NOT_RUN`.

## 5 · Your transfer map

Open the notebook and complete three live functions: `temporal_summary`, `eligible_support`, and `keyed_auc`. The last one must reject missing or duplicate keys and align reordered predictions. The full 30-run audit calls your scorer, so a hard-coded toy answer cannot complete the lab.

Then fill the [transfer-map template](../labs/l167-transfer-map-template.md): for **representation, prior, adaptation, time and evaluation**, state what transfers, what must be added, a concrete counterexample, and the evidence boundary. Defend this conclusion in 300–500 words: “A strong relationally prepared tabular baseline tests whether a more specialized relational model adds value.” Do not claim the experiment proves the thesis across databases.

[[TEACHBACK]]

**Exit check:** explain the count/mean collision; exclude a support label whose window has not closed; reproduce the paired scores; distinguish checkpoint replay from pretraining. Author checks do not mark your learning complete: `PENDING_WRITTEN_DEFENSE`. Tomorrow, redraw the three interfaces without notes. Lesson 168 will ask whether the same reasoning survives an unseen database schema. Ask the agent to review your map or explain any unclear step.

**Primary reading:** [RDB-PFN v5 §§4–5 and Table 9](https://arxiv.org/html/2603.03805v5), comparing the input representation with [TabICL’s two-stage design](https://proceedings.mlr.press/v267/qu25d.html). [Full protocol and runnable fresh-inference lane](../labs/l167-reproduction.md).
