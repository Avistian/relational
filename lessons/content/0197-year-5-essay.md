<p class="eyebrow">Year 5 · Q4 · Lesson 197 · Synthesis and writing</p>

# Draw the landscape. Defend one claim.

[Lesson 195](0195-thesis-stress-test.html) challenged the undervaluation thesis. [Lesson 196](0196-community-engagement.html) turned one implementation finding into an answerable question. Now turn the year's evidence into an argument a skeptical reader can inspect.

**Your win:** write a foundation-model landscape essay with one defensible conclusion and a concrete condition that would change your mind. Read the core in 15 minutes, then spend 30–45 minutes drafting. This is rehearsal for choosing a Year 6 research direction, not a completed Year 5 exit exam.

[[STATUS]]







## 2 · Three strategies, three different questions

> **In plain terms.** Ask where relational information enters, where the model learned its prior experience, and what must be trained for a new database. These are different questions.

A **foundation model** reuses knowledge learned across tasks. **Pretraining** learns reusable parameters before the target task. **In-context learning (ICL)** conditions a prediction on labeled examples without updating those parameters. The labeled examples are the **support**; the unlabeled example is the **query**. A frozen model can still consume target labels through its support set.

A **foreign key** points from one table row to another. A **temporal cutoff** specifies when a prediction is made; information must be available by that time. **Deep Feature Synthesis (DFS)** constructs columns by aggregating linked records, such as a customer's historical order count. A flat feature table can therefore contain relational information.

The planned L170b synthesis is not yet a standalone lesson. The following map supplies its prerequisite here. Its three strategies overlap: a graph-native model could also use synthetic pretraining.

[[FIGURE]]

<div id="landscape-map"></div>
<noscript><p>The figure and matrix contain the complete three-strategy overview; the interactive explorer adds the same query trace one strategy at a time.</p></noscript>

| Strategy | Where relational information enters | What is learned beforehand | New-task adaptation | Main question to test |
|---|---|---|---|---|
| Graph-native learning | Row encoders and messages/attention along database links | Shared representations from pretraining tasks | Model-dependent: fine-tuning or labeled context | Do learned interactions improve on equally informed summaries? |
| Synthetic relational prior | Generate relational tasks, then linearize with DFS for RDB-PFN | A predictor trained on generated tasks | Labeled context with frozen weights | Does that prior transfer beyond its generated assumptions? |
| Reuse a tabular foundation model | Relational aggregation produces a feature table for RDBLearn | The existing tabular backend was already pretrained | Context at inference; no new relational pretraining | Are these summaries sufficient under the same information budget? |

**Graph-native does not imply ICL.** Griffin uses cell encoding, graph message passing and task decoders, followed by target-task fine-tuning in the selected protocol. [Griffin §§3–4](https://arxiv.org/html/2505.05568v1). KumoRFM-2 instead offers relational ICL and separate fine-tuning experiments; its attention spans cells, linked rows and examples. [KumoRFM-2 §§3–4](https://arxiv.org/html/2604.12596v1).

**A relational prior need not mean graph-native inference.** RDB-PFN trains on synthetic relational tasks and consumes DFS-linearized inputs. [RDB-PFN §5](https://arxiv.org/html/2603.03805v5). RDBLearn combines aggregation with an existing tabular model. “Training-free” here does not erase that backend's pretraining, feature construction or inference costs. [RDBLearn §3](https://arxiv.org/html/2602.18495v1).

**Worked query trace.** Predict whether customer Ada will churn after day 30. Her orders before the cutoff are known; her future churn label is hidden. A graph model combines Ada's row with eligible order rows. A DFS pipeline may replace the orders with count and sum columns, then pass Ada and labeled support rows to a predictor. RDB-PFN and RDBLearn can share that feature representation while differing in the prior learned before this task. If the summaries discard order sequence, they cannot recover that sequence merely by using a larger predictor. This does not prove sequence matters for this target.

**Write a prediction:** which comparison would isolate the value of relational information? Hold the predictor fixed and compare target-only features with legal relational summaries. Comparing two systems that both use related rows answers a different question.

## 3 · “Open versus proprietary” needs a comparison contract

**Openness has several axes:** implementation code, model weights, training data, license, and enough protocol detail to rerun the experiment. An API provides access to predictions. An open repository provides access to code. Neither alone establishes that you can reproduce a historical model.

For this essay, freeze the L191 comparison: **KumoRFM-2 v1, Tables 3, 4, 7 and 8**. The bounded open-code foundation pool is RDBLearn and Griffin; the supervised pool is LightGBM, GraphSAGE, RelGNN and RelGT, where present. These are inspected comparison pools, not a census of every current method. [L191 protocol](../labs/l191-reproduction.md).

<div id="gap-predict"></div>
<noscript><p>Predict: does including supervised open methods shrink the Table 3 gap? Yes: 4.0533 becomes 1.5417 AUROC percentage points.</p></noscript>

**Worked example.** On Table 3's 12 tasks, Kumo's mean published AUROC exceeds the best eligible single foundation method by **4.0533 percentage points**. Include supervised methods and that gap becomes **1.5417 points**, against RelGNN. The Kumo scores did not change; the eligible comparison changed. [Complete reconstructed report](../labs/evidence/l197/report.json).

**AUROC** measures ranking of positive examples above negative ones; higher is better. Here its table uses a 0–100 scale. **MAE** averages absolute prediction error; lower is better. Regression tables divide each method's error by LightGBM's error on that same task before averaging. These ratios are dimensionless. Do not average raw errors from unrelated targets or mix them with AUROC.

[[GAPS]]

A **taskwise oracle** selects the best published test result separately for each task. It is an optimistic retrospective envelope, not one deployable method. A single-method winner selected from these test tables is also descriptive; prospective selection must use validation data.

> **Scope check.** All four tables are reconstructed in full: 401 task cells plus 90 published summary cells. Five aggregates fall outside the derived rounding bounds; 34 displayed mean ranks differ from the published ranks. Preserve these differences, rather than quietly substituting corrected-looking numbers. The audit uses displayed precision and cannot recover hidden precision. No eligible foundation comparator is present in Tables 4 or 8. That means missing comparison, not an infinite advantage. [Independent checks](../labs/_verify_l197_results.json).

## 4 · Let the evidence change the sentence

The complete L195 replay retains all five L149 seeds, all thirty L182 saved runs and the full 21-task L194 inventory. Keys include both entity and cutoff. Regression is scored in float64 against common saved targets, while retaining the original GNN float32-target discrepancy. This is a new execution of old evidence, not an independent experimental replication. [Frozen protocol](../labs/l197-reproduction.md).

[[REPLAY]]

**Read the uncertainty correctly.** The regression interval resamples whole drivers, keeping their rows together. It uses 2,000 original draws with seed 137 and holds trained models and the split fixed. It does not cover all shared race/time dependencies or generalization to new databases. An interval crossing zero establishes neither superiority nor equivalence. The ten ICL support draws reuse one test population; they are not ten independent databases.

**Missing scores are still missing.** RDBLearn's 21 published comparisons have 17 favorable, 3 unfavorable and 1 equal reference signs against AutoGluon+DFS. Every fresh score remains null behind `INCOMPLETE_SOURCE_PREPROCESSING_GATE`. The preprocessing issue is an obstacle to reproduction; it is not a measured model defeat. [L194 analysis](0194-open-fm-analysis-report.html).

<div id="claim-audit"></div>
<noscript><p>Published tables support scoped published comparisons. Saved predictions support scoped replayed pipeline comparisons. Neither establishes fresh model reproduction, an architectural cause, or economic undervaluation.</p></noscript>

## 5 · Build one paragraph from five parts

> **In plain terms.** Give the reader a claim, the evidence behind it, and a reason the evidence bears on that claim. Then say what it cannot decide and what would change your position.

A **warrant** is the reasoning that connects evidence to a claim. A **revision condition** is an observable result that would make you narrow, strengthen or abandon the claim.

| Part | Worked sentence |
|---|---|
| Claim | This saved F1 comparison does not establish a practical advantage for the basic GNN. |
| Evidence | Across five seeds, its test MAE is 4.1231 versus 3.9489 for relational FE; the driver interval for its advantage is −0.4496 to +0.0859. |
| Warrant | The point estimate favors FE, and the conditional interval includes both signs. |
| Limitation | Both pipelines use relational information; the comparison does not isolate the value of that information or test every graph architecture. |
| Revision condition | A preregistered study on untouched tasks with matched information, search budgets and a justified practical margin could change this conclusion. |

**Repair the overclaim.** “The graph model lost, so relational learning has no value.” Replace it with the worked claim above. The first sentence changes the subject from a particular pipeline to all uses of relational data.

**The undervaluation thesis has four steps.** Useful relational signal; value from learning how to use it; robust benefit under fair comparisons; and economic value underrecognized by adopters or investors. Predictive scores alone do not measure engineering cost, business utility or market expectations. Preserve the dossier's C1–C4 identities when discussing these steps. [Thesis stress-test reference](../reference/thesis-stress-test.html).

<div id="essay-structure"></div>
<noscript><p>Write five fields: claim, evidence, warrant, limitation, revision. Filling all fields makes a draft ready for review; it does not grade its truth or establish mastery.</p></noscript>

## 6 · Your essay: one position, three strategies, a fair objection

Write **800–1,200 words**. Treat this as a target, not an automatic quality score. First draw the map from memory; then consult sources. Organize the essay into five short sections:

1. **Position:** state what remains plausible and what remains unestablished about the thesis.
2. **Landscape:** trace the three strategies for the same query. Explain their overlap and separate prior, representation and adaptation.
3. **Open/proprietary gap:** declare date/version, eligible pool, tasks and metric. Use a measured example and discuss reproducibility access separately from reported performance.
4. **Strongest objection and response:** use both favorable and unfavorable evidence, retain missing results, and limit your conclusion.
5. **Revision condition:** propose a fair comparison that could change your mind. Include untouched tasks, information access, validation-only selection, a practical margin and total cost.

[[RUBRIC]]

**Pass standard:** a human review finds every dimension defensible, with no invented fresh scores, causal claims from uncontrolled comparisons, or equivalence claims from an interval crossing zero. Word count and field presence cannot certify this. Your current state stays `PENDING_WRITTEN_DEFENSE` until your own argument is reviewed.

<div id="teachback"></div>
<noscript><p>Teach back: why can an API-only model lead a published table while a complete local model reproduction remains unfinished? Distinguish source-reported performance, access and local evidence.</p></noscript>

<details><summary>Read the author’s worked essay after drafting yours</summary>

[[ESSAY]]

</details>

## 7 · Run the audit and submit your defense

[Student notebook](../labs/0197-year-5-essay.ipynb) · [Executed solution](../labs/html/0197-year-5-essay.html) · [Solution notebook](../labs/solutions/0197-year-5-essay.ipynb) · [Portable audit ZIP](../labs/evidence/l197/reproducer.zip) · [Reference](../reference/year-5-essay.html).

Implement three live functions: preserve task coverage, admit only claims supported by a declared evidence lane, and check the five argument fields without pretending to grade prose. The provided source then reconstructs every selected table and rescored prediction. Wrong implementations must fail the checks; each function affects your report or submission.

For a local run from the repository root:

```bash
.venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_audit_l197.py
.venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_verify_l197.py
```

The standalone ZIP has its own instructions. The notebook embeds all frozen inputs and displays the load-bearing code. No model weights, API calls or new training are used. **The author run has a $0 cloud/API budget and an aggregate 1,800-second local execution limit.** [Budget ledger](../labs/evidence/l197/local-budget.json).

**Primary reading:** [KumoRFM-2 v1, architecture and Tables 3/4/7/8](https://arxiv.org/html/2604.12596v1). Read one table caption and the context policy before using its gap. Use [Griffin](https://arxiv.org/html/2505.05568v1), [RDB-PFN](https://arxiv.org/html/2603.03805v5) and [RDBLearn](https://arxiv.org/html/2602.18495v1) to verify the map's distinctions.

**Bring your draft back to the agent.** Ask about any unclear mechanism or inference; paste one paragraph for a claim-by-claim critique. In two days, redraw the three routes and reproduce one five-part argument without notes. Next, [Lesson 198](0198-three-research-directions.html) turns the unresolved evidence into three ranked research proposals. This lesson prepares that decision; it does not choose your direction for you.
