## Start cold · choose before reading

<div id="warmup"></div>

A fraud table has 6,000 labeled transactions, merchant IDs, a text description and timestamps. Predictions must serve next month's traffic within 5 ms per request. Write down your first baseline, validation split, strongest challenger and one result that would change your mind. **Commit to four sentences before reading on.** A model name without a split is not yet a decision.

**Today's win:** write a one-page decision guide that another researcher can execute and challenge. Spend about 20 minutes on sections 1–4, then 20 minutes on the writing task. The optional audit notebook rebuilds every displayed rank from the complete corrected Lesson 60 predictions. It does not train new models.

## 1 · The missing step between knowing models and choosing one

[Lesson 77](0077-single-table-ceiling.html) showed an information ceiling: a better learner cannot recover information discarded by its input. [Lesson 78](0078-message-passing-preview.html) supplied a mechanism for bringing in neighboring information. Neither result says that a GNN should be your first baseline. You still need to establish that the extra information is available at prediction time, useful beyond simple aggregates, and worth its cost.

The same logic applies within single-table learning. Choose in this order: **available information → deployment-matching split → feasible candidates → validation comparison → locked test evaluation**. A *regime* is the combination of task, sample size, feature types, availability, shift and resource constraints. A *shortlist* is a set of hypotheses to test, not a prediction of a universal winner.

For the fraud example, define one prediction per transaction at arrival. Build merchant aggregates only from events and labels already available then. Use earlier periods for training and later periods for validation/test, allowing the fraud-label delay. If production includes unseen merchants, report that subgroup separately; a group split is useful for that question but does not replace chronological evaluation. Measure p95 latency on the intended hardware, batch size, context size and complete preprocessing path.

<!--figure:decision-->

## 2 · A decision table, with reasons and falsifiers

This is a guide to the **historical model versions studied in Year 2**, not a September 2026 leaderboard. Pin the exact implementation and checkpoint before testing. Row-count thresholds are feasibility prompts, not scientific phase transitions. The paper sources support the mechanisms; the candidate order below is our operational judgment.

| Regime | Start and challenge | Why this is a useful test | What would change the decision? |
|---|---|---|---|
| Ordinary mixed numeric/categorical table | Tuned GBDT; challenge with RealMLP and TabM | Tree partitions are a strong baseline for heterogeneous features; strong MLP recipes and efficient ensembles deserve a fair comparison | A neural candidate improves the prespecified validation metric within the serving budget |
| Few labels; checkpoint fits task and memory | GBDT plus historical TabPFN v2; TabICL 2025 for supported classification | In-context inference reuses a learned prior without fitting fresh weights for each task | Chronological validation, calibration or end-to-end inference cost removes the apparent advantage |
| More rows or larger context | Trees/TabM; feasibility-test the pinned TabICL classification model | Full-context ICL and gradient-trained predictors incur different memory and prediction costs | Measured batching/context cost or validation performance changes the feasible set |
| A plausible feature-interaction hypothesis | Keep trees/MLPs; add FT-Transformer | Feature-token attention offers a distinct mixing mechanism | Its gain disappears under equal selection budget or a simpler model matches it |
| High-cardinality categories or unseen IDs | CatBoost with its native categorical route; compare a training-fitted vocabulary/embedding route | Identity reuse and unseen-category behavior matter more than a nominal model family | Group/cold-start evaluation exposes memorization; larger embeddings do not rescue it |
| Informative names or free text | Compare no-text, simple text features and an explicitly pinned semantic encoder; consider CARTE for suitable cross-table transfer | Text representation changes the information interface and the cost | A frozen text ablation shows no value, or text was unavailable at the cutoff |
| Complementary validation errors; enough serving budget | Out-of-fold blend/stack of retained families | Diversity may improve predictions; every selection step must stay out of held-out folds | Honest OOF comparison and a locked holdout fail to beat the best feasible single model |
| Relevant history or neighbors absent from the row | First add point-in-time aggregates; then compare a relational model | Test access to additional information before crediting a graph architecture | Matched aggregates erase the gain, or any neighbor path leaks future information |

Read the mechanism evidence: [trees, §5](https://arxiv.org/abs/2207.08815), [RealMLP](https://arxiv.org/abs/2407.04491), [TabM](https://arxiv.org/abs/2410.24210v3), [FT-Transformer](https://arxiv.org/abs/2106.11959), [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), and [TabICL 2025](https://arxiv.org/abs/2502.05564v2). The original TabICL paper is a classification study: do not silently apply later regression support to that historical checkpoint.

**Mechanisms are not slogans.** RealMLP is an MLP plus a carefully designed training recipe. TabM trains multiple predictions using shared parameters and averages them for inference. FT-Transformer mixes feature tokens through attention. TabPFN/TabICL use labeled training rows as inference context; “no task-specific gradient fitting” does not mean “no prediction cost.” An external ensemble combines separately obtained predictions and adds their serving requirements. Explain the particular mechanism that makes your challenger worth measuring.

## 3 · Evidence cannot travel farther than its protocol

The corrected [Lesson 60 comparison](0060-broad-model-comparison.html) has 210 selected runs: 14 dataset/regime cells × five methods × three initialization seeds. It uses small training caps, two candidates per method and compressed neural training. Its RealMLP-TD-S and TabM-mini arms are reduced recipes. CatBoost receives common one-hot inputs. FT-Transformer, TabPFN and TabICL are **absent**. A blank cell for these families means **not measured**, not “worse.”

Reconstruct each saved loss from targets and predictions, verify split IDs and validation-only candidate selection, then average the three seed errors within each dataset. Rank the five means within that dataset; average those ranks with equal dataset weight. Low rank is good. Never pool log loss with RMSE or treat three seeds as three new datasets.

**Worked tie.** Dataset A has seed-mean errors `[.20,.20,.50]`, giving ranks `[1.5,1.5,3]`. Dataset B has `[20,10,30]`, giving `[2,1,3]`. Their equal-weight mean ranks are `[1.75,1.25,3]`. The large numerical scale of B does not give it more votes. Ranking loses effect-size information, so inspect within-dataset losses too.

<!--results-->

<!--figure:ranks-->

<div id="cohort-viz"></div>

The random panel contains eleven datasets; the temporal panel contains three. Restrict random results to those same three names before interpreting the difference. Even then, these are separately sampled train/test populations, not an isolated intervention on chronology. A matched rank reversal is a reason to test the deployment split, not a causal estimate of time's effect or a universal ordering.

The notebook preserves per-dataset seed values, sample SD and ranks. Its inherited Friedman/Nemenyi summaries are exploratory, especially with only three temporal blocks; their presence does not establish equivalence or significance for a hand-picked pair. Seed variation is conditional on fixed rows. Repeating seeds cannot establish uncertainty over new business domains.

**Reproduction boundary.** We freshly audit all 210 frozen prediction records here. No predictive models are retrained in this unit; no paper benchmark is newly reproduced. The [reproduction contract](../labs/l079-reproduction.md) gives exact analysis commands, the full local training rerun command, and links to the original experiment contracts. A deterministic reanalysis is a reproducible artifact, not a replacement for full paper training.

## 4 · Make the recommendation executable

Here is a **synthetic decision example**, not measured model performance. The validation log losses are Trees .32, TabM .29 and ICL .27. Their measured-in-the-example p95 latencies are 2, 8 and 35 ms. All other constraints are assumed satisfied. Predict the choice under a 5 ms budget before moving the control.

<div id="budget-viz"></div>

<!--figure:budget-->

At 5 ms only Trees qualifies. At 10 ms TabM becomes the lowest-loss feasible candidate; at 40 ms ICL does. Under 2 ms none qualifies. The decision rule is: filter by the declared constraint, then minimize validation loss, with a prespecified tie rule. Real latency is a measured distribution, so include a margin and repeat measurements near a threshold. The synthetic fixture uses exact point estimates to expose the logic.

The test set has no role in that rule. Once a family, preprocessing recipe and ensemble are chosen, freeze them before the final evaluation. If you inspect test results and redesign the shortlist, the old test has become development evidence; acquire a new holdout or report the adaptive nature of the evaluation. Revisit [validation overfitting](0059-validation-set-overfitting.html) and [OOF ensembling](0057-cross-family-ensembling.html).

## 5 · Write the one-page artifact

Download the [blank decision guide](../labs/l079-decision-template.md). Fill six to eight rows, at most about 650 words. Each row must name a regime, baseline, challenger, split/metric, resource limit, evidence source and a result that would change your mind. Include all Year 2 families across the table. Add a short worked recommendation for the opening fraud case, and a reproduction footer recording source revisions, data/split identities, seeds, environment and unrun work.

**CHECK yourself on an unfamiliar case.** A 120,000-row regression dataset has repeated customers and daily targets. Someone recommends “TabICL because it handles large tables.” Identify three missing premises before accepting the claim.

<details><summary>Feedback · reveal after answering</summary><p>The historical 2025 TabICL paper studies classification; task support must match the pinned checkpoint. Daily targets and repeated customers require a deployment-specific temporal/entity protocol. Actual memory, context size and serving cost must be measured. Also ask whether past targets and text are available at the cutoff. Row count alone cannot settle the choice.</p></details>

**Rubric: 12 points.** Award 0–2 each for deployment/split precision, mechanism-based shortlist, evidence with scope, feasible costs, falsifiable change criteria, and a complete reproduction footer. Two means concrete enough to execute; one means partially specified; zero means missing. A passing rehearsal is at least 10/12, with no test-based selection, future-information leakage or invented paper-reproduction claim. This is self-assessment until your written artifact is reviewed.

The optional notebook has two live TODOs: constrained validation selection and matching dataset populations. Its CHECK cells test your definitions; the audit runs those same functions. Submit the guide, the audit output if attempted, and your four initial sentences. A [worked guide](../solutions/l079-example-guide.md) is available **after your own draft**; different defensible shortlists are welcome.

<div id="teachback"></div>

Tomorrow, reconstruct the choice order and one counterexample without reopening the table. Before Lesson 80, explain why a model can have the best benchmark rank yet be excluded from your deployment. Ask the agent follow-up questions and bring your draft for a strict evidence review. Creating this lesson does not mark the exit exam—or this essay—complete.

**Primary reading:** [TabM, abstract and comparative experiments](https://arxiv.org/abs/2410.24210v3). Use its distinction between architectural complexity and measured performance-efficiency to challenge one row of your guide. Then read the primary paper linked in the row you find least certain.
