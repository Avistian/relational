<p class="eyebrow">Year 5 · Q4 · Lesson 191 · Read the comparison contract</p>

# What does “ahead of the best open model” mean?

[Lesson 190](0190-research-gap-checkpoint.html) turned saved evidence into a research question. Before choosing the baseline for that question, you need a trustworthy performance comparison. A large headline advantage can shrink when you change the eligible methods, the task coverage or the arithmetic.

**Your win:** build a dated, auditable KumoRFM-2 versus open-method gap table and defend exactly what it means. Read the core in about 15 minutes; use the notebook for implementation and the expandable tables for audit detail. This supports our mission: make a relational-learning research claim that a skeptical reader can check.

[[STATUS]]







## 2 · Model architecture: where the task enters

> **In plain terms.** The model first learns which cells matter for this task. It then combines related rows and labeled examples to predict a query row.

**Context** means examples whose targets are already known at the relevant cutoff. The **query** is the example whose target we want. A foreign key connects a row to a row in another table. The query label stays hidden; known context labels enter early, so feature extraction can depend on the task.

The paper describes a small table-level network alternating attention across columns and rows. A larger network mixes row representations along foreign-key links and across context examples. The four axes are directions of information exchange, not four mandatory loss terms. The final representation predicts the query target. [Primary reading: §3 and Figure 3](https://arxiv.org/html/2604.12596v1#S3).

[[FIG:architecture]]

**Trace the highlighted query:** its cells become a row vector; connected child rows contribute information; labeled context examples inform the prediction. Each context subgraph respects its example's time cutoff. The paper also describes local/history and recent global context, lagged targets, and ensembling. These are protocol choices to audit, not details to replace with convenient defaults.

Pretraining learns weights across synthetic and real tasks. In the selected paper tables, in-context inference uses frozen base-model weights and at most 10,000 context examples, drawing from training and validation splits. Fine-tuning is a separate experiment. The schematic uses symbolic dimensions: *R* rows, *C* columns, *d* embedding width and *K* context examples. It does not invent unreleased widths, layer counts, weights or a trainer. [§§3–4](https://arxiv.org/html/2604.12596v1#S4).

**Predict:** if the desired target changes from churn to future spending, which stage should first be able to use that task information? Explain why simply concatenating a task label at the final head would depict a different mechanism.

## 3 · Freeze the question before computing the gap

A **comparison contract** records source version, tasks, metric, eligible methods and aggregation. Our frozen source is **KumoRFM-2 v1, 14 April 2026, Tables 3, 4, 7 and 8**. It contains 28 task columns across four suites of results. Every one of its 401 task cells and 90 summary cells is preserved and independently parsed.

“Open” here means **an inspected open implementation**. It does not certify every backend's weight license, the original training corpus or historical reproducibility. The bounded foundation pool is RDBLearn and Griffin. The bounded supervised pool is LightGBM, GraphSAGE, RelGNN and RelGT. Other published rows remain in the complete source tables but are outside this access audit. OpenRFM is a later paper and has no row in this frozen comparison.

All included values are **paper-reported test results**; we did not independently establish equal information access across the authors' runs. The paper permits training-plus-validation context for Kumo. Treat the resulting gap as descriptive published evidence, not as a controlled architecture ablation.

**Two different questions:**

- **Single method:** which eligible method has the best aggregate across all tasks in this table? This is a retrospective test-set summary; a deployable choice needs validation data.
- **Taskwise oracle:** what if we picked the best eligible test result separately for every task? This is an optimistic envelope, not one trained model or a valid selection policy.

Require complete coverage. A missing task is not a zero or a loss. Do not let a method evaluated on one easy task compete against an all-task average.

## 4 · Work one calculation, then change the pool

**Worked example — driver-dnf.** The displayed AUROCs are 84.59 for KumoRFM-2 and 70.87 for RDBLearn. The difference is **13.72 percentage points**, or 0.1372 on a 0–1 AUROC scale. It is not a 13.72% relative improvement.

For a regression task, smaller MAE is better. On ratebeer/user-count, 7.298 versus 7.374 means **0.076 raw MAE units in Kumo's favor**. These units belong to that task. To combine regression tasks, first divide each method's MAE by the same task's LightGBM MAE, then average those ratios. The LightGBM reference is therefore exactly 1.

$$\text{gap}_{AUC}=AUC_{Kumo}-AUC_{open},\qquad \text{gap}_{MAE}=MAE_{open}-MAE_{Kumo}.$$

$$\mu_m=\frac{1}{T}\sum_{t=1}^{T}\frac{MAE_{m,t}}{MAE_{LightGBM,t}}.$$

Here *T* is the number of tasks, *t* indexes a task, and *m* names a method. A positive gap means Kumo is better. For regression aggregates we subtract normalized means; that result is dimensionless, not a percentage or a raw MAE.

**Predict before revealing:** will the “best open” gap grow or shrink if supervised methods become eligible?

<div id="gap-predict"></div>
<noscript><p>It shrinks on Table 3: the bounded supervised pool includes RelGNN, which is stronger on the aggregate than RDBLearn.</p></noscript>

[[FIG:gaps]]

[[SUMMARY]]

**Try it:** change the pool, then the selection rule. Keep the source fixed. Explain why a different gap need not imply that any model changed.

<div id="sota-explorer"></div>
<noscript><p>The static summary above and full tables below preserve every result. The notebook lets you change the same comparison assumptions in Python.</p></noscript>

[[CODE]]

## 5 · Audit the arithmetic before repeating the headline

Our independent parser and rational-arithmetic check found **five aggregates outside displayed-rounding bounds**. For example, the five Table 4 KumoRFM-2 cells average to **79.782**, while the paper prints **79.96**. A printed value with two decimals represents approximately ±0.005 rounding uncertainty. Even allowing that uncertainty in both task cells and summary cannot bridge this difference.

[[DISCREPANCIES]]

This is a reproducible discrepancy under the stated arithmetic. It does not identify its cause. Different unpublished inputs, aggregation choices or a transcription problem would require author clarification. We preserve the published number alongside the recomputed one.

**Ranks need a contract too.** Rank 1 means best within a specified method pool. Our ranks include all displayed rows in that table and assign tied displayed scores their average occupied rank. Two tied leaders therefore receive 1.5 each. **34 of 45** printed mean ranks differ from this rule. Hidden precision can break displayed ties; a different rank pool can also change ranks. Those causes are not established here. Rank differences are reported as `DIFFERS_DISPLAYED`, not automatically as paper errors.

> **Scope check.** Rounding intervals are bounds from printed precision, not statistical confidence intervals. The tables do not give us per-seed predictions. Do not invent error bars, a significance test or a fresh-run uncertainty estimate.

[[TABLES]]

## 6 · Keep the date visible

A frozen paper comparison answers “what does this source report?” It cannot certify today's global SOTA. Our bounded **2 October 2026** source check found a June OpenRFM paper and a July RDBLearn v1.1 paper. These need separate protocol review before their results can enter a comparable pool. The April tables stay immutable. [OpenRFM v1](https://arxiv.org/html/2606.04320v1) · [RDBLearn update v1](https://arxiv.org/html/2607.05476v1).

The source-access attempt also encountered HTTP 404 for the Kumo repository and HTTP 401 for a leaderboard data endpoint. Those are retrieval failures, not proof that the resources do not exist. Keep the failed attempts, leave **current global SOTA `NOT_ESTABLISHED`**, and state the actual search boundary. [Frozen tracking ledger](../labs/evidence/l191/packet/tracking-ledger.json).

For [Lesson 192's open-model reproduction](0192-open-fm-setup-data.html), use this table to shortlist a baseline. Then audit weights, data identity, temporal rules, feature construction, context/seed choices, validation-only selection and aggregate cost before execution. An inspected repository alone does not finish that contract.

## 7 · Build it and defend it

[Student notebook](../labs/0191-kumorfm2-sota-tracking.ipynb) · [Executed solution](../labs/solutions/0191-kumorfm2-sota-tracking.ipynb) · [Read the executed lab](../labs/html/0191-kumorfm2-sota-tracking.html) · [Reference card](../reference/kumorfm2-sota-tracking.html) · [Reproduction protocol](../labs/l191-reproduction.md).

The notebook embeds the frozen source packet and the full parser/reconstruction implementation. Three learner functions control metric direction, evidence eligibility and complete-coverage comparison. Its checks feed those functions into the real 401-cell reconstruction. It uses Python's standard library; no model, API key or repository checkout is needed.

**EXIT:** submit one gap with its table, pool, rule, units and source date. Explain one published/recomputed discrepancy, one missing comparison and one condition that would make a fresh run comparable. State how your baseline choice changes the research question from L190.

<div id="gap-teachback"></div>
<noscript><p>Explain why “4.05 points ahead” cannot by itself establish current global superiority or a successful fresh reproduction. Include the pool and the evidence date.</p></noscript>

Ask the teaching agent about any unclear calculation or bring your written defense for feedback. Tomorrow, reconstruct the direction of both gap formulas from memory. In one week, defend the difference between one method and a taskwise oracle without reopening the table.
