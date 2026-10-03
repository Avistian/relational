<p class="eyebrow">YEAR 5 · LESSON 194 · ANALYSIS & REPORT</p>

# Explain what the evidence can support

**Your tangible win:** write a reproduction report that distinguishes a result, a possible explanation, and the experiment needed to test that explanation. This closes the reporting step of lessons 192–194; the model reproduction itself remains incomplete.

[[STATUS]]

## 1 · Recall before reading

From [L192](0192-open-fm-setup-data.html), what must stay consistent between support and query preprocessing? From [L193](0193-open-fm-full-task-set.html), what happens to a task mean if one required seed is missing? Looking ahead to analysis: what evidence would turn a suspected failure mode into a tested explanation?

<div id="warmup"></div>
<details><summary>Check your recall</summary><p>A fitted category must keep its meaning. Missing seeds leave the full-seed result incomplete. A controlled comparison must change the suspected factor while holding relevant alternatives fixed; a score alone does not identify a cause.</p></details>

**Mission connection.** A persuasive case for relational learning needs explanations a skeptic could test. A polished report that hides missing results weakens that case. Today's skill is deciding which sentence the evidence permits.

## 2 · Trace the model before explaining it

RDBLearn builds features by following relationships between database rows, then passes those features to a pretrained tabular predictor. **Support** means the labeled examples supplied at prediction time. A **query** is a row whose label must be predicted. A **cutoff** is the time beyond which information is unavailable for that prediction. **DFS**, deep feature synthesis, composes aggregations along relationship paths. The backend uses support examples without updating its pretrained weights. [Primary reading: RDBLearn v1, §3 and §5](https://arxiv.org/html/2602.18495v1#S3).

[[FIG:architecture]]

Follow one customer query: select records permitted by its cutoff; summarize linked transactions; encode support and query with the same fitted map; supply both to the frozen predictor; score its prediction. Depth selects how far feature construction follows relationships. Validation chooses depth and backend; the test split evaluates that choice. This is the intended pipeline, not a certification that all temporal checks have passed.

**Worked example — recorded source diagnostic.** Fit categories `b,c,d` as `0,1,2`. In the recorded environment, adding query category `a` expands and sorts the class list, shifting the known codes to `1,2,3`. Stored support still has the earlier meaning. Numeric controls remain unchanged. This identifies a representation consistency failure. It does not tell us whether any benchmark query triggers it or how a predictor's score changes. [Recorded observations](../labs/evidence/l194/packet/preprocessing.json).

**Held fixed:** known categories, fitted support and numeric controls. **Varied:** the unseen query category. **Measured:** codes, not predictions. L194 authenticates and replays this saved diagnostic; it does not rerun the upstream preprocessor.

## 3 · A difference is a question, not its explanation

<div id="prediction"></div>
<noscript>Predict: does a favorable published difference prove that relational structure caused the improvement? No. The compared pipelines differ in more than one component.</noscript>

**In plain terms.** First ask which score is better. Then ask what else changed between the two systems. These are different questions.

**Comparator fixed before analysis:** AutoGluon+DFS, the supervised pipeline with relational feature synthesis, in all 21 rows of Tables 1–3. This comparison asks how two complete pipelines rank in the published table. It does not isolate the value of adding relational structure: both already use it. The paper describes its comparison families in [§5](https://arxiv.org/html/2602.18495v1#S5).

**AUROC** measures how well a binary predictor ranks positives above negatives; larger is better. **MAE**, mean absolute error, averages absolute prediction errors in the target's units; smaller is better. Define an **oriented gap** so positive always favors RDBLearn:

- AUROC gap = RDBLearn score − comparator score.
- MAE gap = comparator error − RDBLearn error.

**Worked examples from the printed table.** Item churn: `.8188 − .7953 = +.0235` AUROC. Item lifetime value: `57.0000 − 48.5044 = +8.4956` target units of MAE reduction. The signs have the same interpretation; the magnitudes have different units. Neither supplies a causal explanation.

[[FIG:gaps]]

[[GAP_CODE]]

A missing result returns `None` (an explicit absence), not zero. A zero gap means equal displayed scores; it is not evidence of equivalence. Rounded scalars cannot recover seed variability, paired prediction errors, or a significance test. Three tasks from one database are not three independent databases.

## 4 · The complete report, including what did not run

The target remains the **RDBLearn v1 column across all 21 tasks**, with the released source pinned to commit `b5b03ebf8091547285a6e06cba53d2d1a40cb171`, release 0.1.2 and FastDFS 0.2.1. Per task, course seeds `0,1,2` each search depths `2,3,4` × TabPFNv2/TabPFNv2.5/LimiX-16M, official splits, and a 10,000-example support cap. These seeds extend repeatability; they are not recovered historical seeds. See the [inherited protocol](../labs/evidence/l194/packet/protocol.json).

That is `21 × 3 × 9 = 567` validation candidates and `21 × 3 = 63` selected tests. The source admission failure stopped all 630 model evaluations. The downstream model executor has not been implemented or validated. The full raw-data, feature availability, checkpoint and regression-normalization audits also remain unfinished. No new inference is authorized in this lesson.

[[TABLE]]

[[COUNTS]] These are descriptive signs from rounded published values. Every fresh mean and sample standard deviation (variation across required seeds) stays blank. The [complete report](../labs/evidence/l194/report.md) includes all task rows, limitations and continuation requirements; the [JSON report](../labs/evidence/l194/report.json) is generated by the same visible functions used in the notebook.

**Scope check.** Do not average AUROC with MAE, or raw MAEs measured in different target units. Regression normalization requires a verified reference error per task; those denominators remain `NOT_ESTABLISHED`. Missing benchmark results establish neither that the model wins nor that it loses.

## 5 · Turn each explanation into a test

A **mechanistic explanation** names an operation that could change the result. An **intervention** deliberately changes a specified input or operation. A **confounder** is another varying factor that could explain the observed difference. We can write testable plans now; executing them needs valid model results and a separately approved protocol.

| Suspected reason | Vary or stratify | Hold fixed | Measure and limit |
|---|---|---|---|
| Label coverage | Nested support sizes using only labels available by each cutoff | Query set, checkpoint, features, depth, preprocessing and paired seeds | Per-task score changes. Tests context sensitivity, not pretraining scale. |
| Schema information | Remove one specified relationship path | Support, queries, checkpoint, remaining paths and preprocessing policy | Paired change after the feature intervention. Tests that path in this pipeline, not arbitrary schema transfer. |
| Cold start | Prespecify entities seen/unseen in allowed support history | Same fitted predictor, cutoff policy and scorer | Sizes, label counts and separate scores. An observational subgroup gap may reflect degree, time or label differences. |

**Worked hypothesis.** “Sparse support hurts” predicts improvement when support grows while queries and the predictor stay fixed. Use nested samples so the smaller support is contained in the larger set. Freeze the comparison before viewing test results. If the predicted improvement does not appear, the result challenges this hypothesis for the tested conditions. If it appears, it still does not explain all 21 tasks.

**Cold-start caution.** An entity unseen in labeled support can still have historical neighbors. State which kind of novelty you mean. AUROC is undefined for a subgroup containing only one class; keep its label counts and do not invent a score. These plans are course analysis proposals, not experiments reported by the paper or completed here.

**Predict before changing the evidence selector:** what additional claim does a source counterexample permit compared with a missing benchmark result?

<div id="evidence-explorer"></div>
<noscript>Published table → descriptive comparison. Saved source diagnostic → code-invariant failure. Unrun benchmark → no performance conclusion. Proposed intervention → untested hypothesis.</noscript>

## 6 · Build, challenge, defend

[Student notebook](../labs/0194-open-fm-analysis-report.ipynb) · [executed solution](../labs/solutions/0194-open-fm-analysis-report.ipynb) · [readable lab](../labs/html/0194-open-fm-analysis-report.html) · [field guide](../reference/open-fm-analysis-report.html).

Implement `oriented_gap`, `evidence_license` and `summarize_comparisons`. Each function feeds the full 21-task report. Then delete a task, insert a nonfinite score, or try an unsupported evidence type: the checks must refuse the claim. The portable notebook embeds its evidence and uses only Python's standard library for replay; no account, download or model is needed.

**Your report has six parts:** question and frozen protocol; complete per-task results; uncertainty; diagnostic and competing explanations; limitations; the next decisive experiment. Write an abstract that says the source diagnostic is informative while benchmark attribution is unestablished. Explain why the 17 favorable published signs do not complete the Year 5 reproduction exit.

<div id="teachback"></div>
<noscript>Write your own explanation before checking: the source diagnostic establishes code instability under recorded conditions; the published table supplies descriptive comparisons; no fresh predictions exist to measure benchmark effects.</noscript>

**Execution budget:** $0 new cloud/API spend, 1,800 aggregate local execution seconds including failed attempts and delivery checks. [Reproduction contract and commands](../labs/l194-reproduction.md). A repaired pipeline needs a separately named protocol, data and checkpoint audits, a validated runner and a complete-grid cost forecast; silently reducing tasks, seeds or candidates would change the target.

The report is prepared; your written defense remains `PENDING_WRITTEN_DEFENSE`. Ask me about any unclear step, or send your abstract for feedback. [Lesson 195](0195-thesis-stress-test.html) stress-tests the thesis using the strongest available evidence. Author checks, learner mastery, full reproduction, live Colab and deployment are separate claims.
