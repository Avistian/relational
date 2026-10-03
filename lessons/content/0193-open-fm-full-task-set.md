<div class="eyebrow">YEAR 5 · LESSON 193 · REPRODUCTION PRACTICE</div>

**One skill:** report a multi-task experiment without letting missing runs, incompatible units or selection decisions disappear inside an average. Read the core in about 15 minutes; do the notebook separately.

[[STATUS]]

## 1 · From one task to a task set

Our mission is to test whether relational structure adds value. A good result on one convenient task cannot carry that claim across databases. [Lesson 192's setup work](0192-open-fm-setup-data.html) chose RDBLearn and inspected one task. Here we keep the same paper and released pipeline, but expand the **contract** to every reported task. L192 completed its setup audit, while model inference remained blocked. Its single-task protocol and saved preprocessing audit motivated the [fresh L193 check](../labs/evidence/l193/packet/preprocessing.json) that forms the bridge here.

**Retrieve first:** What identifies a temporal query? Which split chooses a model? What does another support seed replicate?

<details><summary>Check after answering</summary><p>A complete key includes entity and cutoff; the same entity can recur. Validation selects a configuration. Another support seed repeats within-task sampling; it does not create another independent database.</p></details>

[Lesson 178](0178-fair-model-comparison.html) introduced this relational-features-plus-tabular-model approach. [Lesson 190](0190-research-gap-checkpoint.html) separated evidence from a research claim. Today's new difficulty is accounting: the denominator is part of the experiment.

## 2 · Model architecture: where the shared risk lives

RDBLearn turns related rows into a fixed-width table, then supplies examples and queries to a pretrained tabular predictor. The relational feature generator has no learned encoder weights. The backend weights stay frozen during in-context prediction; fitting support/preprocessing is still data-dependent. Depth and backend are selected on validation. See the [primary paper, §§3–5](https://arxiv.org/html/2602.18495v1#S3).

[[FIG:architecture]]

Read one path: a query is `(entity, cutoff)` → restrict reachable rows to the allowed history → aggregate along join paths → produce one feature vector → apply the fitted preprocessing map → predict using labeled support. Support labels must also be available by the relevant prediction time. This is the intended contract; the drawing does not certify the released pipeline's temporal validity.

**Predict:** if preprocessing changes a fitted category's code after seeing a new query category, can every task safely reuse that pipeline?

We freshly ran the original full preprocessor on synthetic inputs. Fitting categories `b,c,d` gives codes `0,1,2`. Introducing `a` produces a sorted class list `a,b,c,d`, so the same known categories become `1,2,3`. The frozen support matrix still contains the earlier codes. The same query `b` is encoded differently alone and beside `a`.

[[FIG:codes]]

<div id="category-explorer"></div>
<noscript>Measured diagnostic: unseen a or 0 shifts all three known codes; unseen z or e shifts none. Numeric controls stay unchanged. Benchmark score impact is NOT_ESTABLISHED.</noscript>

This establishes a **source preprocessing invariant failure under the recorded environment**. It does not establish how often real tasks encounter the condition, how a backend reacts, or whether a paper score changes. We stop the declared suite at this shared admission check. Patching the encoder and claiming the original experiment would change the protocol. [Fresh observations](../labs/evidence/l193/packet/preprocessing.json) · [original-source diagnostic](../labs/_preflight_l193.py).

## 3 · Freeze the whole experiment before measuring

The chosen target is **RDBLearn, arXiv 2602.18495v1, Tables 1–3**: eight RelBench classification tasks, eight RelBench regression tasks and five 4DBInfer classification tasks. The full scope is the RDBLearn column; retraining every comparator or pretraining the tabular backends is outside it. [Paper §5 and Appendix A](https://arxiv.org/html/2602.18495v1#A1).

For each task and each course seed `0,1,2`, search depths `2,3,4` × backends `TabPFN-v2, TabPFN-v2.5, LimiX-16M`. Keep the published 10,000-example support cap and official splits. Three repeatability seeds are a disclosed extension: the original seeds are not recovered. Source release 0.1.2 and FastDFS 0.2.1 are pinned; checkpoint bytes, full data identities and the historical environment remain unestablished.

**Count it:** `21 × 3 × 9 = 567` validation candidates, then `21 × 3 = 63` selected test evaluations. A failed candidate is not permission to select among eight. A missing seed is not permission to report a two-seed result as three-seed reproduction.

[[SELECTION_CODE]]

The function refuses incomplete candidate sets and test-split scores. It breaks equal validation scores using the order frozen before execution. Classification maximizes AUROC; regression minimizes MAE. Fresh scoring would align complete query identities and independently recalculate metrics before admitting any `COMPLETE` row.

**Admission and budget:** $10 total, planned stop at $8 with $2 reserve, including preparation, retries and verification. The full-run forecast remains `NOT_ESTABLISHED`: the source gate failed before a model pilot. No paid run was launched. The [executable admission operator](../labs/_reproduce_l193.py) exits with a documented stop; its downstream GPU executor is **not implemented or validated**. The full [validation schedule](../labs/evidence/l193/validation-schedule.json) and [test schedule](../labs/evidence/l193/test-schedule.json) remain available for continuation.

## 4 · Two denominators before one mean

First summarize **within each task** over its complete seed set. Then summarize **across tasks** within a comparable metric group. These are different sources of variation. For synthetic seed scores `[0.60,0.70,0.80]`, the mean is `0.70` and sample SD is `0.10`, using denominator `3−1`. A missing third run leaves the full-seed mean blank. Seed SD is not uncertainty across databases.

[[FIG:aggregation]]

The browser exercise below is synthetic. Its eight task scores are `[.9,.9,.9,.5,.5,.5,.5,.5]`. Each checked box stands for a task with all three seeds complete. Start with only the three easy tasks. The observed-subset mean is `.900`; restoring all eight gives `.650`. Dropping difficult tasks creates a more flattering answer to a different question.

<div id="coverage-explorer"></div>
<noscript>Synthetic example: three selected tasks average .900; all eight average .650. Until all eight are present, the full-suite mean is withheld. Actual L193 model coverage remains 0/21.</noscript>

**Equal task weighting:** average each task's seed mean once. Do not pool every prediction row; large tasks would dominate. Tasks within one database may also be dependent, so an across-task SD is descriptive rather than a database-transfer confidence interval.

**Units matter:** AUROC is dimensionless. Raw MAEs for money, counts and rates are not commensurate. Given verified baseline errors `b_t`, normalized MAE is `MAE_t / b_t`, then averaged across tasks. Example: errors `[5,.2]` and baseline errors `[10,.1]` yield normalized errors `[.5,2]`, mean `1.25`. The smaller raw error can be worse relative to its baseline.

The paper's §5 describes normalization using a no-relational-context baseline, but Table 2 labels its values MAE and does not supply an unambiguous normalization recipe and denominators. The AutoGluon-without-RDB column is not automatically that baseline. We retain raw per-task targets and leave aggregate regression parity **NOT_ESTABLISHED**. The lab rejects missing or nonpositive denominators rather than silently substituting a target standard deviation.

<div id="multitask-quiz"></div>
<noscript>Choose: “Keep every missing task visible.” This preserves the declared denominator. Averaging only successful tasks changes the population.</noscript>

## 5 · The full results table, including absence

[[RESULTS]]

Every row retains the published target beside a blank fresh result. The paper numbers are reference scalars; they are not saved predictions or fresh measurements. For compactness, the table shows one task row; the [run ledger](../labs/evidence/l193/packet/runs.json) retains all 63 seed slots and reasons. Every row has the same suite-level stop reason, not an asserted task-specific defect.

Arithmetic on the rounded published classification values gives **0.7452875** over the eight RelBench tasks and **0.7879000** over the five 4DBInfer tasks. We independently parsed and checked all 21 table entries. Those two means reproduce a calculation on printed numbers, not the experiments that generated them; seed variance cannot be recovered from these single scalars.

Actual fresh model evaluations: **0 / 630**. Actual completed task results: **0 / 21**. The diagnostic is complete; full model reproduction is **INCOMPLETE_SOURCE_PREPROCESSING_GATE**. [Report](../labs/evidence/l193/report.json) · [independent verification](../labs/_verify_l193_results.json).

## 6 · Implement, challenge, defend

[Student notebook](../labs/0193-open-fm-full-task-set.ipynb) · [executed solution](../labs/solutions/0193-open-fm-full-task-set.ipynb) · [readable lab](../labs/html/0193-open-fm-full-task-set.html) · [field guide](../reference/open-fm-full-task-set.html).

Implement three live contracts: `choose_config`, `summarize_task`, `aggregate_suite`. Each feeds the checks or complete packet report. Reject a test-split candidate; mark one seed missing; try averaging two regression tasks without their denominators. The portable replay authenticates evidence and recomputes the report without downloading models. It replays recorded diagnostic observations; the original-source diagnostic is a separate environment-dependent command.

**EXIT:** write a paragraph containing (1) the named 21-task target, (2) the strongest completed evidence, (3) the precise reason model inference stopped, (4) the denominator and uncertainty you would report after a valid complete run, and (5) what must be established before continuation. Explain why fixing the encoder creates a new experimental variant. [Lesson 194](0194-open-fm-analysis-report.html) turns this ledger into a per-task analysis and report; blank scores cannot support win/loss attribution.

Ask the tutor about any unclear step, or submit your implementation and defense for review. This prepared package does not establish learner mastery: **PENDING_WRITTEN_DEFENSE**.
