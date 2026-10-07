# B24 · Defend the architecture and the thesis

**The win:** write a claim that your evidence can support, choose a fair way to challenge it, and say what result would make you change your mind.

[Previous: B23 · reproduce one comparison](b23-declared-comparison.html) · [Context and cost: B18a](b18a-context-state.html) · [Metrics: B19a](b19a-predictive-distributions.html) · [Next: Year 6 handoff](../plan/year-5-6-bridge.md#b24)

[Student notebook](../labs/b24-architecture-thesis-defense.ipynb) · [Executed notebook](../labs/html/b24-architecture-thesis-defense.html) · [Reference card](../reference/b24-architecture-thesis-defense.html) · [Five-page proposal template](../reference/b24-proposal-template.html) · [Editable template](../labs/b24-proposal-template.md) · [Reproduction contract](../labs/b24-reproduction.md)

**Route through the lesson.** Spend 10 minutes on the claim and worked example, 15 on baselines and falsification, then use the notebook to audit the evidence. Draft the proposal over later sessions. This is the bridge's exit defense; reading it does not mark the bridge complete.







B23 asked whether a declared computation was reproduced. B24 asks which research decision that computation justifies. The mission is to test whether learned relational models deliver value beyond strong single-table systems. A defense must therefore explain where information enters, what the baseline receives, and which observation would contradict the claimed benefit.

A **claim** is a statement about a specified population and procedure. **Evidence** is the observed result and its provenance. A **warrant** is the reasoning connecting the evidence to the claim. A **falsification test** is a planned observation that would make us revise the claim. These four pieces belong together.

> **In plain terms.** “It scored higher” is an observation. “Therefore graph attention is necessary” needs another experiment.

## 2 · Start with the result we actually have

**Frozen experiment.** `B24-RESEARCH-DEFENSE-AUDIT` replays the complete B23 packet: 30 published-model checkpoint evaluations and ten course logistic fits. Each arm uses the same ten draws of 512 labeled supports and the same 702 queries from `rel-f1/driver-dnf`. A **support** is a labeled example available to the predictor. A **query** is a row whose answer is withheld until evaluation.

The primary reading is [RDB-PFN v5, evaluation protocol and Table 9](https://arxiv.org/html/2603.03805v5#A6). Inspect its selected column, then compare the [B23 frozen contract](../labs/b23-reproduction.md). All B23 inputs, target orientation, preprocessing and checkpoints remain fixed. B24 changes only the audit implementation; it makes no new model predictions.

<div id="b24-predict"></div>

{{RESULTS}}

**Worked example.** The mean paired RDB-PFN minus TabICL effect is **+0.004366 AUROC**. RDB-PFN wins on six of the ten draws. “Paired” means subtracting model scores with the same support identity before averaging. AUROC measures how often a positive example ranks above a negative one, with half credit for a tie. It does not establish calibrated probabilities. The **sample standard deviation (SD)** describes how scores vary across support draws: subtract their mean, square the differences, sum them, divide by nine for ten draws, then take the square root. It is not a confidence interval for performance on new databases.

{{EFFECTS}}

A defensible statement is: “On this released F1 task and these ten support draws, RDB-PFN has a higher mean AUROC than TabICL v1.1.” A stronger statement about other databases needs independent database evidence. A statement about graph-native inference is unsupported by this comparison: RDB-PFN and TabICL here both consume the same materialized feature table.

**Independent replay.** B24 authenticates the archive, checks every complete `(driverId, date)` identity, support identity and label, recomputes all 40 AUROCs using average ranks, and reconstructs all ten logistic probability vectors from saved coefficients. The largest AUROC discrepancy is below `1.2e-16`; the baseline vectors agree exactly. These are replay checks of the B23 artifacts, not another fit.

> **Scope check.** Historical checkpoint identity and raw feature availability remain `NOT_ESTABLISHED`. The checkpoint uses width 96 where the paper appendix describes 128. Released labels retain their original orientation, which complements the current raw DNF reconstruction. Full feature regeneration, pretraining and the whole-paper benchmark remain `NOT_RUN`. Numerical agreement cannot resolve those gaps. B23's fixed logistic comparator is not tuned trees.

## 3 · Defend an information path, not a model name

A **representation** is the form in which information reaches the predictor. An **architecture** is the set of operations mapping that representation to an output. A **training prior** is the distribution of tasks used to learn the weights. Changing any one can change predictions. Name which one your hypothesis concerns.

**Worked example: two histories, one summary.** History A contains values `[0, 1]`; history B contains `[0.5, 0.5]`. Both means equal `0.5`. Their population variances are `0.25` and `0`. A predictor receiving only the mean cannot distinguish them. This is loss in the representation, not proof that the predictor is weak. Add variance to the flat features and the distinction returns.

{{ARCHITECTURE}}

The diagram asks a precise question: **which operation can preserve the distinction?** For A, subtract the mean to get `[-0.5, 0.5]`, square to get `[0.25, 0.25]`, and average to get `0.25`. For B, both centered values are zero. A learned neighbor aggregator might learn a useful distinction; a hand-built variance feature already supplies this one. Neither is guaranteed to win a real task.

Three architecture paths belong in your proposal:

- **Flat path:** legal historical records → fixed aggregates or deep feature synthesis → vector → trees, a multilayer perceptron (MLP), or a tabular foundation model (TFM) → prediction. An MLP is a stack of learned linear transformations and nonlinear activations. Deep feature synthesis composes declared aggregations along table relationships. It can retain substantial relational information.
- **Graph path:** the same legal records → typed entities and foreign-key edges → learned neighbor messages → entity representation → task head. A task head converts the learned representation into the prediction. See [RelGNN/RelGT mechanisms in B11](b11-supervised-relational-baselines.html).
- **Cell path:** legal table cells and their row/column/relationship identities → typed cell tokens → allowed attention operations → task output. Attention combines permitted source vectors according to learned weights. See [RT in B10](b10-relational-transformer.html). Trace the actual sampler as well as the attention mask.

RDB-PFN belongs to the flat inference path even though its pretraining generator is relational. TabSTAR's parameter updates differ from forward-only in-context adaptation. A hypernetwork generates predictor weights; a support-context method supplies labeled input state. [B13](b13-synthetic-relational-data.html), [B07](b07-semantic-transfer.html), and [B07a](b07a-hypernetworks.html) supply the model-specific traces.

**Your task.** Sketch your chosen path. Mark one load-bearing operation, its legal inputs, its output, and a simpler operation that could replace it. Predict the effect of that replacement before running it. An **ablation** changes a specified part while preserving the rest of the comparison.

## 4 · Make the baseline capable of disproving you

A **baseline** is a comparator answering a research question. Its value comes from the alternative explanation it tests. A weak baseline may be easy to beat while leaving the central explanation untested.

**Worked decision.** If the thesis says learned graph aggregation improves over engineered summaries, include tuned trees with time-safe features. Give both methods access to the same eligible historical records and target labels. Their representations may differ; record the transformation and cost of building each. Also include a strong flat foundation model to test whether task pretraining, rather than graph computation, explains the gain.

**Matched information** means comparable access to raw observations, labels and prediction-time history. It does not require identical tensors for a graph and a tree. **Matched selection budget** means a declared equal opportunity to choose a configuration using training/validation data. Log total training, feature engineering, inference, failed attempts and human design effort separately. Equal tuning budgets do not equalize inherited pretraining cost.

Use the inventory below to make decisions. “Core” means required in the proposed future comparison unless a task-specific reason is recorded. “Conditional” means include when its mechanism addresses the hypothesis. “Defer” means a named limitation and a reopening condition, never “it would probably lose.” **No inventory row is a new B24 training result.**

{{COVERAGE}}

**Current candidate check · 4 October 2026.** The [source receipt](../labs/sources/b24/candidate-audit.json) records the dated official pages and access results. This is a candidate/source audit, not a current leaderboard. LimiX-2, TabFM, EXAONE, Nori and RT-J remain explicit decisions even if their elective was skipped. Seldon and NEXUS provider claims require the same information/cost protocol before comparison. A provider-controlled benchmark is not independent verification.

**Selection rule.** Pick a compact set that can falsify the chosen explanation. If the full comparison exceeds the budget, report the missing runs and narrow the empirical claim. Do not remove a strong competitor after seeing its test score. Cite both the reason for exclusion and the evidence that would reopen the decision.

## 5 · Freeze two ways the thesis could fail

A falsification rule must name the task, challenger, reference, primary metric, direction, numerical threshold, matched information and selection budget, and the action taken if the rule fires. **Preregistration** means fixing those decisions before inspecting the evaluation outcomes they govern.

For a higher-is-better metric such as AUROC, define the oriented effect as challenger minus reference. For a lower-is-better loss, define it as reference minus challenger. Positive then always favors the challenger. A rule such as “revise if mean effect is at most zero” has an unambiguous boundary: a tie triggers revision.

1. **Simpler-baseline test.** On a named future task, compare the proposed relational method with tuned trees plus time-safe features. If the prespecified paired mean effect is at most zero, withdraw superiority on that task. Preserve every planned run, including failures. A practical margin greater than zero can be chosen before evaluation if the deployment benefit requires it.
2. **Untouched-task test.** Lock a database/task that has not informed checkpoint, architecture, feature, prior or threshold selection. Compare with the locked strongest baseline. If the effect is at most the prespecified threshold, narrow the transfer claim. Record prior dataset use across this course; another F1 support seed is not an untouched task.

An untouched task is stronger evidence than another seed, but one new task still cannot establish universal transfer. If no genuinely untouched candidate is available, record `PENDING_TASK_RESERVATION`; do not relabel a reused benchmark. Uncertainty should match the sampling unit: show support variability within tasks and task-level variability when claiming cross-task benefit.

{{FALSIFICATION_WIDGET}}

**Metric contract from B19a.** State whether the decision needs ranking, class probabilities, a point estimate or a predictive distribution. The example F1 claim is ranking-only, so AUROC is primary and calibration is unestablished. **Calibration** asks whether stated probabilities agree with observed frequencies. A **proper probability score** rewards reporting the true probability distribution in expectation. If a downstream decision uses probabilities, predeclare such a score, a separate calibration partition and calibration diagnostics. Do not fit calibration on final test labels. [Use the metric/calibration contract](../labs/b19a-metric-contract.md).

**Context contract from B18a.** Freeze support selection, support-label availability, preprocessing, checkpoint, cache identity, update trigger, rebuild frequency, cold/warm latency, memory and total cost. Define what happens when an entity or label arrives late. Changing a support row can change the deployed predictor without changing weights. [Review context as state](b18a-context-state.html).

**Task expansion is conditional.** If your thesis forecasts future outcomes, add B19b's horizon, origin and as-of covariate contract. For a static classification thesis, explicitly mark forecasting out of scope. [Forecasting contract](../labs/b19b-task-contract.md).

## 6 · A score cannot repair invalid evidence

The defense rubric has five axes. Score each 0 for missing/invalid, 1 for partial, or 2 for complete and justified. An assessor must read the actual proposal and evidence; string-filled fields are not proof.

| Axis | What earns 2 |
|---|---|
| Protocol validity | Frozen identities, temporal availability, selection and metric; unresolved gaps bar claims needing them. |
| Baseline fairness | Strong relevant comparators, matched information and selection budgets, justified exclusions. |
| Reproducibility | Named scope, executable path, immutable inputs/predictions and complete evidence for the declared claim. |
| Evidence interpretation | Effect and variability at the right unit; source, replay, fresh execution and historical claims separated. |
| Falsifiability | Two concrete reversal tests with untouched-task audit and a decision that changes if they fail. |

Eligibility requires **at least 8/10, no zero, no unresolved leakage, and the required reproduction evidence**. A proposal about future research may cite complete selected-release evidence while declaring historical validity open. A claim of historically valid deployment cannot pass using the same packet. The assessor determines which gate the actual claim requires.

{{GATES}}

{{DEFENSE_WIDGET}}

> **Scope check.** The controls simulate hypothetical assessment inputs. They do not change the actual B23 evidence or grade your writing. Your status remains `PENDING_WRITTEN_DEFENSE` until your submission is assessed. A replay proves executable evidence, not learner mastery.

## 7 · Build the five-page defense

Open the [printable proposal template](../reference/b24-proposal-template.html). It supplies five page-sized prompts, not a filled learner answer:

1. **Hypothesis and claim:** population, mechanism, practical motivation, current evidence and nonclaims.
2. **Architecture and coverage:** your end-to-end information path and justified family inventory.
3. **Frozen protocol:** identities, splits, availability, selection, metrics, context/update/cost contracts.
4. **Evidence and limits:** B23 replay, per-draw effects, deviations, run status and missing historical facts.
5. **Falsification and decision:** two reserved tests, rubric evidence and next research action.

**A short worked opening, not your submission.** “I hypothesize that learned neighbor aggregation can improve a specified future task beyond tuned trees with time-safe aggregates at a fixed selection budget. B23 motivates studying pretrained predictors: it shows a small positive mean RDB-PFN–TabICL difference on one released F1 task. Both methods use flattened inputs, so this does not establish my graph hypothesis. I will test the graph-specific claim with matched eligible records and a strong flat comparator. Missing historical feature provenance bars a deployment-validity claim from B23.”

The notebook implements three decisions you need to defend: pair effects by draw identity, apply the rubric without bypassing gates, and reject incomplete falsification contracts. Blank and incorrect functions must fail before a report is produced. **A validator checks structure; it cannot verify that your prose is true or that a task was never used.** Attach the underlying prior-use and availability evidence for the assessor.

<div id="b24-teachback"></div>

## 8 · Exit and Year 6 handoff

Submit the proposal, architecture map, baseline table, B23 evidence folder plus B24 replay receipt, and two falsification contracts. Ask the agent to challenge one warrant and assess each rubric axis. Do not fill a learning record from author-run checks.

| Next lesson | B24 handoff |
|---|---|
| L201 · Hypothesis | Narrow claim, mechanism and revision condition. |
| L202 · Sources | Versioned primary-source matrix and access/deviation ledger. |
| L203 · Protocol | Task reservation, splits, legal information, selection and metrics. |
| L204 · Baselines | Inclusion/exclusion decisions and total budget accounting. |
| L219 · Stress tests | Simpler-baseline and untouched-task failure rules. |

**Return after 1, 7 and 30 days.** Day 1: derive the two-history variance example from memory. Day 7: defend one exclusion, then argue for including it. Day 30: reconstruct the two failure rules without looking. Bring any unclear step back to the agent; ask for another worked example or a skeptical review.

## Reproduction appendix · exact scope and commands

From the repository root, run:

```bash
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_audit_b24.py
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_verify_b24.py
```

The downloaded notebook embeds the authenticated evidence and visible audit code; it needs only Python and NumPy to replay. It can run without a repository checkout or network access. The executed HTML is an author reference. Live Colab execution remains `NOT_CHECKED`.

B24 uses a **$0 paid-compute budget** and a **3,600-second aggregate local numerical/check cutoff**, including preparation, failures and validation. The [local ledger](../labs/evidence/b24/local-budget.json) records attempts. At cutoff, leave `INCOMPLETE`; retain all declared runs in the contract. A separate fresh-inference path remains documented in [B23](../labs/b23-reproduction.md); B24 does not dispatch it.

[Full replay report](../labs/evidence/b24/report.json) · [Independent verification](../labs/_verify_b24_results.json) · [Primary paper](https://arxiv.org/html/2603.03805v5) · [Bridge specification](../plan/year-5-6-bridge.md#b24)
