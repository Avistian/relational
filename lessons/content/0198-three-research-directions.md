<p class="eyebrow">Year 5 · Research proposals · Lesson 198</p>

# Three directions worth trying to disprove

**Your win:** turn the landscape essay into three ranked research proposals that a skeptical collaborator could assess. Each proposal must say what changes, what stays fixed, what would count against it, and what must be resolved before running it. Work through the core in about 20 minutes, then spend a separate session drafting your cards and completing the lab.

The mission is to test whether learned relational models add value beyond strong tabular approaches. A useful negative result also advances that mission: it tells us where a more complicated model is unnecessary.

[Lesson 189](0189-identify-open-problems.html) collected candidate gaps. [Lesson 197](0197-year-5-essay.html) separated a defensible claim from its evidence. **The new step here is experimental commitment:** turn each gap into a complete comparison and an explicit decision rule. [Lesson 199](0199-select-primary-direction.html) supplies the method for choosing a primary direction; the ranking here remains provisional.

[[STATUS]]





**Prerequisite recap.** A *query* is an entity at a prediction time. Its label may describe an outcome during a later window. *Availability* asks whether an input was actually knowable then. A *baseline* is the comparison method under the same information and selection rules. *Pretraining* learns from source tasks before the target task. A *prior* is the distribution of tasks used to teach a foundation predictor. An *encoder* turns rows and relationships into representations used by that predictor.

*AUROC* measures how often a positive example ranks above a negative one, with half credit for ties. It ranges from 0 to 1; higher is better. Here 0.01 means one AUROC percentage point. The proposed margin is a planning choice, not a universal standard of practical importance.

## 1 · A topic is not yet a contribution

“Use graph structure” names an interest. A proposal needs five connected statements:

1. **Known result:** what the nearest work already establishes.
2. **Candidate contribution:** the specific unanswered comparison.
3. **Hypothesis:** the change you predict in a defined population.
4. **Experiment:** intervention, matched control, complete run matrix and scoring rule.
5. **Decision:** a result that supports, weakens or leaves the hypothesis unresolved.

> **In plain terms.** Give someone who disagrees with you a fair way to prove your idea unhelpful.

**Worked narrowing.** “Temporal pretraining helps” is already studied. The [temporal-pretraining paper, §4.4](https://arxiv.org/html/2609.35219v1#S4.SS4), leaves unseen-database transfer and matched computational cost open. Narrow the proposal to those conditions. [Relational Transformer](https://arxiv.org/html/2510.06377v1) already studies transfer across relational data, so “cross-database” alone is not the novelty argument.

**Novelty is a comparison with the nearest work.** Read methods and limitations, not just titles. Write down which axis differs: population, information policy, operator, objective, comparator or evaluation. If the closest work answers the same question under the same conditions, revise or retire the proposal.

> **Scope check.** The six frozen primary texts are selected evidence, not an exhaustive review. Two additional abstract-level leads—[Curriculum Matters](https://arxiv.org/abs/2607.29120) and [PluRel-to-RDB-PFN](https://arxiv.org/abs/2607.29129)—make broad claims about “new synthetic relational pretraining” particularly unsafe. Their results are not reproduced here. Novelty remains `NOT_ESTABLISHED` for all three proposals.

## 2 · Direction one: do availability assumptions change the comparison?

**Question.** On `rel-f1/driver-dnf` and `rel-trial/study-outcome`, does the RDB-PFN versus flattened-tree contrast change when timestamp filtering is replaced by an explicit availability policy?

A timestamp records an event time. An availability policy records when an input can be used. These may differ: an outcome whose window ends next month is not known merely because its row begins today. If true arrival times are unavailable, state the assumptions you vary. Do not call an assumption-driven difference proof of historical leakage.

**Hold fixed:** full query identities, targets, data release, model settings, validation-only selection, and the paired labeled-support schedule. Both models receive the same permissible labeled information within a policy. Only the eligibility policy changes. If that contract cannot be maintained, stop before running models.

[[FIG:temporal]]

**Worked example — invented AUROC values.** Under timestamp filtering, tree = 0.70 and PFN = 0.74: PFN advantage = 0.04. Under the availability policy, tree = 0.71 and PFN = 0.72: advantage = 0.01. The policy changes the advantage by **0.01 − 0.04 = −0.03**. This is a *difference of differences*: one model contrast minus the corresponding contrast under another condition.

**What would refute material sensitivity?** A justified interval wholly inside (−0.01, +0.01). An interval wholly above +0.01 or below −0.01 supports a material change. Crossing or touching a boundary is inconclusive under our conservative teaching rule. The uncertainty procedure is still an execution-protocol requirement; the picture does not supply one.

**Apply the rule at the boundary.** Keep the illustrative sensitivity margin fixed at 0.01 and vary only an externally justified interval for the policy contrast:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:3px"><thead><tr><th>Interval</th><th>Decision</th><th>Reason</th></tr></thead><tbody><tr><td>[−0.009, +0.009]</td><td>Below useful margin</td><td>Strictly inside the band</td></tr><tr><td>[−0.010, +0.009]</td><td>Inconclusive</td><td>Touches a boundary</td></tr><tr><td>[+0.011, +0.020]</td><td>Material sensitivity</td><td>Strictly above +0.010</td></tr></tbody></table>

These are hypothetical inputs to `interval_decision`, not uncertainty estimates for the proposed experiment. The function applies a declared rule; it cannot justify how the interval was estimated. Preserve sufficient numerical precision before checking the boundary: a rounded display is not the decision input.

**Transfer the trace.** Change only the last interval's lower endpoint to +0.010. The decision becomes inconclusive, despite the positive mean you might imagine inside it. This lesson deliberately uses strict boundaries; do not import the inclusive numerical-parity gates used elsewhere in the course.


[RelArena](https://arxiv.org/html/2608.16319v2) already provides standardized relational comparisons. The candidate contribution is a measured sensitivity study with documented information assumptions, not benchmarking itself or fixing one cache key.

## 3 · Direction two: separate the prior from the encoder

**Question.** Does composite message passing add a useful benefit under a relational prior, and is that benefit larger than under a single-table prior?

[RelGNN](https://arxiv.org/html/2502.06784v2) already introduces composite message passing. [RDB-PFN](https://arxiv.org/html/2603.03805v5) already learns from synthetic relational tasks and uses DFS-derived features. Here *DFS* means deep feature synthesis: constructing flat features by aggregating related rows. Changing the synthetic generator changes the learning experience; changing the encoder changes the predictor. Changing both at once hides which change mattered.

**A factorial design crosses two interventions.** Train every combination of two priors and two encoders. Keep the prediction head, labeled supports, permissible path endpoints, selection budget and compute accounting matched. A conventional two-hop encoder must see the same reachable information. The proposed graph encoder and ICL head need training from compatible initialization; fitting an embedding width does not make a released checkpoint valid.

<div id="interaction-predict"></div>
<noscript><p>Predict: does a positive composite gain under the relational prior guarantee positive interaction? No; its gain may be larger under the other prior.</p></noscript>

[[FIG:composite]]

**Worked example — invented AUROC values.** The composite gain is 0.74 − 0.70 = 0.04 under the flat prior and 0.73 − 0.72 = 0.01 under the relational prior. The *interaction* is **0.01 − 0.04 = −0.03**. Composite helps both, but this example does not support extra benefit from combining it with the relational prior.

Two questions therefore remain separate: is the relational-prior conditional gain usefully positive, and is the interaction positive? A positive interaction alone is insufficient if both encoders perform worse. Show all four absolute scores and both contrasts.

**Complete accounting.** Two priors × two encoders × three initialization seeds gives 12 checkpoints. Evaluate each on two tasks and three support draws: **72 primary prediction batches**, corresponding to 24 checkpoint/task combinations. Add six released-checkpoint comparator batches and 18 tree fits/batches. A *support draw* chooses labeled examples for in-context prediction; it is not another independent database. These units count different operations and must be costed separately.

## 4 · Direction three: does pretraining beat spending the budget on the target?

**Question.** Do historical-relation or future-activity objectives improve a wholly held-out database beyond an equally expensive supervised baseline?

A *historical-relation objective* learns to recover eligible past links. A *future-activity objective* learns outcomes over a later window using only source examples whose label windows are complete. A *database holdout* excludes the entire target database from source training and source preprocessing. Excluding target labels alone is insufficient.

[[FIG:transfer]]

**Worked example — invented AUROC values.** Ordinary scratch training scores 0.70. Extra-compute scratch scores 0.73. A pretrained arm scores 0.74. Its apparent 0.04 gain over ordinary scratch becomes **0.01 over the decisive control**. Source training, adaptation, preprocessing and selection all belong in the total budget. Equal update counts do not guarantee equal computational cost.

**Complete accounting.** Two held-out databases × two objectives × three seeds requires 12 source fits. Two target tasks × four arms × three seeds requires 24 target fits. The four arms are scratch, extra-compute scratch, historical-pretrained and future-pretrained. Report each objective/task separately; freeze any multiple-comparison policy before seeing outcomes.

An upper uncertainty bound below the proposed +0.01 useful-gain margin rules out that useful benefit in the declared setting. An interval crossing the margin remains inconclusive. A failed exclusion or cost audit invalidates the comparison before statistical interpretation. Two databases still form a narrow study.

## 5 · Decide what an interval actually says

An *estimand* is the quantity the experiment intends to estimate: here a paired gain or difference of differences. An *interval* describes uncertainty under a specified procedure and sampling unit. A sampling unit might be a driver or a database; choosing it incorrectly can make certainty misleading.

> **In plain terms.** First decide what would matter. Then ask whether the evidence is precise enough to distinguish it from what would not matter.

<div id="interval-explorer"></div>
<noscript><p>Illustrative gain interval [−0.005, 0.025], useful margin 0.01: INCONCLUSIVE. Illustrative sensitivity interval [−0.005, 0.005]: BELOW_USEFUL_MARGIN. These are invented intervals, not fitted results.</p></noscript>

The explorer classifies supplied intervals; it does not estimate them. For the future studies, specify paired resampling, dependency groups, confidence level and multiplicity before execution. Do not treat three seeds as three databases. L197's driver bootstrap belongs to its original regression comparison and cannot be borrowed as the uncertainty model for these new tasks.

## 6 · Rank the next useful decision, not a promised breakthrough

Impact means how much answering the question could change a research decision, including a negative answer. Feasibility concerns the **next informative step**, scored separately for data access, implementation and compute. These are authored judgments on a 1–5 scale, not measured probabilities.

The existing rubric is **impact × weighted mean feasibility**. With equal weights, temporal = 4 × (5+4+5)/3 = 18.67; composite = 5 × (4+3+2)/3 = 15; transfer = 5 × (3+2+1)/3 = 10.

<div id="proposal-priority"></div>
<noscript><p>Default priority: temporal 18.67, composite 15, transfer 10. Reduce temporal impact from 4 to 3: temporal becomes 14 and composite leads. All full-run cost bounds remain unverified.</p></noscript>

All 27 weight triples from {1,2,3}³ retain temporal as the leader at the original scores. This does not establish robustness to every assumption: reducing temporal impact to 3 changes the equal-weight leader to composite. Ties must remain ties.

**Why these scores?** Temporal uses existing local audit paths, although arrival history remains uncertain. Composite needs a new trainable interface and twelve pretrained checkpoints. Transfer also needs a source corpus and auditable whole-database exclusions. The complete cards record the original rationale and preparation-effort estimates; these are estimates, not measured runtimes.

**Priority does not establish affordability.** Include preparation, training, selection, evaluation, retries and validation. Every full experiment currently has unknown phases, so all remain `NOT_ESTABLISHED` against the standing $10 cap. Unknown expense is not zero. L198 itself executes only the approved $0 proposal/evidence audit.

## 7 · Inspect the complete proposal cards

Each card preserves its original evidence, explains what the later lessons add, and lists exact execution gaps. A complete proposal can honestly contain unresolved implementation details. That makes it reviewable, not ready to dispatch.

[[CARDS]]

## 8 · What the complete reproduction establishes

[[RESULTS]]

The audit starts from original frozen inputs, not summary scores. It re-parses every selected published table cell, recomputes comparison pools and ranks, re-scores every saved prediction using complete query keys, and preserves original intervals. Independent checks use different arithmetic and joins. All prior discrepancies and missing fresh scores survive.

> **Scope check.** This is full reproduction of the declared **proposal/evidence audit**. It is not fresh model training, independent replication, whole-paper reproduction, or proof of novelty. The RDBLearn result remains `INCOMPLETE_SOURCE_PREPROCESSING_GATE`. Earlier incomplete learner exits remain incomplete.

## Lab · make the proposal executable as a decision

[Student notebook](../labs/0198-three-research-directions.ipynb) · [Executed walkthrough](../labs/html/0198-three-research-directions.html) · [Solution](../labs/solutions/0198-three-research-directions.ipynb) · [Portable reproducer](../labs/evidence/l198/reproducer.zip) · [Printable reference](../reference/three-research-directions.html) · [Exact protocol](../labs/l198-reproduction.md).

Implement three live functions: paired contrasts, interval decisions and complete matrix expansion. Your functions feed the proposal audit; incomplete matrices and unsupported conclusions fail checks. All load-bearing replay code is visible. The portable packet contains the full original evidence and independent verifiers.

**Your writing task.** Draft three 250–400 word proposals using the [blank template](../labs/evidence/l198/proposal-template.md). For each, cite the closest work; distinguish the candidate contribution; state a falsifiable hypothesis, matched controls, full run count, uncertainty plan, unresolved fields and next decision. Rank them with one justified change to the authored assumptions. Name an observation that would reverse your ranking. Do not select the final direction yet.

<div id="proposal-teachback"></div>

**Exit evidence:** your three cards, a reproduced audit report, your revised ranking and a written defense. Automated checks verify arithmetic and structure, not scientific originality or understanding. Your status stays `PENDING_WRITTEN_DEFENSE` until your work is reviewed.

**Primary reading:** [Temporal Heterogeneous Graph Pretraining, §4.4](https://arxiv.org/html/2609.35219v1#S4.SS4). Explain exactly how its stated limitation differs from the transfer experiment above. Then read the closest source for your preferred candidate before defending it.

**Ask the agent follow-up questions.** Bring one proposal or an unclear control for critique. Tomorrow, reconstruct the three contrasts without notes. In a week, find the strongest primary-source objection and revise one card. Practitioner feedback can help test the novelty argument later; this lesson sends no external messages.
