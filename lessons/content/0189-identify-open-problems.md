> **Under construction.** This is a reviewable draft. Its saved audit and ranking are available, but the lesson is not certified complete. [Lesson 190’s research-gap checkpoint](0190-research-gap-checkpoint.html) is now available as a worked synthesis.

## Your tangible win

Turn “this looks promising” into **three ranked, falsifiable research questions**, then defend the next inexpensive decision. This is how the course mission becomes a research plan: evidence that relational structure adds value must survive strong baselines, time constraints and a finite budget.

**15-minute core:** read sections 1–4 and change one judgment in the explorer. **Optional 30-minute lab:** implement three contracts and replay the entire frozen audit. The detailed research cards are reference material for Lesson 190, not three experiments to run today.

[[STATUS]]

## Recall before reading

Without opening your notes, answer: What distinguishes a support-size sweep from a pretraining scaling law? Why must a prediction be keyed by both entity and cutoff? Why does a within-database gain not establish transfer to an unseen database?

**Prerequisite recap.** An entity can appear at several prediction times, so its ID alone is not a query identity. A *baseline* is the comparison method under the same information and selection rules. *Pretraining* learns before the target task; a *database holdout* excludes the entire target database from that stage. These distinctions connect [L169’s context sweep](0169-scaling-laws-open-questions.html), [L177’s cost accounting](0177-compute-budget-realism.html) and [L182’s composite hypothesis](0182-rdb-pfn-composite-message-passing.html). [L188’s literature-tracking collection](0188-systematic-literature-tracking.html) is still incomplete in our frozen receipt; this lesson does not assume it established coverage.

## 1 · A topic is not yet an open problem

“Build a relational foundation model” names a direction. A research question specifies **what changes, what stays fixed, which population matters, and which result would change your mind**. An open problem also needs a careful account of what the closest work already establishes.

Use the [RDL survey, §§2.4 and 5](https://arxiv.org/html/2506.16654v1) as a map of temporal, heterogeneous and foundation-model challenges. It was written in 2025; it cannot certify that an idea is new in October 2026.

[[PREDICT]]

**Worked narrowing.** “Does temporal pretraining help?” is already studied. The [September 2026 temporal-pretraining paper, §4.4](https://arxiv.org/html/2609.35219v1#S4.SS4) leaves unseen-database transfer and matched computational cost for further work. Narrow the question to: *Does a temporal objective beat extra-compute supervised training on an entirely held-out database at equal total cost?* This is a source-backed candidate gap. RT already studies [cross-database transfer](https://arxiv.org/html/2510.06377v1), so cross-database transfer itself is not the proposed novelty.

[[FIG:trace]]

A source-stated limitation is a lead, not an exhaustive literature review. We read six selected primary sources and freeze their versions. **Novelty remains `NOT_ESTABLISHED`** for all three candidates.

## 2 · Three questions, three different interventions

| Candidate | Change | Keep fixed | First useful decision |
|---|---|---|---|
| Temporal validity | Explicit availability policy | Query identities, labels, model/selection rules | Can every input’s availability be defended? |
| Composite structure | Conventional versus composite encoder | Prior, context, ICL head, compute | Is the proposed graph/head interface trainable? |
| Cross-database pretraining | Source objective and initialization | Target holdout, information, all-in cost | Is there a complete, affordable exclusion/cost contract? |

**Temporal validity.** L184’s frozen report records 7,838 overwritten query entries across 8,712 queries when indexed by entity alone. That is a local source finding; fixing it is engineering. The research question asks whether *matched model comparisons* are sensitive to documented availability policies across tasks. If true historical arrival times are unavailable, the result concerns assumptions—not proof of historical leakage. Standardized evaluation already exists in [RelArena-α](https://arxiv.org/html/2608.16319v2).

**Composite structure.** [RelGNN](https://arxiv.org/html/2502.06784v2) already provides composite message passing; [RDB-PFN](https://arxiv.org/html/2603.03805v5) already uses relational synthetic priors and DFS. Changing a synthetic generator changes the task distribution. Changing an encoder changes the predictor. A factorial design separates them; fitting an embedding width does not make a released checkpoint compatible. L182’s 21,060 saved-prediction claim belongs to its released-model comparison, not to a hybrid.

**Pretraining transfer.** A good supervised graph transformer is not evidence that pretraining helps it. Compare against both ordinary scratch training and scratch training allowed the extra compute. Keep the target database out of source preprocessing as well as source gradients. Two target databases remain a narrow study; do not turn repeated seeds into extra databases.

The full cards below state the hypotheses, controls, proposed populations, falsifiers, costs and next decisions. **None of their model experiments has been run in L189.**

## 3 · Rank decisions with an explicit rubric

Rate impact and feasibility from 1 (low) to 5 (high). Impact means the value of resolving the question, including a negative result. Feasibility has three dimensions: data access, implementation effort and compute. Here feasibility rates the **next useful decision**; full-experiment cost passes through a separate gate.

| Score | Data access | Implementation | Compute for next decision |
|---:|---|---|---|
| 1 | Critical access unresolved | Major new system | Large unbounded run |
| 3 | Public but needs substantial audit | New integration | Bounded pilot needed |
| 5 | Local inputs available | Existing audited path | Small local audit |

Scores 2 and 4 lie between these anchors. Impact anchors: 1 resolves a narrow local inconvenience; 3 changes one useful comparison; 5 could influence a broader model-design decision. These ordinal judgments are not measured probabilities. The arithmetic is a declared decision aid, not a scientifically calibrated utility model.

**Priority = impact × (w_data × data + w_impl × implementation + w_compute × compute) / sum(weights).**

Temporal validity has impact 4 and feasibility (5,4,5). Equal weights give **4 × 14/3 = 18.67**. Composite structure has 5 × (4+3+2)/3 = **15.00**. Pretraining transfer has 5 × (3+2+1)/3 = **10.00**. Estimated researcher effort to prepare their minimum protocols is respectively **8–16, 24–48 and 40–80 hours**; these are planning judgments, not measured completion times or compute prices.

**Predict first:** which ranking changes if temporal impact drops from 4 to 3? Try it. Then put compute weight at 3. A tie must remain visible.

[[PRIORITY]]

[[FIG:ranking]]

On a phone, the ranking plot scrolls horizontally; the explorer and results table give the same values without panning.

The executable audit tries **all 27 weight triples from {1,2,3}³**. Temporal validity leads all 27 at the authored scores. This only shows stability to those weights. At equal weights and temporal impact 3, its score becomes 14, so composite structure leads at 15. Changing the assumptions can change the decision.

## 4 · Priority is not permission to run

Unknown expense is not zero. Count preparation, every training arm and seed, selection, full evaluation, retry reserve and validation. A proposal with any missing phase or an unverified bound stays `NOT_ESTABLISHED`; a known total above $10 is `OVER_CAP`. A failed source audit blocks interpretation even when the arithmetic looks affordable. Passing the numerical gate still does not authorize execution.

All three full experiments have unknown cost phases and unfinished execution protocols. The third, for example, needs **12 source fits plus 24 target fits**, not one convenient fine-tuning run. The first decision can be inexpensive even when the eventual experiment cannot fit $10.

**Falsification example.** Before collecting new results, define a useful conditional gain, here a proposed 0.01 AUROC. An interval entirely below that threshold rules out the specified useful gain under the stated uncertainty model. An interval crossing it is inconclusive. A noisy mean or a non-significant p-value is not proof of no effect. Freeze the interval procedure, clustered resampling unit and split policy in the future execution protocol; they are not resolved by this planning audit.

[[CHECKLIST]]

## Detailed research cards

[[CASES]]

## Lab · reproduce the decision, then challenge it

[Student notebook](../labs/0189-identify-open-problems.ipynb) · [Executed walkthrough](../labs/html/0189-identify-open-problems.html) · [Solution notebook](../labs/solutions/0189-identify-open-problems.ipynb) · [Printable reference](../reference/identify-open-problems.html) · [Reproduction contract](../labs/l189-reproduction.md)

Implement three live functions: a complete-cost gate, the priority formula, and a tie-preserving 27-setting sweep. The notebook embeds the frozen sources and receipts, all load-bearing audit code, and portable figures. It authenticates all 16 files and audits every case. It does **not** re-score inherited prediction arrays or run models. SHA256 detects changed bytes, not a scientifically correct claim.

[[RESULTS]]

## Your written defense

[[TEACHBACK]]

Submit the ranked shortlist with one revised assumption and its reason. For each question, name the closest work, the useful effect, the complete comparison, a stop condition, and the cheapest informative next decision. The automatic checks cannot certify your novelty argument or understanding.

**Tomorrow:** reconstruct the priority formula and explain why an unknown cost is not zero. **In one week:** reread the nearest work, change one justified score, and defend whether the shortlist changes. This becomes the three-problem spine of the [Lesson 190 research-gap document](0190-research-gap-checkpoint.html), not a claim that its checkpoint is complete.

**Primary reading:** the [RDL survey](https://arxiv.org/html/2506.16654v1), then the temporal-pretraining paper’s [limitations](https://arxiv.org/html/2609.35219v1#S4.SS4). For practitioner feedback, prepare a concise related-work comparison for the [RelBench project](https://github.com/snap-stanford/relbench) community; posting is your choice. Ask the agent follow-up questions about any hypothesis, baseline, or budget assumption you cannot yet defend.
