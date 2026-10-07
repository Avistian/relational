# 199 · Choose a direction you can defend

<p class="eyebrow">Year 5 · Research decision · 15-minute core + 30-minute lab</p>

**Your win:** write one provisional research choice, explain what you are giving up, and name the first observation that would make you stop. The mission is to test relational learning's value with a credible experiment—not just select an appealing architecture.

[[STATUS]]





<details><summary>Check your reasoning</summary><p>A replay can verify old arithmetic while the proposed study still lacks data, healthy baselines or a feasible budget. Weight sensitivity leaves the assigned ratings fixed; changing those judgments can change the winner.</p></details>

**Where we are.** [L190](0190-research-gap-checkpoint.html) assembled candidate gaps; [L195](0195-thesis-stress-test.html) asked what would weaken the thesis; [L197](0197-year-5-essay.html) connected evidence to claims. [L198](0198-three-research-directions.html) now turns those gaps into three proposal cards. It appeared after this lesson's evidence freeze. Our approved audit keeps L190's cards as **provisional author examples**, not your completed proposals. Use your own L198 cards for the final selection memo. Earlier incomplete exits stay incomplete.

**Keep the two shortlists distinct.** L198 compares temporal-policy sensitivity across named models, prior × encoder interaction, and held-out-database transfer. The frozen L190 example uses measured-availability investigation, encoder × pretraining initialization, and joint serving/privacy constraints. Its ordinal scores do not rank L198's different experiments. Carry the decision method forward; rewrite the candidate set and justify fresh ratings for your own memo. L199's worked context-versus-target-row contrast is a proposed narrowing, not an already-approved execution of L198's protocol.

## 2 · A priority is not a launch decision

A **priority** says where to spend your next unit of investigation. A **mandatory requirement** says whether a particular experiment is executable. A **kill criterion** names the evidence that would make you abandon or redesign it. A high priority cannot compensate for data that do not exist.

[[FIGURE]]

Consider three candidates. Each still has unestablished novelty and future experiment cost. The statements below are proposals, not findings. [Frozen candidate cards](../labs/evidence/l199/packet/evidence/l190/packet/cases.json).

| Candidate | Falsifiable question | First unresolved requirement |
|---|---|---|
| Availability | Does relational-context benefit shrink under measured arrival visibility? | Real arrival histories linked to query cutoffs |
| Composite structure | Does composite message passing interact with relational pretraining? | Compatible, healthy four-arm implementation |
| Joint constraints | Does useful accuracy survive fixed freshness, latency and privacy constraints? | Measured workload and defensible privacy accounting |

**Vocabulary:** event time is when something happened; arrival time is when the predictor could actually access it. A late-arriving event can have an old event timestamp. The [RelBench v2 paper](https://arxiv.org/abs/2602.12606) motivates temporally constrained relational autocomplete; it does not by itself certify that a selected archive contains historical arrival times.

## 3 · Work one ranking by hand

The inherited course rubric is `impact × weighted mean(data, implementation, compute feasibility)`. Each authored rating runs from 1 to 5; higher feasibility means easier. These are ordinal judgments. Multiplying and averaging them is a transparent decision aid, not a calibrated measurement of research value.

| Candidate | Impact | Data / implementation / compute | Equal-weight priority |
|---|---:|---|---:|
| Availability | 4 | 5 / 4 / 5 | 4 × (5 + 4 + 5) / 3 = **18.667** |
| Composite | 5 | 3 / 2 / 4 | **15.000** |
| Joint constraints | 4 | 3 / 3 / 2 | **10.667** |

The inherited availability data rating of 5 is debatable when arrival histories are missing. We preserve it so you can inspect the original judgment; the independent data requirement remains UNKNOWN. Rewriting a rating silently would conceal the disagreement.

**Predict:** lower availability's impact from 4 to 3, leaving everything else unchanged. Its priority becomes 14. Which direction leads now?

<div id="direction-ranking"></div>
<noscript><p>With impact 4, availability leads 18.667 to 15. With impact 3, composite leads 15 to 14. All 27 original weight combinations favor availability; changing a rating can reverse the choice.</p></noscript>

The audit checks every original weight triple from {1,2,3}, then all **21 feasible one-step changes to one rating at a time** at equal weights. It does not cover every joint change or estimate a probability of success. Ties stay ties. [Full executable report](../labs/evidence/l199/report.json).

## 4 · Specify what would be learned

The author example prioritizes an **availability feasibility check**, not training. A later study could use one fixed learner family in four matched arms:

| Information policy | Target-row-only control | Related-row context added |
|---|---|---|
| Event time | Eligible target-row fields | Eligible target fields + related rows |
| Measured arrival time | Same fields, arrival-filtered | Same context definition, arrival-filtered |

A target-row-only control is explicit. A flat model using relational aggregates already sees related information. This contrast tests the added information under two policies; it does not isolate graph architecture.

For a higher-is-better metric, define **D = (context − control)event − (context − control)arrival**. Positive D means the apparent context advantage shrank under actual availability. **Sign check with L198:** its temporal example subtracts event from availability; here we subtract arrival from event so shrinkage is positive. Reversing the same two contrasts reverses the sign. The comparator also changes from PFN versus tree to context versus target-only input, so these lessons propose different quantities to measure. Keep task, labels, query identities, splits, learner family and tuning budget matched. Select models on validation, then evaluate untouched outcomes.

**Illustration only:** event scores 0.78 and 0.74 give benefit 0.04; arrival scores 0.75 and 0.73 give benefit 0.02. Thus D = 0.02 AUROC, or 2 percentage points. These four values are invented arithmetic, not experiment results.

Before the new evaluation, justify a useful-effect margin δ and a dependence-aware uncertainty method. An interval wholly above δ supports practically meaningful shrinkage; one overlapping δ is inconclusive; one wholly below δ weakens that prespecified useful-shrinkage claim. δ = 0.01 could illustrate this rule but is **not an approved scientific threshold**. Repeated entities, shared times and seeds do not constitute independent databases.

**Repair an inherited shortcut.** The old composite card calls an interaction interval including zero a falsifier. Zero overlap alone is inconclusive. To argue against a useful positive interaction, define its margin and obtain sufficient precision. We retain the old source bytes and state the correction here.

## 5 · Ask for the missing evidence

<div id="direction-gates"></div>
<noscript><p>All four current requirements are UNKNOWN: data, baseline, design and budget. Therefore DO_NOT_LAUNCH. All PASS would mean READY_FOR_REVIEW, not authorization.</p></noscript>

For this exact future experiment, none of these requirements has yet been admitted:

- **Data:** measured arrival histories, documented semantics, sufficient coverage and stable complete query keys.
- **Baseline:** healthy reproducible implementations with matched information and validation-only selection.
- **Design:** exact task, falsifiable contrast, justified margin and uncertainty unit, fixed before new test inspection.
- **Budget:** the complete aggregate forecast, including preparation, all seeds, retries, validation and overhead, with a reserve and cutoff.

**First discriminating step:** establish whether a usable arrival-history sample can be obtained and linked correctly. If it cannot, stop this historical-availability study and reopen the alternatives. Synthetic delays could answer a different mechanism question; they cannot manufacture the missing history.

**Opportunity cost:** spending your next week chasing unavailable arrival histories may delay a feasible composite experiment. Record a bounded feasibility effort and its stopping rule before spending it. The L199 audit's $0 cloud/API and 1,800-second cap cover this teaching package's execution; the future research cost is still unknown.

## 6 · What did we fully reproduce?

[[REPLAY]]

The L190 replay covers the released [RDB-PFN v5 Table 9](https://arxiv.org/html/2603.03805v5) comparison: three methods × ten support draws × all 702 queries, with 512 support rows. The mean RDB-PFN minus TabICL difference is **0.004369 AUROC**, positive in 6/10 draws. **That comparison contains no arrival-policy intervention.** It supports a scoped saved-prediction statement, not the proposed hypothesis.

Original support sets, label identity, temporal separation and complete `(entity_id, cutoff)` keys are checked. Released labels and DFS features remain trusted source inputs. Independent pairwise AUROC, raw XML and rational ranking checks guard against merely copying a report. L194 and L197 are authenticated status receipts here; their experiments are not rerun. [Exact protocol and commands](../labs/l199-reproduction.md).

## 7 · Make the choice yours

Write **400–700 words**: one direction, one testable contrast, the strongest reason to prefer another direction, and a concrete reopening condition. An honest choice may be “investigate X first; launch blocked.” Choosing no currently executable study is better than pretending a requirement passed.

<div id="direction-memo"></div>
<noscript><p>Use the linked Markdown template to write your twelve-field selection memo. Field completion is only a readiness check, never mastery.</p></noscript>

[Blank memo](../labs/evidence/l199/memo-template.md) · [Student notebook](../labs/0199-select-primary-direction.ipynb) · [Executed solution](../labs/html/0199-select-primary-direction.html) · [Portable reproducer](../labs/evidence/l199/reproducer.zip) · [Quick reference](../reference/select-primary-direction.html).

**Lab:** implement three functions—priority with ties, mandatory admission with UNKNOWN, and memo presence checks. Then run the complete replay using your functions. Deliberately remove one saved run and explain the rejection. Recompute the impact-rating reversal. Finally write your memo before opening the author example.

<details><summary>Read the worked author memo after your own draft</summary>

[[MEMO]]

</details>

**Primary reading:** [COS preregistration guidance](https://www.cos.io/initiatives/prereg). A prospective plan helps distinguish planned confirmation from exploration. Our old replay is already observed evidence; a local memo does not retrospectively make it confirmatory, and no public registration has been submitted.

**Exit check:** Can another person identify your exact comparison, its unresolved gates, what would weaken your hypothesis, and why the deferred alternative could be better? Send your memo or any unclear step to the agent for follow-up. Filling boxes does not establish mastery: `PENDING_WRITTEN_DEFENSE`.

**Revisit:** tomorrow reconstruct D from memory; in seven days argue for the strongest alternative; in thirty days revisit the choice only with new evidence or an explicitly changed judgment. [Lesson 200](0200-year-5-exit-exam.html) supplies a separate, fresh RDB-PFN checkpoint reproduction and asks you to defend it alongside your proposal. That selected result does not complete the blocked RDBLearn search; your proposal and defense still require review.
