# A gap between recorded time and usable knowledge

**Research-gap document · worked author example · 2 October 2026**

## The question

On a fixed relational autocomplete task, how much does measured information availability change the apparent value of relational context? The question is narrower than “Do relational models work?” It asks whether a specific comparison survives when a predictor can use only information actually available at its query cutoff.

The course mission is to test whether relational learning unlocks value that strong single-table pipelines miss. That argument requires a credible information contract. A sophisticated model cannot repair a comparison whose two arms see different information.

## Why this is a candidate gap

[RelBench v2](https://arxiv.org/abs/2602.12606) introduces temporally constrained autocomplete. The inherited L181 report records a training-health blocker for its selected GNN lane. L184 records a source/temporal gate. These are local evidence boundaries, not proof that the research community has neglected availability.

Our proposed contribution would be a **paired availability audit and controlled comparison**, conditional on obtaining arrival histories and a healthy predictor. The contribution is not a new architecture by declaration. If related work already supplies the same contract and experiment, the novelty claim must be narrowed or abandoned.

## What this document establishes

The accompanying L190 replay authenticates saved artifacts, reconstructs a complete selected model comparison, and reproduces incomplete literature coverage. It also makes an authored priority rubric inspectable. It does not run a new model or establish novelty.

**Decision:** investigate availability first, because the question is precise and the next missing input is identifiable. Do not launch the proposed experiment until its data, implementation and cost gates pass.

<footer>Page 1 / 5 · Scope: a falsifiable candidate, not an established contribution.<br>Evidence: frozen L181–188 status receipts; numerical replay defined on page 2.</footer>

<!-- PAGE -->
# What the evidence actually says

[[RESULTS]]

The named comparison is RDB-PFN v5 Table 9, F1/driver-dnf: three released models, ten paired support draws, 512 support rows and all 702 test queries. The L190 replay authenticates and re-scores **21,060 saved predictions**. It verifies query/label identity, complete run coverage, paired support sets and support/test separation against released prepared arrays.

The RDB-PFN minus TabICL paired mean is **0.004369 AUROC**, positive in **6 of 10** draws. The plot shows every paired difference. These draws reuse one task and test population; their dispersion is not independent cross-database uncertainty.

[[FIG:paired]]

The comparison supports a narrow descriptive sentence about these saved predictions. It does not test a composite hybrid. The [RDB-PFN source](https://arxiv.org/html/2603.03805v5) uses a relational synthetic prior and a DFS interface; [RelGNN](https://arxiv.org/html/2502.06784v2) already supplies composite message passing. An interaction between the two remains a proposal.

**Literature coverage:** raw L188 responses yield **31 records, 30 unique papers, and 2 of 4 completed searches**. Four failed attempts remain in the packet. Successful replay preserves the incomplete search; it does not turn missing results into evidence of absence.

**Boundaries:** released labels/DFS are trusted inputs here; raw label reconstruction, historical arrival histories, new inference and pretraining are not performed. Other lesson reports provide attributed status receipts only; their numerical experiments are not replayed in L190.

<footer>Page 2 / 5 · Primary target: RDB-PFN v5 Table 9.<br>Audit: labs/evidence/l190/report.json and labs/_verify_l190_results.json. Exact files and origins: input-manifest.json and packet/origins.json.</footer>

<!-- PAGE -->
# Three questions worth separating

[[RANKING]]

**Rubric:** impact × weighted mean of data, implementation and compute feasibility. Each authored score is ordinal, from 1 to 5; larger feasibility means easier. Default weights are equal. Across all 27 triples from {1,2,3}, availability remains first. Weight stability does not validate the assigned scores or predict scientific success.

This checkpoint substitutes joint serving/privacy constraints for L189’s pretraining-transfer candidate. Its shortlist and composite contrast are separate authored choices, not a replay of L189’s ranking.

## 1 · Availability-aware autocomplete

**Hypothesis:** relational-context benefit shrinks under measured arrival-time visibility. Compare the same predictor and queries under an event-time policy and an availability policy. Existing keys and audits lower implementation effort; absent arrival histories can still block the whole study. A null or reversed effect would weaken the proposed failure mode.

## 2 · Composite structure in a foundation learner

**Hypothesis:** composite routes interact positively with relational pretraining. Use a matched four-arm design: conventional/composite encoder × scratch/pretrained initialization. Compare the interaction, not only whether the combined arm wins. Architecture changes may invalidate checkpoint reuse; no compatible pretrained hybrid is established by our saved scores.

## 3 · Accuracy under serving and privacy constraints

**Hypothesis:** a relational pipeline retains useful accuracy while meeting fixed freshness, latency and entity-level privacy requirements. Compare it with a strong flattened baseline under the same workload and accounting. L186's course simulation and L187's privacy experiment motivate requirements, but do not establish joint real-world performance. Any mandatory constraint failure defeats the joint claim.

**Choice:** availability first; composite structure second; joint constraints third. All three retain `NOT_ESTABLISHED` novelty and aggregate execution cost. These rankings are an authored decision aid, not permission to train.

<footer>Page 3 / 5 · Controlled score assumptions: packet/cases.json.<br>The notebook exposes all 27 weight scenarios and preserves ties. Revisit scores after obtaining disconfirming evidence.</footer>

<!-- PAGE -->
# An experiment that could change our mind

## Proposed paired study · not executed

**Task:** the L181 F1 results-position autocomplete contract. Preserve the released train/validation/test split, complete query population, target masking and proxy exclusions. Pin the task, database, source and preprocessing hashes before any fitting. Require recorded availability timestamps for every consumed feature; missing histories block the historical study.

**Arms:** a strong flattened baseline and a relational predictor, each under (A) event-time visibility and (B) event-time plus measured availability visibility. Fit each policy using its own permitted inputs. Keep architecture, preprocessing rules, tuning allowance and validation selection paired across A/B; missing context follows one declared rule. Repair and re-audit the inherited training-health issue first. Exact compatible model configuration remains an admission deliverable, not an implied fact.

**Replicates:** prospective seeds 0, 1 and 2 for every arm. Select checkpoints on validation only; evaluate every held-out query once after freezing choices. Retain per-query predictions, full keys, visibility masks, per-seed checkpoints, failed attempts and all preparation/training/validation costs.

## Estimand and decision

Let benefit under policy P be `MAE(flat,P) − MAE(relational,P)`. The primary contrast is `Δ = benefit(A) − benefit(B)`. Positive Δ means the apparent relational advantage shrank when measured availability was enforced. Pair queries and training seeds throughout.

Before execution, adopt **0.10 finishing-position units** as an illustrative course threshold for a useful shrinkage, and justify or revise it with the intended stakeholder. Use 1,000 paired driver-cluster bootstrap resamples, RNG seed 19001, retaining all cutoffs for each sampled driver. Report per-seed results separately; the interval describes sampled-driver variation on this task, not database transfer.

**Interpretation rule:** lower 95% interval bound above 0.10 supports the declared useful shrinkage; upper bound at or below 0.10 fails that criterion; otherwise the result is inconclusive. A negative Δ challenges the direction. Do not replace missing histories with fabricated dates. Synthetic delays may support a separately labelled mechanism experiment.

<footer>Page 4 / 5 · Design only: NOT_RUN. Threshold, bootstrap and seeds are proposed course choices.<br>Cost admission: source/data/health checks, exact configuration and all-phase forecast must precede separately approved execution.</footer>

<!-- PAGE -->
# Limits, gates and the next decision

## Four limits that remain visible

**Scientific coverage:** the quarterly search is incomplete. Inspect disconfirming prior work and its methods before claiming novelty. A local source bug or absent artifact does not prove an open scientific problem.

**Historical validity:** a released event timestamp is not an ingestion log. The current replay does not reconstruct raw labels or DFS, and it cannot prove what a historical predictor knew. Arrival evidence is the first admission gate for the proposed study.

**External validity:** the saved comparison uses one fixed task and population. Support draws measure sensitivity to supplied examples. They are neither new databases nor independent benchmark replications.

**Execution readiness:** exact compatible model configuration, healthy training and an aggregate cost forecast remain unresolved. A high ranking cannot waive them. Serving/privacy simulations do not certify a live joint utility claim.

## Budget and stop rules

**Executed L190 lane:** saved-evidence replay and validation only; $0 new cloud/API spend; 1,800 aggregate local execution seconds including failed attempts, notebook execution and checks. Stop and preserve `INCOMPLETE` if the cap is reached. The mutable attempt ledger is separate from the immutable evidence packet.

**Future study:** standing aggregate cap about $10, including preparation, all arms/seeds, selection, evaluation, retries and verification. Cost is currently `NOT_ESTABLISHED`; this document authorizes no paid run. Do not reduce the design after seeing costs or results while retaining the original claim.

## Checkpoint verdict and learner task

The selected evidence replay is complete. The research checkpoint remains **INCOMPLETE**: novelty, experiment admission and the learner's written defense are separate gates.

Submit your own five-page argument. Replace the author judgments with justified choices; identify the strongest existing counterexample; explain what result would reverse your ranking. The teacher must review the reasoning. A notebook pass cannot establish research judgment or personal mastery.

**Next cheap decision:** determine whether genuine arrival histories can be obtained under the task contract. If not, retain the historical study as blocked and explicitly propose a narrower mechanism study for separate approval.

<footer>Page 5 / 5 · Learner: PENDING_WRITTEN_DEFENSE.<br>Ask the teaching agent for feedback; revisit after one day and one week. Primary readings: RelBench v2 (2602.12606), RDB-PFN v5 (2603.03805v5), RelGNN v2 (2502.06784v2).</footer>
