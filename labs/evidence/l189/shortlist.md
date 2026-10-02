# L189 ranked research shortlist

**Author evidence:** `COMPLETE_SELECTED_AUDIT` — 3 case files, 6 primary sources, 6 inherited receipts, all 27 weight settings. Prediction replay, fresh training and whole-paper reproduction: `NOT_RUN`. Novelty and complete literature coverage: `NOT_ESTABLISHED`. Learner: `PENDING_WRITTEN_DEFENSE`.

| Rank | Question | Equal-weight priority | Full-run cost |
|---:|---|---:|---|
| 1 | Which availability assumptions change the model comparison? | 18.67 | NOT_ESTABLISHED |
| 2 | When does a composite operator help a relational foundation learner? | 15.00 | NOT_ESTABLISHED |
| 3 | Do temporal objectives transfer under database holdout and equal cost? | 10.00 | NOT_ESTABLISHED |

These are authored priorities reproduced by code, not measured model performance. [Frozen audit report](../../evidence/l189/report.json) · [Input hashes](../../evidence/l189/input-manifest.json).

### Which availability assumptions change the model comparison?

**Question.** Across rel-f1/driver-dnf and rel-trial/study-outcome, does replacing timestamp-only filtering with an explicit availability contract change paired model contrasts?

**Hypothesis.** A material comparison change persists after preserving every (entity_id, cutoff) query and auditing when each feature and label becomes observable.

**Already known.** Chronological evaluation and temporal-aware models already exist. L184 reports repeated-entity cache collisions in one source snapshot. L169 context curves do not measure pretraining scaling.

**Candidate gap.** The proposed contribution is sensitivity of a matched comparison to documented availability assumptions across tasks. Fixing a cache key alone is an engineering repair, not the research claim.

**Approach.** Represent every query by its complete key; attach evidence for each context cell availability and label-window completion; compare strict known-availability inclusion against timestamp-only inclusion. Treat unknown arrival histories as explicit assumptions.

**Matched control.** Same cached data release, targets, splits, feature definitions and validation-only selection for RDB-PFN and a flattened tree baseline; two visibility policies. Reuse identical support rows and seeds across policies, with any ineligible support reported before running.

**Minimum full comparison — proposed, unrun.** 2 tasks × 2 models × 2 visibility policies × 3 seeds = 24 complete fits/evaluations. Full validation/test populations; predeclare one configuration per model, no test tuning. Seal train/validation/test key universes and support schedules before scoring. Independently reconstruct targets first.

**Quantity to estimate.** For each task and seed, compute (AUROC_PFN − AUROC_tree)_strict − (AUROC_PFN − AUROC_tree)_timestamp. Report each task separately; seed spread is not database uncertainty.

**Falsifier and limitation.** A preregistered useful absolute change is 0.01 AUROC per task. An interval wholly inside [−0.01,0.01] argues against material sensitivity for that task; overlap is inconclusive. A missing arrival history prevents a historical leakage conclusion, even if policy sensitivity is large.

**Related work.** [survey](https://arxiv.org/html/2506.16654v1), [temporal](https://arxiv.org/html/2609.35219v1), [relarena](https://arxiv.org/html/2608.16319v2).

**Inherited receipts.** [l184-report.json](../../evidence/l189/packet/inherited/l184-report.json), [l169-report.json](../../evidence/l189/packet/inherited/l169-report.json), [l177-report.json](../../evidence/l189/packet/inherited/l177-report.json). Reports are authenticated; underlying predictions are not replayed here.

**Authored feasibility rationale.** Data 5: local full-key snapshots and audit receipts exist; actual historical arrival times remain unknown. Implementation 4: policy masks and keyed scoring are bounded; full DFS availability lineage still needs audit. Compute 5: first useful decision is a CPU key/availability audit, not new pretraining; complete model comparison cost is unverified.

**Human effort estimate:** 8–16 hours to prepare the protocol. **Full-run cost:** unverified; all six phases unknown. **Next decision:** Audit every query key and classify context timestamps as known, assumed or unknown; stop before model runs if the comparison cannot preserve a defensible information contract.



### When does a composite operator help a relational foundation learner?

**Question.** Does a composite encoder improve a schema-shared graph encoder plus ICL head beyond a conventional two-hop encoder when priors, labels, information and total compute are matched?

**Hypothesis.** The composite encoder has positive incremental benefit under a relational prior, beyond any benefit due to switching the pretraining prior itself.

**Already known.** RelGNN already contributes composite message passing. RDB-PFN uses synthetic relational priors and DFS before its predictor. L182 re-evaluates released predictors, not a trained composite hybrid.

**Candidate gap.** Separate a predictor-side operator intervention from a generator-side change. A new operator inserted into a released checkpoint is not automatically compatible or a valid experiment.

**Approach.** Train a graph encoder plus common ICL head from initialization. Cross single-table versus relational priors with conventional two-hop versus composite encoder; keep a separate released DFS+RDB-PFN comparator.

**Matched control.** Four factorial arms share task counts, hidden width/parameter accounting, optimizer/selection budgets, schema-sharing rules, labeled supports and permissible neighbors. Conventional two-hop control sees the same path endpoints. Add a flattened strong baseline before interpreting practical value.

**Minimum full comparison — proposed, unrun.** 2 priors × 2 encoders × 3 initialization seeds = 12 new pretraining checkpoints; each evaluated on 2 held-out real database tasks = 24 primary evaluations, plus the separately costed released and flattened baselines. Freeze target-schema exclusions, three support draws and every validation/test key; validate finite gradients before dispatch.

**Quantity to estimate.** Within each task/seed, (composite_relational − conventional_relational) − (composite_flat − conventional_flat). Also report composite_relational − conventional_relational. A positive interaction alone is insufficient if both architectures get worse.

**Falsifier and limitation.** Predeclare 0.01 AUROC as the useful conditional gain on each classification task. No positive conditional gain falsifies the claimed benefit in this setting; interaction ≤0 fails the stronger synergy claim. Uncertain estimates remain inconclusive; three seeds do not establish cross-domain superiority.

**Related work.** [relgnn](https://arxiv.org/html/2502.06784v2), [rdbpfn](https://arxiv.org/html/2603.03805v5), [relarena](https://arxiv.org/html/2608.16319v2).

**Inherited receipts.** [l182-report.json](../../evidence/l189/packet/inherited/l182-report.json), [l177-report.json](../../evidence/l189/packet/inherited/l177-report.json). Reports are authenticated; underlying predictions are not replayed here.

**Authored feasibility rationale.** Data 4: public task releases and generators exist, but target-schema exclusion must be audited. Implementation 3: route mechanism is demonstrated; a trainable schema-shared encoder/head and adapters are new work. Compute 2: twelve new pretrained checkpoints plus controls have no verified total under $10.

**Human effort estimate:** 24–48 hours to prepare the protocol. **Full-run cost:** unverified; all six phases unknown. **Next decision:** Freeze the insertion point and one exact graph/head interface, then run a local forward/gradient and parameter-budget probe; do not report that probe as an accuracy result.



### Do temporal objectives transfer under database holdout and equal cost?

**Question.** Do historical-relation and future-activity objectives improve performance on unseen databases beyond scratch and extra-compute supervised controls at the same all-in cost?

**Hypothesis.** Source-only temporal pretraining improves a held-out target task after accounting for source training, adaptation, tuning and preprocessing cost.

**Already known.** Temporal pretraining with supervised controls is already studied within databases. RT already studies cross-database transfer. L183 saved supervised predictions cannot establish a pretraining gain.

**Candidate gap.** The temporal paper §4.4 explicitly leaves unseen-database transfer and matched computational cost open. This is a source-stated limitation, not proof that no other paper addresses the combined question.

**Approach.** Use a shared schema-independent backbone; hold out each target database from all source objectives and source preprocessing. Contrast historical relation recovery and horizon-aware future activity; ensure every source label window finishes before its permitted horizon.

**Matched control.** Scratch, extra-compute supervised scratch, historical-objective pretraining and future-objective pretraining use the same target information and evaluation policy. Match total cost, not just update count. Report per-objective cost and adaptation cost separately.

**Minimum full comparison — proposed, unrun.** 2 whole target-database holdouts (F1 and trial) × 3 seeds × 4 arms = 24 target fits, plus 2 objectives × 2 holdouts × 3 seeds = 12 source pretraining fits. Source corpus, adapter, full task/query manifests, schedules and prices must be frozen before approval. No same-database transfer substituted.

**Quantity to estimate.** For each held-out database, AUROC_pretrained − AUROC_extra_compute_scratch at matched all-in cost. Report the two databases separately and treat them as a narrow feasibility study, not a general scaling law.

**Falsifier and limitation.** Predeclare useful gain 0.01 AUROC on each task. No gain over extra-compute scratch fails the useful-transfer claim there. Unverifiable database exclusion or cost matching invalidates interpretation; a positive seed mean alone is insufficient.

**Related work.** [survey](https://arxiv.org/html/2506.16654v1), [rt](https://arxiv.org/html/2510.06377v1), [temporal](https://arxiv.org/html/2609.35219v1).

**Inherited receipts.** [l183-report.json](../../evidence/l189/packet/inherited/l183-report.json), [l177-report.json](../../evidence/l189/packet/inherited/l177-report.json). Reports are authenticated; underlying predictions are not replayed here.

**Authored feasibility rationale.** Data 3: releases exist, but a decontaminated source/target corpus contract is unfinished. Implementation 2: source objectives, adapters and semantic sharing need a full integrated trainer. Compute 1: 12 source fits plus 24 target fits have no verified $10 bound; historical blocked runs caution against optimism.

**Human effort estimate:** 40–80 hours to prepare the protocol. **Full-run cost:** unverified; all six phases unknown. **Next decision:** Audit source/target database exclusion and construct an all-in cost envelope; reject the run if a complete target cannot fit the cap. A small mechanism example cannot replace the declared experiment.

