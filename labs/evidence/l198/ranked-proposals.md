# L198 — three ranked research proposals

Authored examples for critique, not learner submissions. Default ranking concerns the next useful decision; final selection belongs to L199.

### Temporal-policy sensitivity

**Question.** Across rel-f1/driver-dnf and rel-trial/study-outcome, does replacing timestamp-only filtering with an explicit availability contract change paired model contrasts?

**What is already known.** Chronological evaluation and temporal-aware models already exist. L184 reports repeated-entity cache collisions in one source snapshot. L169 context curves do not measure pretraining scaling.

**Candidate contribution.** The new contribution would be a measured change in matched model contrasts under declared availability policies, not the discovery of temporal filtering or a cache repair.

**What L197 adds and leaves open.** L197 provides a complete saved evidence inventory; it contains no results for these two availability policies. L196 shows why a source diagnostic alone cannot estimate accuracy harm.

**Refined hypothesis.** On each declared task, the absolute policy-induced change in the RDB-PFN minus tree AUROC contrast exceeds 0.01.

**Matched controls.** Same cached data release, targets, splits, feature definitions and validation-only selection for RDB-PFN and a flattened tree baseline; two visibility policies. Reuse identical support rows and seeds across policies, with any ineligible support reported before running.

**Estimand.** For each task and seed, compute (AUROC_PFN − AUROC_tree)_strict − (AUROC_PFN − AUROC_tree)_timestamp. Report each task separately; seed spread is not database uncertainty.

**Falsifier.** A justified paired interval strictly inside (-0.01, 0.01) rules out the proposed material change for that task. Entirely beyond either margin supports sensitivity. All boundary-crossing or boundary-touching intervals remain inconclusive.

**Complete accounting.** 24 model/policy/task/seed evaluations, each on complete validation and test populations. RDB-PFN is checkpoint inference; tree arms require fitting. Seed pairs share a frozen support schedule; all models use the same eligible labeled information within each policy.

**Cheapest next artifact.** Availability ledger for all query and feature identities, distinguishing known, assumed and unknown observation times.

**Declared matrix.** Every combination is listed in the machine-readable report; each axis below is frozen as a proposal.

- model evaluations: task ∈ {rel-f1/driver-dnf, rel-trial/study-outcome} × model ∈ {RDB-PFN, flattened-tree} × policy ∈ {timestamp, availability} × seed ∈ {0, 1, 2} = **24**.

**Unresolved before execution:**

- Exact source/data hashes and target reconstruction
- Full (entity_id, cutoff) train/validation/test manifests
- One validation-only selection rule and fixed search budget
- Per-task uncertainty resampling unit and coverage rationale
- Timed all-in cost bound including retries and validation
- Availability lineage for every feature and support label; historical arrival times may be unavailable
- Exact tree config, PFN checkpoint and shared label-budget/support policy
- Stop if paired eligible supports cannot be maintained; do not silently drop queries

**Original priority rationale:**

- Data 5: local full-key snapshots and audit receipts exist; actual historical arrival times remain unknown.
- Implementation 4: policy masks and keyed scoring are bounded; full DFS availability lineage still needs audit.
- Compute 5: first useful decision is a CPU key/availability audit, not new pretraining; complete model comparison cost is unverified.

**Preparation effort:** 8–16 human hours (authored planning estimate). All six full-run cost phases are unknown; budget NOT_ESTABLISHED.

**Closest work:** [survey](https://arxiv.org/html/2506.16654v1), [relarena](https://arxiv.org/html/2608.16319v2), [temporal](https://arxiv.org/html/2609.35219v1).

**Original receipts (reports only):** [l184-report.json](packet/evidence/l189/packet/inherited/l184-report.json), [l169-report.json](packet/evidence/l189/packet/inherited/l169-report.json), [l177-report.json](packet/evidence/l189/packet/inherited/l177-report.json). These receipts are authenticated; their raw experiments are not all replayed. The full L197 packet is replayed separately.

**State:** PROPOSAL_NOT_EXECUTION_READY; models NOT_RUN; novelty NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.


### Prior × encoder interaction

**Question.** Does a composite encoder improve a schema-shared graph encoder plus ICL head beyond a conventional two-hop encoder when priors, labels, information and total compute are matched?

**What is already known.** RelGNN already contributes composite message passing. RDB-PFN uses synthetic relational priors and DFS before its predictor. L182 re-evaluates released predictors, not a trained composite hybrid.

**Candidate contribution.** Isolate the predictor-side composite operator and its interaction with a relational prior. RelGNN already supplies the operator, and RDB-PFN already supplies relational synthetic pretraining.

**What L197 adds and leaves open.** L197 replays a small released-system advantage on study-outcome; that is neither a trained hybrid nor evidence that changing the encoder causes the advantage.

**Refined hypothesis.** The composite encoder gives a useful (>0.01 AUROC) conditional gain under the relational prior, and its gain exceeds its gain under the single-table prior.

**Matched controls.** Four factorial arms share task counts, hidden width/parameter accounting, optimizer/selection budgets, schema-sharing rules, labeled supports and permissible neighbors. Conventional two-hop control sees the same path endpoints. Add a flattened strong baseline before interpreting practical value.

**Estimand.** Within each task/seed, (composite_relational − conventional_relational) − (composite_flat − conventional_flat). Also report composite_relational − conventional_relational. A positive interaction alone is insufficient if both architectures get worse.

**Falsifier.** For useful benefit, an upper interval endpoint below 0.01 rules out the proposed gain; a lower endpoint above 0.01 supports it. Separately test synergy against zero: an upper interaction endpoint ≤0 fails the positive-interaction claim. Uncertain or boundary-touching benefit intervals remain inconclusive.

**Complete accounting.** 12 new checkpoints; 24 checkpoint/task combinations × 3 support draws = 72 primary batches. Also 6 frozen-checkpoint comparator batches and 18 tree fits/batches. Each batch scores the complete declared validation and test populations. Counts name different units and must not be summed as identical training jobs.

**Cheapest next artifact.** Written tensor/interface contract and an explicitly proposed finite-gradient probe, followed by a complete training cost forecast. The probe itself is not run in L198.

**Declared matrix.** Every combination is listed in the machine-readable report; each axis below is frozen as a proposal.

- pretraining checkpoints: prior ∈ {single-table, relational} × encoder ∈ {conventional, composite} × seed ∈ {0, 1, 2} = **12**.
- primary prediction batches: prior ∈ {single-table, relational} × encoder ∈ {conventional, composite} × seed ∈ {0, 1, 2} × task ∈ {rel-f1/driver-dnf, rel-trial/study-outcome} × support_draw ∈ {0, 1, 2} = **72**.
- released prediction batches: model ∈ {released-DFS-RDB-PFN} × task ∈ {rel-f1/driver-dnf, rel-trial/study-outcome} × support_draw ∈ {0, 1, 2} = **6**.
- tree fit prediction batches: model ∈ {flattened-tree} × task ∈ {rel-f1/driver-dnf, rel-trial/study-outcome} × seed ∈ {0, 1, 2} × support_draw ∈ {0, 1, 2} = **18**.

**Unresolved before execution:**

- Exact source/data hashes and target reconstruction
- Full (entity_id, cutoff) train/validation/test manifests
- One validation-only selection rule and fixed search budget
- Per-task uncertainty resampling unit and coverage rationale
- Timed all-in cost bound including retries and validation
- Trainable graph/head interface, finite gradients and compatible initialization
- Exact generator/source exclusions, task curriculum, schedules and parameter/compute matching
- Meaningful same-shape control for single-table prior; do not change both graph access and encoder capacity
- Separate task/initialization uncertainty from repeated support draws

**Original priority rationale:**

- Data 4: public task releases and generators exist, but target-schema exclusion must be audited.
- Implementation 3: route mechanism is demonstrated; a trainable schema-shared encoder/head and adapters are new work.
- Compute 2: twelve new pretrained checkpoints plus controls have no verified total under $10.

**Preparation effort:** 24–48 human hours (authored planning estimate). All six full-run cost phases are unknown; budget NOT_ESTABLISHED.

**Closest work:** [relgnn](https://arxiv.org/html/2502.06784v2), [rdbpfn](https://arxiv.org/html/2603.03805v5), [relarena](https://arxiv.org/html/2608.16319v2).

**Original receipts (reports only):** [l182-report.json](packet/evidence/l189/packet/inherited/l182-report.json), [l177-report.json](packet/evidence/l189/packet/inherited/l177-report.json). These receipts are authenticated; their raw experiments are not all replayed. The full L197 packet is replayed separately.

**State:** PROPOSAL_NOT_EXECUTION_READY; models NOT_RUN; novelty NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.


### Compute-matched transfer

**Question.** Do historical-relation and future-activity objectives improve performance on unseen databases beyond scratch and extra-compute supervised controls at the same all-in cost?

**What is already known.** Temporal pretraining with supervised controls is already studied within databases. RT already studies cross-database transfer. L183 saved supervised predictions cannot establish a pretraining gain.

**Candidate contribution.** Evaluate temporal objectives under whole-database exclusion and matched end-to-end cost. Cross-database transfer and temporal pretraining each already exist.

**What L197 adds and leaves open.** A supervised or published leaderboard advantage cannot isolate a pretraining benefit. Extra-compute scratch is the decisive comparator; source plus target cost belongs in the comparison.

**Refined hypothesis.** Each temporal objective improves the held-out task by more than 0.01 AUROC over extra-compute scratch at the same all-in budget.

**Matched controls.** Scratch, extra-compute supervised scratch, historical-objective pretraining and future-objective pretraining use the same target information and evaluation policy. Match total cost, not just update count. Report per-objective cost and adaptation cost separately.

**Estimand.** For each held-out database, AUROC_pretrained − AUROC_extra_compute_scratch at matched all-in cost. Report the two databases separately and treat them as a narrow feasibility study, not a general scaling law.

**Falsifier.** For each objective/task, an upper paired interval endpoint below 0.01 rules out the useful gain; a lower endpoint above 0.01 supports it. Otherwise inconclusive. Database contamination or unmatched costs invalidate interpretation before statistical scoring.

**Complete accounting.** 12 source pretraining fits plus 24 target fits. Each target fit evaluates its complete validation/test task. All source preprocessing excludes the entire target database; objectives cannot borrow target schema statistics.

**Cheapest next artifact.** Database exclusion manifest and separate source/target cost envelope. A same-database result cannot substitute.

**Declared matrix.** Every combination is listed in the machine-readable report; each axis below is frozen as a proposal.

- source pretraining fits: holdout ∈ {rel-f1, rel-trial} × objective ∈ {historical, future} × seed ∈ {0, 1, 2} = **12**.
- target fits: task ∈ {rel-f1/driver-dnf, rel-trial/study-outcome} × arm ∈ {scratch, extra-compute-scratch, historical-pretrained, future-pretrained} × seed ∈ {0, 1, 2} = **24**.

**Unresolved before execution:**

- Exact source/data hashes and target reconstruction
- Full (entity_id, cutoff) train/validation/test manifests
- One validation-only selection rule and fixed search budget
- Per-task uncertainty resampling unit and coverage rationale
- Timed all-in cost bound including retries and validation
- Explicit source database corpus, target holdouts and preprocessing exclusion audit
- Exact shared backbone, adapters, objectives and learning schedules
- Dollar and runtime matching policy, hardware prices and amortization convention
- Multiplicity policy for two objective claims on each task

**Original priority rationale:**

- Data 3: releases exist, but a decontaminated source/target corpus contract is unfinished.
- Implementation 2: source objectives, adapters and semantic sharing need a full integrated trainer.
- Compute 1: 12 source fits plus 24 target fits have no verified $10 bound; historical blocked runs caution against optimism.

**Preparation effort:** 40–80 human hours (authored planning estimate). All six full-run cost phases are unknown; budget NOT_ESTABLISHED.

**Closest work:** [temporal](https://arxiv.org/html/2609.35219v1), [rt](https://arxiv.org/html/2510.06377v1), [survey](https://arxiv.org/html/2506.16654v1).

**Original receipts (reports only):** [l183-report.json](packet/evidence/l189/packet/inherited/l183-report.json), [l177-report.json](packet/evidence/l189/packet/inherited/l177-report.json). These receipts are authenticated; their raw experiments are not all replayed. The full L197 packet is replayed separately.

**State:** PROPOSAL_NOT_EXECUTION_READY; models NOT_RUN; novelty NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.
