# B21 reproduction contract

Approved2026-10-04; frozen design [here](../docs/plans/2026-10-04-b21-design.md). Paid budgetUSD0; aggregate local numerical/verification cutoff3600s including failures and retries. [Runtime ledger](evidence/b21/local-budget.json). No cloud calls. StandingUSD10 ceiling does not authorize additional scope.

## Exact evidence lanes

| Lane | Target | Status |
|---|---|---|
| Finite mechanism | B21-FK-STRESS:3 seeds ×3 budgets ×3 methods;105 enumerated model/states | COMPLETE |
| Independent numerical audit | NumPy forward of all105 states;315 output coordinates;27 selections | PASS |
| Source behavior | Unmodified extracted public primitives with recorded current dependencies | COMPLETE; failures retained |
| Paper/saved-output audit | All35 printed selected-task mean/std cells; all35 saved tutorial result rows | COMPLETE; protocols INCOMPARABLE |
| Selected historical experiment | Table4 qualifying-position:175 seed/method/budget conditions | INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN |
| Whole paper | Other tasks, original training, all figures, timing platform | NOT_RUN |
| Learner mastery | Three live TODOs and written defense | PENDING_WRITTEN_DEFENSE |
| Live Colab / deployment | Separate operational claims | NOT_CHECKED / NOT_REQUESTED |

## Frozen finite experiment

Inputs in `relkit/structural_b21.py::fixture`, repeated in `evidence/b21/diagnostic.json`. Parents0/1/2 existed by event time; parent3 arrives at12 after query cutoff10. Six immutable events have session IDs[0,0,1,2,3,4], owner IDs[0,0,1,2,1,2]. Session determines owner. Events0/1 must move together, at cost2 FK cells. Other groups cost1. Node features and parent targets[2,4,6] fixed. No fitted weights: NumPy default_rng seeds0/1/2 draw explicitly scaled random weights; float64 two-layer width4 bidirectional mean GraphSAGE-style model, ReLU, one linear scalar head. Query parents0/1/2 only. This demonstrates sensitivity, not predictive accuracy or generalization.

10 atomic group/destination changes;35 legal states within2 FK edits,9 within1,1 within0. Mean squared error objective over three queries. Label access explicitly allowed to this white-box evaluator. Random selects uniformly from exact-cost legal states using seed1000+10*model_seed+budget, one draw per condition; this is not an expected-random baseline. Gradient maximizes the full first-order joint direction over legal at-most-budget states (includes removed edges and reverse edges). Exhaustive maximizes actual MSE over the same feasible states including clean. Both optimizers prefer fewer edits then lexicographic assignment in a tie. This differs from the paper's candidate-sampled greedy ranking and MAE. Do not compare course MSE with paper MAE.

Every condition reports actual cost, allowed budget, complete assignment, predictions, clean and attacked loss, linear gain and regret against exhaustive optimum. Paired access/weights/features/targets fixed. Inputs, model state, predictions and source have hashes in delivery seal. Saved full replay and fresh re-execution remain distinct.

## Historical target

[Paper2607.07089v1](https://arxiv.org/html/2607.07089v1), Table4, qualifying-position only. Seven methods; budgets1,25,50,75,100; seeds39–43; 18epochs,2layers,width128,100candidate samples/source, batch512,128neighbors/hop; frozen trained weights and one sampled batch. Report mean/std MAE with published4decimal precision only if historical identities align.35 aggregate cells represent175 seed conditions. No acceptance threshold is used before identity gates pass.

[Author source](https://github.com/alanganyDB/Structural-Adversarial-Attacks-on-Relational-Deep-Learning-under-Integrity-Constraints), pinned1bc60c7eee605fddbb0975a1c28c64ecc8ce515a. Original README, utils, notebook and paperHTML archived with URLs/SHA256 in `sources/b21/manifest.json`. No checkpoint/data identity manifest or dependency lock appears in that pinned tree; this bounded finding is not a claim about every possible author artifact. Full original model/trainer/attack code remains visible in the archive and portable notebook appendix.

## Observed source gates and differences

1. Tutorial config:1epoch,seed39,budgets1/125/250/375/500. Original Table4 needs18epochs,five seeds and a different grid. Saved outputs include a stale configuration from another task; current seed loop `for seed in [SEEDS]` raises `TypeError` because the key is a list. Saved outputs cannot certify clean execution of these cells.
2. Current-runtime clean model conversion: `strict=False` leaves four new `lin_r.bias` tensors uninitialized from the base model. Three synthetic base/wrapper embedding discrepancies1.422336/0.712215/1.648367. Zeroing only those new biases produces exact embedding equality on these fixtures. This controlled probe establishes a local conversion defect and repair, not historical cause or whole-model benchmark equivalence. Torch2.13.0+cpu,PyG2.8.0.post1,RelBench1.1.0 recorded.
3. Candidate sampler draws parent indices with replacement and checks neither timestamps nor cross-row dependencies. A synthetic probe contains8 future-parent proposals (duplicates included). A one-row edit can violate session->owner. These findings do not prove the original F1 batches contained future-parent candidates; a global validator and original-batch identities are still required.
4. The edit function stores an overridden `edge_index_dict` while underlying relation-store `edge_index` remains clean under current PyG. The attackable forward reads the overridden dictionary, so do not call this a no-op. Database reconstruction and stored graph consistency require explicit checks. The selector also picks1 edit with budget0; paper budgets start at1, so this is an added boundary probe, not an explanation of the published results.
5. Course loss isMSE, model untrained/width4 and graph tiny. Course temporal rule (parent available by event time), coupled session constraint, weighted-mean normalization, exhaustive universe and linearized global optimizer are declared course choices. The tutorial uses sum aggregation; do not attribute mean-normalization behavior to its configured experiment.
6. Exact data/cache versions, historical fitted checkpoints, selected batches, sampled candidates, full precision per-seed predictions and environment have not been authenticated. Source availability and a repaired toy wrapper cannot fill these gaps.

## Operators and stop behavior

From repository root using the prepared environment:

```bash
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_test_b21.py
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_run_b21.py
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_verify_b21.py
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_mechanism_b21.py
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_source_probe_b21.py
.venv/bin/python labs/_budget_b21.py .venv/bin/python labs/_reproduce_b21.py
.venv/bin/python labs/_reproduce_b21.py --fresh
```

Final command deliberately exits nonzero with `INCOMPLETE_SOURCE_PROTOCOL_GATE` before training. Original full training code is preserved, but no executable historical recipe is invented. To unlock: recover identities; freeze a repaired-release versus historical status; verify clean/attackable outputs and integrity end-to-end; estimate all175 conditions plus fitting, setup, checks and retry margin. Obtain a costed approval if paid compute is needed. Do not silently reduce seeds/epochs, select favorable results or exceed the budget.

Source reads and authoring time are not numerical runtime. The numerical wrapper reserves remaining time, terminates process groups at cutoff and records failed attempts.120s preparation allowance covers preliminary numerical/runtime inspection; final verification allowance recorded separately. No paper fit or benchmark inference has run.
