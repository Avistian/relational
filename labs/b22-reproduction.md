# B22 · Reproduction contract

Approved 2026-10-04. [Lesson](../lessons/b22-support-state-refinement.html) · [Design](../docs/plans/2026-10-04-b22-design.md) · [Source manifest](sources/b22/manifest.json).

## Named historical experiment

RefineICL arXiv2609.27679v1 Appendix E.2, support-write intervention. Frozen L24 15K checkpoint; restore block 12's support inputs while preserving its query outputs, then execute later blocks normally. All72 RBF episodes: class counts2/4/8 × components4/16/64 ×8 replicates,8 input features,1024 supports,256 queries, held-out seed 805843. Paper says support class occupancy >=3%. Primary paired final-query cross-entropy change; secondary accuracy change. Paper reports mean+0.0510 CE,95% episode-bootstrap interval[0.0417,0.0608],72/72 positive, accuracy−1.44 percentage points.

**Gate: INCOMPLETE_SOURCE_PROTOCOL_GATE. Fresh paper inference NOT_RUN.** The paper explicitly excludes trained weights from its supplement. Retrieved arXiv v1 source archive has35 members, no Python/notebook/checkpoint files. Exact model/probe, checkpoint hash, episode identities, full-precision episode results and bootstrap recipe are not authenticated. This is a bounded source audit, not a claim that no separate release exists anywhere. An HTML reference number is not a replayed measurement. No original executable model source was recovered; visible course code is an explicitly different construction.

The CLI `--fresh` refuses before model execution. Recover missing artifacts, authenticate them, freeze environment and source recipe, and estimate complete inference/preparation/validation/retry cost before unlocking. Do not train substitute weights or reduce episodes and call the result a reproduction. Full pretraining, TabArena/AMLB/TabZilla benchmark suites and JEPA alternative remain NOT_RUN.

## Complete course experiment: B22-SUPPORT-WRITE

No training. Fixed random models seeds0/1/2, float64 width 8, three single-head attention-gated residual blocks, binary output. Twenty-four shared balanced episodes: linear/XOR/radial ×8 replicates, seeds22000..22023,12 support/16 query rows,2 features. Four paired trajectories each: normal, identity, skip second support write, cyclically permuted second support updates. Every query reads support keys/values only; no query answers in the prediction interface. All288 trajectories,4608 query predictions and216 paired intervention effects recorded, including intermediate states. Normal and identity match exactly; all interventions preserve second-block query outputs. The later block may change queries through changed support memory.

Course support attention includes self-reads. This is distinct from the paper's leave-one-out scoring objective. The permutation arm runs a new downstream trajectory; paper Table 10's permutation control scores one-block counterfactuals rather than a new forward trajectory. Our seeds vary random weights, not trained-model uncertainty. Paired episodes share rules and data across models, so do not treat288 trajectories as independent training trials. No confidence interval is supplied for a population generalization claim.

### Declared deviations

| Dimension | Historical target | Course mechanism |
|---|---|---|
| Weights | Pretrained L24 15K | Random, frozen, no training |
| Stack |24 blocks,width 1024 |3 blocks,width 8 |
| Intervention |After block 12 |After block 2 |
| Inputs |RBF,8 features,2/4/8 classes |Linear/XOR/radial,2 features,2 classes |
| Support/query |1024/256 |12/16 |
| Feature front end |Typed encoding, selected low-rank interactions, RowCLS, typed memory |Numeric projection plus support-label embedding |
| Controls |Normal and skip for downstream effect |Adds identity and cyclic-update permutation |
| Evidence |Checkpoint-specific causal contribution |Correct intervention mechanics and random-model sensitivity |

## Commands

From repository root:

```bash
.venv/bin/python labs/_budget_b22.py .venv/bin/python labs/_test_b22.py
.venv/bin/python labs/_budget_b22.py .venv/bin/python labs/_run_b22.py
.venv/bin/python labs/_budget_b22.py .venv/bin/python labs/_verify_b22.py
.venv/bin/python labs/_budget_b22.py .venv/bin/python labs/_reproduce_b22.py
.venv/bin/python labs/_reproduce_b22.py --fresh
```

The last command intentionally exits nonzero. The independent oracle uses scalar-loop attention and normalization, reconstructs all states/logits and metrics, and rejects missing trajectories, changed labels, altered query states, wrong permutations and changed metrics. Notebook tasks feed the actual full experiment; blank or plausible wrong solutions must fail their checks. Fresh course execution, saved replay, source identity, historical reproduction, deployment and learner mastery are separate statuses.

## Budget and publication

USD0 paid.3600 seconds aggregate local numerical/verification cap, including120 seconds preparation allowance and every retry; durable process-group timeout wrapper records usage. Source acquisition and author writing are not numerical runtime. StandingUSD10 total ceiling is unchanged; no uncosted cloud dispatch authorized. User explicitly requested push and Pages deployment; live success is reported only after workflow and byte/browser verification. Live Colab remains NOT_CHECKED. Learner status PENDING_WRITTEN_DEFENSE.
