# B11 — RelGNN versus RelGT: frozen evidence contract

Approved 2026-10-03. B11 performs CPU mechanism checks and saved-evidence rescoring. Paid execution USD0; aggregate local preparation, failed attempts, numerical execution and verification cutoff 3600 seconds, enforced by `_budget_b11.py`. No new full benchmark training or cloud dispatch.

## Named experiments and what actually runs

1. **B11 block comparison:** seeds 0/1/2, float64 CPU, source-derived RelGNN composite block (four input/output coordinates, two heads) and RelGT local+global block (width8, two heads, one local layer, three centroids/global width4). Both receive the same seven pre-encoded rows: two products, three purchases, two customers. GNN uses directed FK routes; RelGT receives root-first contexts containing all seven rows. Its global branch also reads fixed supplied centroid buffers. These are not equal-information training arms. Source parameters are copied, outputs and all active parameter/input gradients compared at absolute tolerance 1e-9. Test ordinary/empty GNN attention, neighbor/row permutations, forbidden-input perturbations and centroid perturbations. No fitting, row-encoder parity, EMA-update parity, full-model parity or held-out accuracy claim.
2. **B11 saved-evidence audit:** all five L143 RelGNN reconstructed training runs and all six L146 course runs; 499 validation +760 test predictions per run, 13,849 predictions total. Recompute MAE in complete `(entity,cutoff)` key order, verify targets against inherited L146 task records and check all 11 ten-epoch validation selections. Reordering predictions must not change metrics; duplicate, missing and nonfinite packets fail. This reuses labels; it does not newly reconstruct raw F1 outcomes.
3. **Full RelGNN target retained:** v2 §5.2/Table2 `rel-f1/driver-position`, published MAE3.798. Five seeds0–4, 7,453/499/760 queries, ten complete epochs, Adam .005, batch512, temporal128/64, one composite layer, width128, four heads each of output128, sum routes, L1, train2/98-percentile evaluation clipping, first strict validation minimum. Source cffdb8b54627e92c7dd112c1243dde739c90d35b. L143 completed reconstructed fresh runs previously; B11 only rescores them. Historical recipe unresolved and nonfinite source gradients disclosed. Checkpoint-compatibility replay is outside B11's eleven training-run aggregate. [Complete contract](l143-reproduction.md), [visible model](relkit/relgnn_l143.py), [full trainer/materializer](_full_l143.py).
4. **Full RelGT target retained:** v1 Table1/Table6 same task, published MAE3.9170. All nine depth1/4/8 × dropout.3/.4/.5 configurations, seed0, 100epochs, width512, fourheads, K300, 4096centroids, globalwidth256, batch256, Adam1e-4, weightdecay1e-5, gradientclip1, L1, train2/98-percentile clipping, last tied within-fit validation minimum. Historical cross-configuration selection not established; declared reconstruction selects final validation MAE, never test. Source19e423ca3e7cac761130aba790857f2dc3a46ef7. [Complete contract](l145-reproduction.md), [visible full model](relkit/relgt_l145.py), [full trainer](_full_l145.py).

RelGT full search remains **INCOMPLETE_TEMPORAL_AND_BUDGET_GATE**: the inherited source audit found entity-only context overwrites and future-token exposure. Prior shallow-pilot extrapolation is USD80.4115 for nine configurations before overhead; deeper runs were unmeasured. This is an inherited projection, not current pricing or a new invoice. Do not change K, epochs, seeds or contexts and label the result a reproduction. A repair defines a new experiment. All full B11 benchmark fits are **NOT_RUN**.

## Reproduce the approved scope

From the repository root, using the course virtual environment:

```bash
.venv/bin/python labs/_budget_b11.py .venv/bin/python labs/_test_b11.py
.venv/bin/python labs/_budget_b11.py .venv/bin/python labs/_run_b11.py
.venv/bin/python labs/_budget_b11.py .venv/bin/python labs/_audit_b11.py
.venv/bin/python labs/_reproduce_b11.py --audit
```

`_prepare_b11.py` is the author ingestion step. It copies archives, checks inherited source and L143 prediction ledgers, and creates the B11 seal; normal replay verifies the existing seal and does not regenerate it. L146 prediction bytes have a new B11 seal, while their reported scores and context hashes are independently checked against saved records. This does not authenticate historical checkpoint bytes. The notebook embeds the same sealed packet and displays all load-bearing block and audit code. The full original encoders/trainers and licenses are included in the archive for inspection; its default execution is the declared CPU scope.

The retained full operators are executable Python functions, with explicit local paths:

```python
# REFERENCE ONLY: these allocate training work; outside B11 execution scope.
from _full_l143 import materialize, full_run
# materialize(prepared_root, relgnn_source_root)
# for seed in range(5):
#     full_run(new_run_dirs[seed], prepared_root, relgnn_source_root, seed=seed)
from _full_l145 import prepare, full_search
# prepare(token_root, prepared_root, relgt_source_root)
# full_search(token_root, prepared_root, relgt_source_root, new_output)
```

Use the pinned runtime and data/cache requirements in the linked contracts. RelGT's clean `full_search` must reject the known failed audit; no forensic override is authorized. The B11 command `--run-full` refuses before importing trainers.

## Matched study specification — NOT_RUN

Use the full 7,453/499/760 driver-position queries and complete keys, the same labels, feature columns, train-fitted transforms and query-keyed temporal contexts. Admit rows before expanding neighbors; declare static-row and availability policies. Exclude task targets from input. Enumerate RelGNN routes only within admitted evidence. RelGT may use structural encodings derived from those same edges; random PE draws must be frozen per query. A global arm must update centroids only from the declared training population and freeze them for validation/test. Matching local rows alone is insufficient.

Predeclare two comparisons: (A) local evidence only, with RelGT global attention disabled as an explicit ablation; (B) full mechanisms with training-only global state, declared as different inductive biases. Use paired seeds0/1/2, one configuration per arm, ten complete epochs and identical minibatch query orders, Adam.001/weightdecay1e-5/L1/clip1, first strict validation minimum, and training-only clipping thresholds. This is a course protocol, not either paper's full protocol. Width64 is a starting design choice, not equal parameter count. Report parameters, peak memory, actual optimizer steps and measured time separately. Equal epochs do not establish equal compute. Before any launch, pilot each arm and freeze a shared search/compute allowance within the aggregate USD10 lesson budget, including preparation, all seeds, validation and retry reserve; if it cannot fit, stop without truncating arms. No paid pilot is authorized by B11's USD0 scope.

## Evidence boundaries

L146's `gnn` is a typed-mean GNN, **not RelGNN**. Its reduced corrected RelGT is not the nine-configuration source model. Do not rank L143 against L146 as a matched study. No whole-paper reproduction, current SOTA, historical availability, unseen-test confirmation, live Colab, deployment or learner mastery is established. Learner status: **PENDING_WRITTEN_DEFENSE**.
