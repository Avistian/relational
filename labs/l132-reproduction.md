# Lesson 132 reproduction contract

## Named target

RelBench arXiv2407.20060v1, Tables5/8, rel-trial/condition-sponsor-run. Two-tower GraphSAGE versus identity-aware GNN; five fresh seeds0–4 per model, 20 epochs. Published MAP@10 percentages: SAGE validation3.12±.24/test2.89±.39; ID-GNN validation11.33±.04/test11.36±.08. These are paper targets, not author results. Predeclared descriptive mean tolerance:0.5percentage points for each split/model; CLOSE is not an equivalence test. Primary source: https://arxiv.org/html/2407.20060v1.

Author execution status is in `evidence/l132/summary.json`. Synthetic parity and controlled BCE fits are separate evidence in `evidence/l132/fixture.json`. No historical or whole-paper parity is established. Learner PENDING_WRITTEN_DEFENSE.

## Actual execution outcome

Selected experiment **INCOMPLETE**; full ten-fit schedule **NOT_RUN** after the pilot budget gate. Both full-data one-epoch seed100pilots completed. GraphSAGE validation/test MAP1.66945%/1.46864%; ID-GNN9.85571%/10.14694%. These are feasibility pilots, not20-epoch five-seed paper scores; no seed uncertainty estimate or numerical-parity verdict is claimed.

Preparation took824.45worker-seconds including evidence writes. Measured epoch times were65.77s(SAGE) and156.86s(ID-GNN). With observed setup/final-evaluation overhead and25%margin, projected full-fit durations are1755.76s and4217.78s. The latter exceeds the1800sworker cutoff. Projected ten-fit worker costUSD7.80264 plus recorded preparation/pilotsUSD0.36450 plus reserved overheadUSD2.47629 totals aboutUSD10.64342. Consequently no full fits were dispatched. Original pre-worker mount failure remains separately recorded; unitemized startup/build/storage overhead is not an invoice.

The complete raw-label SQLite audit, original neural parity, both synthetic fits, both real-data pilots, and independent ranking checks remain useful executed evidence. They are not relabeled as full reproduction. To attempt the complete comparison independently, use fresh run directories and a new budget/runtime allocation; never edit the existing ledger to pretend those runs occurred.

## Pinned protocol and deviations

| Axis | Released-protocol contract / unresolved detail |
|---|---|
| Source | RelBench9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; original scripts retained under sources/l132, with MIT license |
| Database | rel-trial/db.zip SHA256 9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a |
| Task archive | condition-sponsor-run.zip SHA256 eeef170e06b6728928116c333f56601e036b24fcbef4f97dc93ec1948700c2f8 |
| Task | Future365-day sponsor set for condition; 36934/2081/2057 train/validation/test queries |
| Preprocessing | Released database through test cap; full-database Frame statistics; type inference frozen at42 and shared across fits, not original historical seed provenance |
| Text | average_word_embeddings_glove.6B.300d revision e5e8fec6971be8960cfaa853a77a6ddc62a265d7 |
| Graph | All typed PK/FK and reverse edges from full archived database; future-label task edges not inserted |
| Encoders | Separate four-block Frame ResNet per table;128-dimensional row encodings and relative-time additions |
| GraphSAGE | Two layers, sum aggregation, fanouts128/64, shared two-tower model, sponsor-ID embeddings enabled, inner product, BPR with shared-time negatives |
| ID-GNN | Four layers, fanouts128/64/32/16, one shared root marker, scalar sampled-sponsor head, BCE; bidirectional sampled graph |
| Sampling | Explicit uniform for both training paths, following paper. Released ID-GNN CLI defaults to last; this override is recorded. SAGE evaluation retains released loader defaults |
| Optimizer | Adam.001, batch512, 20epochs, maximum2000steps (upstream break condition retained) |
| SAGE epoch coverage | TimestampSampler drops short timestamp-group tails:62batches ×512=31744draws per epoch from36934available queries; preserve original shuffling/drop policy |
| Selection | SAGE last validation tie (>=); ID-GNN strict improvement (>). Test once after frozen checkpoint |
| Evaluation | MAP@10. Unsampled ID-GNN sponsors receive0; tied zero-score top-k remains upstream behavior. Final validation resampling differs from selection sampling |
| Runtime | Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame0.2.3,RelBench1.1.0; requirements-l117-runtime.txt plus pyg_lib0.4.0+pt25cu124 |
| Seeds | Fresh0–4 planned; historical seeds and historical exact training commit/runtime NOT_ESTABLISHED |
| Availability | Actual arrival/version histories and undated-row creation times unavailable; timestamps alone are not a full availability proof |

The experiment comparison changes objectives, heads, depth and shallow embeddings together. It is not a marker-only causal experiment. The default notebook's marked/unmarked BCE arms provide that narrower intervention on a synthetic graph. They start at identical weights and pair dropout random draws. Local course runtime differs from the pinned GPU reproduction runtime.

## Visible implementation and evidence

`relkit/identity_l132.py` implements marking, owner-key labels, independent MAP, the rooted-walk witness and explicit identity forward pass. `sources/l132/model.py` and `nn.py` contain the complete released architecture. `gnn_link.py` and `idgnn_link.py` contain complete original loss, optimization, sampling, ranking and selection code. The notebook includes those sources, graph construction and loaders visibly, as well as a full execution gate. Learner marker/label functions drive the actual tiny neural fits; learner MAP also scores the standalone full gate. Official author runs retain original model and trainer operations.

The full runner redirects graph construction to a single pinned prepared graph, skips redundant text-model instantiation, and inserts timing/checkpoint-epoch/evidence writes. It preserves source algorithm operations and saves each executed transformed script beside its result. Installed graph/loader/NN sources must match pinned bytes. Completed runs cannot be overwritten. Every successful fit independently reconciles MAP against the official evaluator and stores keyed predictions plus checkpoint. No completed fit is inferred from a dispatch or partial log.

Task archive audit checks all41072query identities/target sets and120constructed ranking lists against the official metric. Independent SQLite regeneration matches all41072target sets and559318positive pairs from the archived condition/study and sponsor/study tables. Input-file hashes and counts are in _sql_audit_l132_results.json. Small neural source parity verifies exact outputs and parameter gradients with matched RNG, before the marker intervention. Six semantic mutants are rejected.

## Commands

From the repository root:

```bash
.venv/bin/python labs/_check_l132.py
.venv/bin/python labs/_mutation_l132.py
.venv/bin/python labs/_fixture_l132.py
.venv/bin/python labs/_task_audit_l132.py
.venv/bin/python labs/_sql_audit_l132.py
.venv/bin/modal run --detach modal/l132_repro.py --mode prepare
.venv/bin/python labs/_collect_l132.py
.venv/bin/modal run --detach modal/l132_repro.py --mode pilot
.venv/bin/python labs/_collect_l132.py
.venv/bin/python labs/_pilot_check_l132.py
# Only if both pilots pass the automatic admission gate:
.venv/bin/modal run --detach modal/l132_repro.py --mode paper
.venv/bin/python labs/_collect_l132.py
.venv/bin/python labs/_analyze_l132.py
.venv/bin/python labs/_figures_l132.py
.venv/bin/python labs/_build_l132.py
.venv/bin/python labs/_execute_l132.py
.venv/bin/python labs/_delivery_l132.py
```

For independent local reproduction in the pinned runtime:

```bash
python3 labs/_run_l132.py --prepare --root labs/results/l132/new-prepared
for variant in sage idgnn; do
  for seed in 0 1 2 3 4; do
    python3 labs/_run_l132.py --variant "$variant" --seed "$seed" --epochs 20 \
      --root labs/results/l132/new-prepared --output "labs/results/l132/new-$variant-$seed"
  done
done
```

Use a fresh volume and new budget ledger for an independent cloud reproduction; original reservations are never erased or reused. The notebook gate is off by default and lacks monetary enforcement. Its pinned-runtime assertions reject incompatible kernels before full training. The default solution was executed independently in an empty directory; this does not establish the full gate or Google's live Colab frontend.

## Budget

USD10 aggregate, all seeds/retries/validation/overhead. Current2026-09-27 resource rates: T4.000164 +2CPU*.0000131 +32GiB*.00000222 =.00026124USD/second, from https://modal.com/pricing. Reserve at most28800worker-seconds =USD7.523712; retainUSD2.476288overhead margin. Preparation has3600s timeout; each pilot/fit has1800s timeout; no automatic retries.

An initial image configuration created a nonempty volume mount path and failed before a worker ran. Its conservative3600s reservation remains in the ledger. Recovery preparation3600s, two pilots3600s and ten fits18000s exhaust the planned28800s ceiling. Pilot projection includes a25%runtime margin and must fit both per-worker timeout and aggregate remaining reservation before any full dispatch. Recorded resource estimates exclude unitemized startup/storage/build overhead and are not invoices.

## Delivery boundary

Browser, copied Pages staging, notebook execution and model execution have separate reports. No publication requested. Live Colab and deployment NOT_CHECKED. Author execution does not establish learner mastery.


## Local notebook memory recovery

The original real-prediction replay cell embedded a large nested Python literal. IPython execution exceeded a 1.5GiB diagnostic cap (1364080KiB RSS) and its rich traceback also ran out of memory; prior unbounded notebook attempts exhausted WSL. Plain Python replay passed at165684KiB. The builder now embeds compressed JSON with SHA256 verification and wrapped base64 lines. The same isolated IPython probe passes at136124KiB, rescoring all8276 rankings. Numerical data and checks are unchanged.

`labs/_execute_l132.py` now sets an inherited4GiB address-space limit,300CPU-second limit and one BLAS/OpenMP thread before launching the kernel. All30code cells pass in an empty working directory. This default execution is the tiny fixture plus author-prediction replay; full benchmark training remains OFF. Recheck safely with `.venv/bin/python labs/_memory_probe_l132.py --ipython` before `.venv/bin/python labs/_execute_l132.py`.
