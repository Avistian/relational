# Lesson 146 — GNN versus graph transformer: reproduction and comparison ledger

Approved option1, 2026-09-30. This is an evaluation lesson with a complete full-data course comparison and a separately retained named-paper reproduction lane.

## Outcomes

**Course comparison COMPLETE:** six fresh full-data fits, three seeds per arm, ten full epochs per fit. **Full selected paper reproduction INCOMPLETE:** nine source RelGT100-epoch fits and a fresh canonical RDL comparator NOT_RUN. Historical identity and cross-configuration selection NOT_ESTABLISHED. Whole paper NOT_RUN. Author execution is not learner mastery: PENDING_WRITTEN_DEFENSE. Live Colab/deployment NOT_CHECKED.

| Arm | Validation MAE, mean ± sample seed SD | Test MAE, mean ± sample seed SD | Parameters |
|---|---:|---:|---:|
| Course typed-mean GNN | 3.197652 ± .056532 | 4.293632 ± .037816 | 3,485,825 |
| Reduced corrected RelGT | 3.048312 ± .102734 | 4.617536 ± .291141 | 3,359,558 |

Paired GNN−RelGT test difference −.323904 ± .253784MAE (sample seed SD, not a confidence interval). GNN wins all3test pairs; RelGT wins all3validation pairs. No post-test changes to the model/training recipe. All7554held-out predictions independently aligned to full query keys and rescored using an independent MAE implementation. `evidence/l146/summary.json` contains complete histories, scores, selected epochs, hashes and per-seed values.

## Primary named paper lane

RelGT v1 Table1 F1 driver-position: published RelGT3.9170 versus RDL4.022MAE. Paper https://arxiv.org/html/2505.10960v1; source revision `19e423ca3e7cac761130aba790857f2dc3a46ef7`, MIT license in `sources/l145/LICENSE`. Source hashes: `evidence/l146/sources.json`.

Keep complete released RelGT9config search: depths1/4/8 × dropout.3/.4/.5, seed0,100epochs, width512,K300,4096centroids,batch256,Adam.0001,weightdecay1e-5,gradientclip1,L1,train2/98-percentile output clipping,last tied within-fit validation minimum. Source full model/trainer: `relkit/relgt_l145.py`, `_full_l145.py`; preparation/commands and compatibility details: `l145-reproduction.md`. Canonical RDL source model and complete trainer: `relkit/rdl_l117.py`, `_run_l117.py`; five-seed full recipe/provenance: `l117-reproduction.md`. No L117 measured result is presented as a fresh L146 comparator.

Table6 displayed validation minimum: L4/dropout.3 (3.1046), test4.6316. Displayed test minimum: L1/dropout.5 (3.917), validation3.3257; matches headline. Under displayed-validation selection, relative gain against cited RDL is−15.1566%; headline arithmetic gives+2.6106%. This is a reconstruction of rounded printed scores, not a rerun. Released final reevaluation is stochastic and may differ from selection-time metrics. Inspected per-fit training and sweep launch code do not identify historical cross-configuration selection; do not infer intent.

Full clean reproduction remains blocked by the L145 source token temporal audit and its budget projection: aboutUSD80.41for all9configs at measured shallow speed, before other costs, with deeper architectures unmeasured. This estimate is inherited pilot evidence, not a new invoice or full runtime. No additional full-protocol pilot is needed merely to rediscover these blockers. Source-faithful replay remains runnable in `_full_l145.py::full_search` with an explicit forensic override, but the clean gate must not bypass the failed audit. Repaired course contexts cannot be substituted silently.

## Frozen course protocol

- Population: full7453train/499validation/760test queries, unique(driver,cutoff) keys. All8712labels independently rebuilt from raw results.positionOrder in(cutoff,cutoff+60days]. Raw future-inclusive database is accessed only for targets.
- Graph: REUSED L143 test-cutoff snapshot under SHA256 `6c72c684d51aaedbba11946d9babee705a7c0bc1e93415f4d48926b38973da99`;74063rows,338842directedFKedges,9tabletypes,26directedrelations. Archive hashes and package versions are in the preparation audit. Nested inherited metadata about RelGNN routes/checkpoints is provenance only, not an L146 mechanism or weight initialization. No checkpoint weights were reused for training.
- Feature/statistic preprocessing is reused, including transductive snapshot statistics and missing-value handling in the source row encoder; not refit at each cutoff. Event-time legality does not establish historical arrival-time legality. Previously examined test population means descriptive evidence, not a new untouched confirmatory experiment.
- Contexts: K32slots including root; query-keyed stable SHA256-seeded BFS, at most2hops, sorted neighbors shuffled under seed146; filter future nodes before frontier expansion; stop atK; repeat selected legal rows if fewer thanK. No global fallback. Preserve each duplicate slot's incident edges. Same cache hashes for both arms/all seeds. Future token occurrences:0. Padding occurrences:43000/1228/10623train/val/test. Root token age0; static row age0.
- Both arms: same source typed row encoder (per-table TorchFrame ResNet), width64 representations,2message/local-attention layers,dropout.1,batch128,Adam.001,weightdecay1e-5,L1,gradientclip1,10completeepochs,pairedseedlabels0/1/2. Fixed per-seed epoch permutations; same examples/order across arms. First strict validation minimum; full state restored. Output clipping uses training-label2nd/98th percentiles. No hyperparameter search. Test scores do not select checkpoints/configurations.
- GNN: row+type+age encoding, separate neighbor mean and linear map for each directed relation, destination-type self transform once, relation sums,LayerNorm/ReLU/dropout,rootMLP. This custom control is not canonical RDL's original HeteroGraphSAGE recipe.
- Reduced RelGT: full source five-element encoder,GINstructure,local/global architecture;2localblocks,4heads,width64,globaldim32,128centroids,K32. Correct eval attention dropout to0; use fixed normal structural draws with seed146for a fixed evaluation batch layout. Training path remains source-equivalent;149gradient tensors and full outputs match original on a six-row fixture. CPU fixture repeats exactly; CUDA aggregation permits≤1e-5absolute differences. No architecture or optimizer retuning followed test access.
- Parameter counts,encodings,readout,EMAstate,relationhandling,RNGconsumption and compute differ. This compares designs under a common recipe; it does not isolate attention. Same seeds do not mean identical initialization. Same epochs do not mean equal compute.

## Failures and recovery

The first GNN timing pilot failed because `self.type` collided with `nn.Module.type`. A full-network regression fixture reproduced the failure; renaming to`type_embedding`fixed it. Failed phase `pilot-gnn-99` remains recorded; replacement timing pilot98is not a primary seed. RelGT timing pilot99is also excluded.

Five primary fits completed all ten epochs and saved their selected checkpoints before failing an overstrict bit-exact repeated-evaluation assertion. CUDA aggregation differences were2.8610e-6(GNN) or3.8147e-6(RelGT), not evidence of active eval dropout. The post-evaluation assertion was changed to absolute1e-5,rtol0. A single recovery worker reevaluated all five unchanged checkpoints; it did not instantiate an optimizer or train. Original histories, failures and checkpoint hashes remain. The sixth fit ran once with the corrected numeric check. `executed-trainer-before-eval-check.py`preserves the original trainer; all dispatch reservations contain source hashes. Validation history still determines each selected checkpoint. Recovered scores come from the saved selected checkpoint's new final evaluation, not the unsaved first final prediction vectors.

A final learner-facing sampler check added an explicit root-only return for K=1. Every recorded context used K=32, so this changes no measured sample or fit. The executed contract source is retained as`evidence/l146/executed-contracts.py`; final notebook validation includes the boundary check.

## Cost and cutoff

Hard aggregate capUSD10;USD2overhead reserve. T4+2physicalCPUcores+16GiB atUSD.00022572/sec; rates checked2026-09-30 https://modal.com/pricing. Every preparation/pilot/fit/recovery dispatch reserved its actual1800stimeout before launch; notebook validation reserves600s. No retries are automatic. Successful pilots projected the complete6fitworker time atUSD.4517,orUSD.9033with2×margin. Runtime estimates are not guarantees.

After one failed pilot and one evaluation-only recovery, reservations including both notebook validations totalUSD6.740120with overhead. Actual final reservation and measured-worker estimates are computed in `_verify_l146_results.json`. Worker-body estimates exclude unitemized build/startup/commit/storage overhead and are not an invoice. Invoice NOT_ITEMIZED. GPU credits do not change the cap.

## Reproduce locally

From repository root:

```bash
.venv/bin/python labs/_check_l146.py
.venv/bin/python labs/_paper_l146.py
.venv/bin/python labs/_analyze_l146.py
.venv/bin/python labs/_figures_l146.py
.venv/bin/python labs/_build_l146.py
.venv/bin/python labs/_execute_l146.py
.venv/bin/python labs/_verify_l146.py
.venv/bin/python labs/_delivery_l146.py
.venv/bin/python labs/_reproduce_l146.py --audit
```

Without`--audit`, the source-reproduction preflight exits before training while its temporal/cost contracts fail. The canonical complete RDL entrypoint is`labs/_run_l117.py --seed N --epochs 10 --output NEW_DIRECTORY`; all five seeds and preparation costs belong to a separately planned full-protocol run, not to this course result.

The portable solution embeds complete course model/trainer source, original RelGT source for independent comparison, reference predictions and raw context timestamps. Default execution in an empty directory audits author artifacts and mechanisms. The explicit `RUN_FULL_COURSE`gate requires a hash-verified prior graph plus a new context directory and runs all six fits. Recreate the raw materialization using `labs/_full_l143.py::materialize`; cross-environment binary hash identity is not guaranteed. Exact prepared caches and selected weights are retained in Modal volume`l146-comparison-evidence`; checkpoints are at`fit-<arm>-<seed>/selected.pt`. The prior graph volume is mounted read-only.

Author commands (already executed, immutable phase names cannot be reused):

```bash
.venv/bin/modal run modal/l146_repro.py --phase prepare
.venv/bin/modal run modal/l146_repro.py --phase pilot-relgt-99
.venv/bin/modal run modal/l146_repro.py --phase pilot-gnn-99  # retained failure
.venv/bin/modal run modal/l146_repro.py --phase pilot-gnn-98
# Six primary phases: fit-gnn-0, fit-relgt-0, fit-gnn-1, fit-relgt-1, fit-gnn-2, fit-relgt-2
.venv/bin/modal run modal/l146_repro.py --phase recover       # evaluation only for first five
.venv/bin/python labs/_collect_l146.py fit-relgt-2            # repeat for all six
.venv/bin/modal run modal/l146_notebook_check.py --phase notebook
```

For a genuinely new cloud reproduction, allocate a new evidence volume and reset a new run's cost ledger; preserve these immutable records. Final local/pinned notebook, browser/mobile/keyboard/reset/noJS/print, deterministic build and copied-Pages results live in the execution/delivery reports. No push or deployment requested.
