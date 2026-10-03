# Lesson 145 — RelGT selected reproduction protocol

Approved scope: RelGT v1 Table 1, rel-f1/driver-position; complete released nine-configuration search, aggregate USD10 cap. Paper target test MAE3.9170; paper RDL comparator4.022. The latter is a cited comparator, not a fresh same-protocol baseline. No fresh GNN run was dispatched after the temporal and cost stop.

## Outcome

**INCOMPLETE** full selected reproduction. Nine full fits **NOT_RUN**. Temporal audit **FAIL**. Historical identity **NOT_ESTABLISHED**. Whole paper **NOT_RUN**. No test predictions were generated. Default notebook execution is mechanism/evidence auditing, not benchmark training. Learner **PENDING_WRITTEN_DEFENSE**. Live Colab and deployment **NOT_CHECKED**.

All8712labels independently reconstructed from raw `results.positionOrder` in `(cutoff,cutoff+60days]` and matched to archived task labels. Populations:7453train/499val/760test. Hash-verified graph reused from L143:74063nodes/338842directed edges. This is **REUSED** preprocessing, not fresh L145 materialization. Graph cache SHA256 `6c72c684d51aaedbba11946d9babee705a7c0bc1e93415f4d48926b38973da99`. Full raw database is used only for label reconstruction; source graph is the released test-cutoff snapshot. Graph metadata includes unrelated L143 route/checkpoint fields; those are provenance only and are not RelGT weights or mechanisms.

## Source and deviations

Paper: https://arxiv.org/html/2505.10960v1
Release: https://github.com/snap-stanford/relgt/tree/19e423ca3e7cac761130aba790857f2dc3a46ef7
Per-file source hashes and MIT license: `evidence/l145/source.json`, `sources/l145/LICENSE`.

Visible model combines codebook/encoders/local_module/model modules; five-component mixing calls learner `mix_five`. The original classes validate outputs/gradients. Source model behavior is preserved, including stochastic evaluation. Source primitive imports (PyG GIN, PyTorch Frame column encoders) remain explicit. Historical versions and preprocessing cache are not released; pinned runtime is a reconstruction.

Serial token preparation evaluates the last query per entity because each `_process_one_seed` resets its own RNG and the released dictionary overwrites earlier entities in the same chunk. All F1 splits fit within one10000-row chunk. This preserves the final source dictionary for these splits; it is not a fix. PYTHONHASHSEED=0 pins otherwise process-dependent hashing/set iteration, a declared reconstruction choice. Worker count and CPU scheduling differ. The training harness omits single-rank DDP/SyncBatchNorm and tracking services; real-batch model parity is measured separately. The default CPU forensic fixture omits only an unused sentence-transformer import while executing original sampler code.

### Confirmed source defects

- Source cache keyed by entity rather than(entity,cutoff):6682train/452val/704test query contexts overwritten. **569502train** future-token occurrences across3277queries; **21368val** across217queries; zero test future-token occurrences. These are token occurrences, not unique database rows. Older cutoffs may overwrite later ones depending on archive order; overwritten does not imply future leakage for every row.
- Global fallback samples all nodes without temporal filtering, independently confirmed using the original `_process_one_seed`. Negative-age masking leaves attributes accessible.
- Duplicate sampled entities map to the last slot as an adjacency destination. Five encodings do not repair a malformed ownership/adjacency contract.
- Scaled-dot-product local attention receives a nonzero dropout probability during eval. PE random scalars also resample in eval. Aligned RNG is required for deterministic source comparisons; no silent eval fix was applied.
- Paper versus code: time is sinusoid+linear over age in days; mixer is two-layer nonlinear; global occupancy buffer is randomly initialized. Codebook buffers update by EMA, not direct gradient training. Warmup flag is parsed but unused in the released training loop.

## Frozen experiment

Depth1/4/8 x dropout.3/.4/.5; seed0;100epochs each;batch256;width512;4heads;K300including root;4096centroids;global projection256;Adam1e-4;weight decay1e-5;gradient norm clipping1;L1loss;train percentiles2/98prediction clipping;maximum3000batches per epoch (not reached on F1). Last tied validation minimum replaces checkpoint. The timing pilot used the equivalent direct `val<=best_score` condition; the final visible runner calls learner `last_validation_min` for this decision. The final runner also verifies token-cache hashes before loading batches. Exhaustive short-history equivalence is checked; no new training is claimed for that wiring change. Cross-config selection declared as lowest final re-evaluated validation MAE, first config breaks exact ties; historical aggregation unavailable. Test does not select. A clean run requires temporal PASS before entering the full gate. The visible `full_search(...,allow_invalid_source_replay=True)` exists only for explicit forensic replay and is not invoked by the budgeted clean gate.

## Executed pilot and cost stop

One depth1/dropout.3 pilot:3training batches/768queries,1validation batch/256queries; partial val MAE9.084049479. Test NOT_RUN. First-batch nonfinite parameter-gradient count0. Real-batch original model max output error1.9111e-6 under aligned RNG (atol/rtol1e-5). These are source-replay timing/mechanism results, not a valid benchmark score. Saved256predictions independently keyed to archive targets and rescored.

Measured train36.5313seconds for3batches and validation15.2573seconds for1batch. Linear extrapolation to30train+2valbatches projects395.8272seconds/epoch. Three shallow100-epoch configs projectUSD26.80; nine at that same shallow speed projectUSD80.41. Deeper configurations unmeasured; these are estimates, not measured full epochs or guarantees. Preparation, final evaluations, deeper-layer costs and overhead are additional. No deeper pilot needed after the shallow subset already exceeds budget and temporal contract independently fails.

Resource configuration is **one T4, two physical CPU cores,16GiB**:0.000164+2×0.0000131+16×0.00000222=USD0.00022572/sec. Current rates checked at https://modal.com/pricing. `labs/_budget_l145.json` reserves entire worker timeouts before dispatch and retains attempts. USD2overhead reserve covers build/startup/commit/storage and non-itemized costs; measured worker-body estimates do not claim to be an invoice. Final computed totals are in `_verify_l145_results.json`.

## Reproduce checks

From repository root:

```bash
.venv/bin/python labs/_source_l145.py
.venv/bin/python labs/_check_l145.py
.venv/bin/python labs/_mechanism_l145.py
.venv/bin/python labs/_figures_l145.py
.venv/bin/python labs/_build_l145.py
.venv/bin/python labs/_execute_l145.py
.venv/bin/python labs/_verify_l145.py
.venv/bin/python labs/_delivery_l145.py
```

Author cloud commands (already executed; immutable phase IDs reject repeats):

```bash
.venv/bin/modal run modal/l145_repro.py --phase prepare
.venv/bin/python labs/_collect_l145.py prepare
.venv/bin/modal run modal/l145_repro.py --phase pilot-1
.venv/bin/python labs/_collect_l145.py pilot-1
```

Complete guard (expected to stop before allocating compute):

```bash
.venv/bin/modal run modal/l145_repro.py::full
```

The prepared hash-verified L143 volume is mounted read-only. For a fresh independent setup, recreate that materialization using `labs/_full_l143.py::materialize` and its pinned archives/model revision, then verify the graph/task contracts. Historical hash identity is not guaranteed across changed environments. A corrected-cache experiment needs separate preparation, audit and budget approval; it cannot reuse this failed cache as valid evidence.

The default solution executes in an empty temporary directory with embedded source, PNG diagrams and compressed author audit packets. A second execution in the pinned Modal environment validates portability there; it does not establish live Colab. No push or deployment was requested.

Delivery history: initial pinned-notebook startup referenced a stale volume name copied from L144 and failed before worker dispatch; fixed to l145-relgt-evidence. Image-build overhead remains inside the USD2 reserve. Initial figure-count checker counted markdown cells rather than embedded images; corrected to count image occurrences. Visual inspection caught and fixed one diagram overlap. These are delivery failures, not hidden training retries.
