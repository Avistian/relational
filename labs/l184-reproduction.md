# L184 — GelGT reproduction contract and observed decision

Approved target: Gaussian Relational Graph Transformer, arXiv2605.15575v2 (2026-09-28), Table2 rel-f1/driver-position. Published test MAE3.7345 ± .1200; uncertainty convention and exact historical seeds unverified. Whole21-task benchmark is outside this selected experiment. Actual status: **INCOMPLETE_SOURCE_TEMPORAL_GATE**; fresh training **NOT_RUN**; measured model MAE is null. $0 cloud/API.

## Frozen evidence

Official release commit1997b2c2f480ce5d3cbdb48d46f33cc303f5feb4. Source URLs and SHA256 in sources/l184/source-ledger.json. Unmodified model, trainer, encoders, local attention and sampling code archived. Packet file hashes in evidence/l184/input-manifest.json. Cached F1 database archive ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; task archive775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e. All9database tables and3task splits embedded. Current cached bytes are authenticated; their use in the authors' historical runs is NOT_ESTABLISHED.

Full train/validation/test populations:7453/499/760, all8712labels independently reconstructed via SQL. Target averages positionOrder in (cutoff,cutoff+60days], using the released driver's eligibility condition. The source eligibility subquery has a lower time bound only; preserve this fact without calling it a prediction-time feature policy. Query identities are (driverId,date). Label reconstruction does not establish historical feature availability.

## Published / release discrepancies and unresolved details

| Setting | Paper Table12 | Release argparse default |
|---|---|---|
| Layers | 4 global/model layers | num_layers1 (local attention depth); outer model loop fixed1 |
| Batch |64|512|
| Max steps/epoch |500|3000|
| Warmup |10|1000 (argument; actual optimizer loop has no warmup schedule)|
| Candidate/retained regression tokens |500/300|800/300|
| Attention/FF dropout |.3|.1/.1|
| Width/heads/epochs |512/4/10|512/4/10|
| Adam lr/weight decay |.0001/.00001|.0001×world_size/.00001|
| Seeds/aggregation |not authenticated|single default42; no complete experiment sweep|

Task selection is driver-position (release defaults driver-top3). Regression uses L1 loss and clamps evaluation outputs to training2nd/98thpercentiles. Source chooses checkpoints by validation MAE (ties choose latest); it evaluates test during training. Source global maximum lag is calculated across train/val/test and initializes Gaussian centers; this is a fitting-scope discrepancy needing explicit treatment. Source attention passes dropout_p unconditionally into scaled_dot_product_attention, including eval mode. The local parity test uses dropout0 and cannot certify ordinary evaluation determinism. Requirements list RelBench1.1.0, Torch2.5.1, Frame.2.5; local audit uses installed CPU libraries and records versions separately. No historical environment equivalence claimed.

## Blocking temporal counterexample

utils.py local_nodes_hetero keys S[table][entity] without query cutoff. _precompute_sampling retrieves by entity within each10000row chunk. Every selected split fits one chunk. Distinct key losses: train6682, validation452, test704. These counts quantify cache identity collisions, not actual leaked row counts.

Execute unchanged original gather, per-seed sampler and local_nodes functions with a serial Pool adapter. Synthetic driver0 has neighbors at days5and15. Queries at days10and20 produce2and3tokens; the returned cache has one entry holding the late context. Retrieving that context for day10 admits the day15node. Stored lag is relative to the late query, so checking cached nonnegative lags alone would miss this. Raw timestamps and full query keys are required.

This independently sufficient blocker prevents dispatch. No silent repair, alternate seed sweep, shortened fit, or numeric paper comparison is presented. A future declared repair needs composite cache keys, per-query raw-time audit, invalid-padding masks, deterministic eval policy, resolved configs and seed aggregation, fitting-scope decisions, full input/output/gradient checks and a measured aggregate runtime forecast. Post-gate cloud operator NOT_VALIDATED.

## Verified mechanisms and deviations

Course temporal BFS is deterministic sorted two-hop traversal, strictly earlier timestamps, rejects untimed context, protects the seed. Source admits <= and untimed rows; paper uses both strict and non-strict wording. Course semantic refinement refuses a budget smaller than the seed+one-hop population, rather than using tied sentinel scores. These are explicit teaching policies, not a repaired replica.

Gaussian exp(-((lag-mu)/sigma)^2) matches release kernels using effective sigma=abs(raw)+1e-5. Full original attention EncoderLayer tested with dropout0 in float64 against explicit QK/softmax/V arithmetic, including input gradients. Kernel errors0; forward4.44e-16; gradient2.15e-15. These are local mechanism evidence, not full model, trainer, checkpoint or paper-result parity.

## Commands and cost

From repository root, using its virtualenv:

```
.venv/bin/python labs/_budget_l184.py .venv/bin/python labs/_verify_l184.py
.venv/bin/python labs/_budget_l184.py .venv/bin/python labs/_run_l184.py --preset paper
.venv/bin/python modal/l184_paper_repro.py --preset paper
.venv/bin/python labs/_build_l184.py
.venv/bin/python labs/_budget_l184.py .venv/bin/python labs/_execute_l184.py
```

Admission commands intentionally exit2. They reauthenticate inputs and execute the scientific counterexample; they do not create cloud jobs. smoke/closer/paper all preserve the same gate. There is no working full reproduction claim hidden behind a preset name. Budget ceiling$10total, plannedstop$8+$2reserve; local numerical3600seconds including failures and notebook verification, recorded in evidence/l184/budget.json. No paid pilot is needed to diagnose the earlier scientific stop. Delivered lesson and executable audit can be complete while selected reproduction remains incomplete. Learner PENDING_WRITTEN_DEFENSE; live Colab and deployment NOT_CHECKED.
