# B01-MATCHED-COMPARISON-AUDIT

Approved 2026-10-03. Full scope: all L200 saved RDB-PFN v5 Table 9 rel-f1/driver-dnf evidence, three configurations × ten paired support draws × 702 queries = 21,060 predictions. This is a complete selected saved-evidence audit. It is not fresh inference, pretraining, a new graph/flat experiment, or either benchmark paper's full reproduction.

## Frozen inputs and original protocol

`evidence/b01/source-lock.json` pins the copied L200 archive and inherited protocol by SHA256. The archive contains original raw predictions, complete keys, labels, support identities, receipts, prepared features, manifest, report, visible model, pairwise scorer and independent rank verifier. Extract only to a temporary directory. Never overwrite L200 or silently refresh its archive. Its internal manifest authenticates all files before metrics are read.

Original protocol: code a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL checkpoint eaf789a9b25ee8486d6f48997ba076f850bbc30b. RDBPFN model_eval00528; single-table-prior model_eval00360; TabICL0.1.3/v1.1 with 32 estimators. 11,411 train / 566 validation / 702 test; 512 training supports per draw. Draws 0–9 use the first four big-endian bytes of SHA256(rel-f1-dfs-2:driver-dnf:{s}) as NumPy default_rng seed; no replacement; model RNG42. Same support per draw across arms. Support medians, zero for entirely missing columns, population normalization and ±100 clamp. No query labels or identity features. Float32 source inference; AUROC comparison targets .7219/.6640/.7176 and descriptive tolerance .02. No new tuning or selection.

The [original complete protocol and fresh commands](l200-reproduction.md) remain available. The default B01 notebook executes only replay. Optional fresh inference requires its own budget and is outside this approval. Do not interpret the original script's inherited COMPLETE_SELECTED_REPRODUCTION label as new B01 inference: B01 explicitly reports COMPLETE_REPLAY.

## Executable audit and contracts

Run from the repository root:

```sh
.venv/bin/python labs/_budget_b01.py .venv/bin/python labs/_test_b01.py
.venv/bin/python labs/_budget_b01.py .venv/bin/python labs/_audit_b01.py
.venv/bin/python labs/_budget_b01.py .venv/bin/python labs/_verify_b01.py
```

The portable solution embeds the frozen evidence and visible functions. It checks all identities, every run, receipt scores, independent rank AUROC, paired differences and exact original report parity. Corrupted copies test altered bytes, missing/duplicate runs, wrong keys, support identities, labels and nonfinite probabilities. Additional B01 tests reject incompatible/unknown comparison contracts and incomplete pairings. Unknown fields cannot earn MATCHED.

MATCHED applies to the released-materialized-input comparison only. Matching declarations are not empirical proof of all scientific controls. Exact feature arrays are authenticated, but raw DFS regeneration, complete arrival provenance and historical identity remain unestablished. For a historical-availability claim the visibility field is NOT_ESTABLISHED. Different checkpoints, pretraining priors, model families and inference recipes remain explicit variants. Fixed configuration budgets do not mean equal measured runtime, training data or cost.

## Budget and evidence boundaries

USD0 cloud/API. 1800 aggregate local execution seconds including failures, retries, preparation and delivery checks; `evidence/b01/local-budget.json` accounts attempts. Stop at the cap, keep all 30 runs, and report INCOMPLETE rather than shrink scope. This is a local execution-time allowance, not a learner time limit or measured model-speed comparison.

Inherited deviations: released labels complement the reconstructed raw DNF orientation; checkpoint width96 differs from appendix128. Preserve both original labels and probabilities. Full feature arrival proof and historical identity NOT_ESTABLISHED; raw DFS reconstruction, fresh B01 inference/training, full TabArena and fair-RDB benchmarks NOT_RUN. RDBLearn's separate source-preprocessing gate is unchanged. Ten support draws share a test population. Learner B01 PENDING_WRITTEN_DEFENSE; L200 exit INCOMPLETE. Live Colab and deployment NOT_CHECKED.
