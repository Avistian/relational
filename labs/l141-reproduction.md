# Lesson 141 — RelGNN selected-task reproduction contract

Approved 2026-09-30. Selected paper: Chen et al., ICML 2025, arXiv 2502.06784v2, Table 2, rel-f1/driver-position, reported five-seed test MAE 3.798. This package implements the complete selected architecture and runs the full released task. It does not reproduce the whole 30-task paper or establish its historical five-run training identity.

## Sources and data

- RelGNN commit `cffdb8b54627e92c7dd112c1243dde739c90d35b`; verbatim MIT source files and SHA256 manifest in `sources/l141/` and `_sources_l141.json`.
- Hugging Face `tianlangchen/RelGNN`, revision `321e6f6e7af5d7546b637f147783fc28ab5d4a7a`, file `rel-f1_driver-position.pth`, SHA256 `3ba2b6e5c99bc0939d13debb6b06d8d0f0828148361662d54bab83f6c23943df`.
- F1 database archive SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; task archive `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e`.
- Training/validation/test queries: 7,453 / 499 / 760. Full model snapshot has 74,063 rows, 169,421 forward FK edges, 338,842 directed edges and 20 atomic routes. Preprocessing freshly materialized in this lesson; subsequent fits reuse its verified graph, not weights.
- GloVe model `sentence-transformers/average_word_embeddings_glove.6B.300d`, revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`.
- Runtime: Python 3.11, Torch 2.5.1+cu124, PyG 2.6.1, pyg-lib 0.4.0+pt25cu124, PyTorch Frame 0.2.3, RelBench 1.1.0 and `requirements-l117-runtime.txt`. Local default notebook uses the available CPU runtime; pinned execution is a separate report.

## Released model and reconstructed training

Table-specific ResNet encoders (internal width 128, 4 layers, default dropout 0.2),128 output channels; query-relative positional time encoding; one composite layer; four heads of 128 channels each; sum fusion and route aggregation; route-specific fact-root term; destination skip;512→128 projection; per-node LayerNorm/ReLU; scalar one-layer head. Source-specific fact updates are retained. Visible arithmetic in `relkit/relgnn_l141.py`; original PyG TransformerConv and SAGEConv are differential references, not hidden forward implementations.

Uniform temporal fanouts 128/64, two sampling hops, bidirectional sampled subgraphs, batch 512, 0 loader workers. Every timestamped occurrence is checked against its own query cutoff. The released full test-cutoff snapshot and its feature statistics are preserved; this is not train-only normalization. Timestamp legality is not historical arrival-time evidence.

Fresh reconstruction: seeds 0–4, Adam 0.005, 10 complete epochs, L1 loss, first strict minimum validation MAE. Every epoch visits all 7,453 training queries in 15 batches. An evaluation-mode materialization batch precedes optimizer creation and consumes one sampled batch; this is a documented reconstruction choice. Test is opened only after the selected checkpoint is frozen. Evaluation clips to training-target percentiles 2/98 = [2, 31]; training is unclipped. There is no post-test retuning, discarded seed or model retry.

The upstream entry point only loads a checkpoint. Historical optimizer, schedule, initialization sequence, seed list, model-selection history and feature-type cache are not fully released. These fits are `RECONSTRUCTED_TRAINING`, not an exact historical five-seed reproduction. Predeclared descriptive test tolerance ±0.20 MAE; seed sample SD uses ddof 1.

## Checkpoint compatibility and failed original replay

The unmodified released-checkpoint loading attempt fails: fresh seed 42 type inference gives qualifying numerical [`number`], categorical [`position`]. The checkpoint requires numerical [`number`, `position`] and has no categorical encoder for that table. The saved mean/std vectors match those two raw numerical columns in that order within 1e-6 relative/absolute tolerance. `_compat_l141.py` rematerializes only qualifying with numerical position, retains every other table/edge/weight, and records a separate graph hash and full before/after ledger.

This successful lane is `CHECKPOINT_COMPATIBILITY_REPLAY`. It establishes execution under an evidence-backed reconstruction; historical preprocessing remains `NOT_ESTABLISHED`. Fresh fits retain the frozen original categorical type. Do not treat differences between these lanes as a controlled architecture effect. The original failed loading trace and spend reservation remain in the evidence folder/budget ledger.

## Numerical and independent evidence

- Four CPU operator cases (ordinary/composite × nonempty/empty attention edges) agree exactly in float64 outputs; 46 parameter-gradient tensors plus input gradients agree within declared tolerances.
- All 7,554 primary held-out predictions (one compatible replay + five fresh fits) agree with the original whole-model architecture at shared weights/batches within atol/rtol 2e-4; observed maximum fresh error ≤9.54e-7, replay error 0.
- Raw pandas reconstruction independently rebuilds 8,712 labels and query populations. Target: mean positionOrder in `(cutoff,cutoff+60days]`. The source's lower-only past-year eligibility condition is automatically satisfied by future participants; no-future-race drivers have no target row. This is not an all-driver prospective cohort.
- All query keys/targets are compared directly with the checksum-verified task archive. Independent keyed MAE and sklearn MAE agree. Temporal traces and complete histories are audited.
- Real first-batch encoder gradients include 640 nonfinite entries in `results` numerical weights. The original model has identical nonfinite masks; matched-dropout finite gradients differ by at most 2.3842e-7. Preserve this released behavior. Finite predictions/source parity do not imply healthy optimization. A sanitized encoder is an unrun extension.
- Initial gradient comparison failed because stochastic dropout masks differed; its log is retained. Matching CPU/GPU RNG before each forward fixes the comparison. Two audit dispatch/configuration failures are also recorded; reservations are not erased. The first pinned-notebook attempt failed because its execution namespace was not registered for dataclass annotation lookup. The corrected notebook namespace passes; both attempt records remain.

Primary results: compatible replay validation 2.836661 / test 3.737381 MAE. Five fresh fits validation 3.216126 ± 0.095152 / test 4.259311 ± 0.267820 MAE. Fresh score `OUTSIDE_TOLERANCE`; historical training `NOT_ESTABLISHED`; whole paper `NOT_RUN`. Source/artifact hashes and per-seed metrics are in `evidence/l141/training.json`.

## Budget and exact execution commands

USD 10 aggregate including preparation, pilot, seeds, retries, validation and overhead. Reserve USD 3 overhead and at mostUSD 7 worker commitments. T4 + 2 physical CPU cores + 16 GiB costs USD 0.00022572/s at https://modal.com/pricing (checked 2026-09-30). Workers have explicit timeouts, no automatic retries, and immutable attempt markers. Reservations are pessimistic timeout allocations; measured worker-body time excludes startup/build/commit/storage and is not an itemized invoice. Final conservative reservations plus overhead total **USD 6.656664**. Known worker-body estimates total USD 0.080256; this excludes unrecorded failed diagnostics and the overhead above. See `_budget_l141.json` for every reservation and accounting scope.

Run from the repository root using `.venv/bin/python` / `.venv/bin/modal`. Existing phase reservations and evidence markers deliberately refuse duplicate dispatch; for a genuinely new experiment use a new volume/output identity and a newly approved budget ledger.

```bash
.venv/bin/python labs/_check_l141.py
.venv/bin/python labs/_parity_l141.py
.venv/bin/modal run modal/l141_repro.py --phase prepare
.venv/bin/python labs/_collect_l141.py prepare
.venv/bin/modal run modal/l141_repro.py --phase replay
# Above is the preserved strict-loading failure; use the documented compatibility lane:
.venv/bin/modal run modal/l141_compat.py
.venv/bin/modal run modal/l141_repro.py --phase seed-0
.venv/bin/python labs/_collect_l141.py seed-0
.venv/bin/modal run modal/l141_repro.py --phase remaining
.venv/bin/python labs/_collect_l141.py full
.venv/bin/python labs/_audit_l141.py
.venv/bin/python labs/_figures_l141.py
.venv/bin/python labs/_build_l141.py
.venv/bin/python labs/_execute_l141.py
.venv/bin/modal run modal/l141_notebook_check.py
.venv/bin/python labs/_delivery_l141.py
.venv/bin/python labs/_verify_l141.py
.venv/bin/python labs/_check_pages_checkout.py
```

For a new local/pinned notebook execution, set `RUN_FULL_REPRODUCTION=True`, choose fresh output directories and use the full GPU environment. The portable notebook contains graph preprocessing, complete model, trainer and compatibility helper. Its default gate is off; all three learner functions stay live. Full notebook validation runs one extra seed 100 and a compatible replay; neither enters the five primary seed statistics.

## Delivery and learner boundary

HTML lesson, reference, student/solution notebooks, executed HTML, portable figures, canonical source, protocol and evidence ship together. Desktop/mobile/keyboard/reset/no-JS/print and clean-index Pages checks are recorded in their reports. Live Colab and live deployment remain `NOT_CHECKED`; publication was not requested. Author execution never marks learner mastery: `PENDING_WRITTEN_DEFENSE`.
