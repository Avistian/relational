# B10 · RT-v1 cells, tasks and relational attention

Approved 2026-10-03. Named paper target: **RT-v1, arXiv 2510.06377v1 Table 1, rel-f1/driver-dnf, target database excluded from pretraining, 82.0% AUROC**. Complete mechanism checks and saved-context replay are separate experiments. Selected benchmark status: **INCOMPLETE_TEMPORAL_GATE**; checkpoint inference **NOT_RUN**. Whole-paper pretraining and benchmark reproduction **NOT_RUN**.

## Frozen identity

- Original source `8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb`, archived in `sources/b10/upstream/` with complete model, trainer, preprocessing, dataset wrapper and Rust sampler.
- Original preprocessed dataset `hvag976/relational-transformer`, revision `e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee`.
- Weights `stanford-star/rt-v1`, revision `299701dedae451f3dfa40717b831d9dc17c0e4e7`, file `pretrain_rel-f1_driver-dnf.pt`. Weight bytes **NOT_DOWNLOADED / NOT_AUTHENTICATED**. This is release provenance, not independently reproduced pretraining lineage or historical checkpoint identity.
- Full 702 query keys `(driverId, cutoff)`; labels reconstructed from raw F1 results (`statusId != 1` during `(cutoff, cutoff+30 days]`). Do not substitute RDB-PFN's reversed released labels.
- 1024 cells, BFS width 256, original sampler. Seed0 release-aligned, seeds1/2 explicit robustness extension. Three context seeds are not three independently pretrained models.
- Original configuration: 12 blocks, hidden256, 8 heads, FF1024, 384-coordinate MiniLM name/text embeddings, bfloat16. Native batch8 rather than training batch32 is an inference-memory deviation inherited from L175; per-node RNG is independent of batch position.
- Evaluation only. No B10 optimizer, epochs, target fitting or checkpoint selection. Authors used target-task validation for released checkpoint selection; this is not a claim of no target-label access. Prior completed task labels can enter context under the declared rolling-label policy.
- Score AUROC on the complete population, individually by seed, then mean/sample SD. Seed0 vs printed82.0 is descriptive only until exact historical identity is established; no tolerance-based MATCH/CLOSE verdict is authorized now. No predictions exist.

## B10-MECHANISM

Three seeds0/1/2, one fixed three-table/ten-slot fixture including padding; two blocks, width16, four heads, FF32, six-coordinate illustrative name/text vectors, CPU float64. Model parameters copied from original source. Original mask expressions/model/loss remain unchanged; its sparse CUDA attention and block-mask builder are replaced with dense PyTorch SDPA math. Course mirror implements explicit safe softmax. Compare every typed output, loss, every parameter gradient and every floating input gradient with absolute tolerance1e-9. Verify permutation equivariance, hidden-target intervention, schema-name sensitivity, strict future-row selection and empty-neighbor zero output. Twenty additional random graph fixtures use independent scalar mask loops. Three deliberately wrong learner functions must fail.

This is a finite-fixture implementation check. It does not validate the original CUDA/bfloat16 kernels, production dimensions, pretrained semantics, tokenizer preprocessing, or benchmark performance. Illustrative six-coordinate name vectors do not reproduce MiniLM embeddings. The stricter `eligible_rows` policy is an explicitly separate course intervention; it is not the original sampler. No small-model training is used as paper-result evidence.

## B10-CONTEXT-REPLAY

Authenticate all original source and input hashes, independently reconstruct all2106query labels and keys, and rescore2,156,544saved cell slots. No native resampling. The complete replay retains385future-dated cells in77contexts,432,050untimed cells, zero exposed query targets and zero visible task labels with incomplete outcome windows. Future cells are scheduled race attributes: this violates the stated event-time bound, but does not prove that their content was unavailable historically. Arrival evidence is missing. Original broad-population preprocessing is an additional strict train-only reconstruction gap.

A repaired sampler, new preprocessing, current RT-J checkpoint, or different name embedder is a different experiment. Do not edit a receipt to unlock this one.

## Commands from repository root

```bash
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_prepare_b10.py
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_verify_b10.py
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_reproduce_b10.py
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_build_b10.py
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_execute_b10.py
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_delivery_b10.py
```

Explicit inference request (expected nonzero `BLOCKED_TEMPORAL_AUDIT`, before model loading):

```bash
.venv/bin/python labs/_budget_b10.py .venv/bin/python labs/_reproduce_b10.py --run --checkpoints /path/to/rt-v1-checkpoints --out /path/to/new-b10-run
```

`_infer_b10.py` preserves the native-source inference loop for the one checkpoint and three seeds. Its post-gate GPU path is **UNVALIDATED**; the wrapper's refusal is tested. Original RT dependencies include its compiled Rust extension and CUDA runtime; the local teaching requirements are not a historical training environment. The original full trainer/configuration are readable in the notebook appendix. No cloud dispatch is authorized or needed while the input gate fails.

## Budget and delivery

USD0 paid compute; aggregate3600seconds local preparation/numerical execution/verification including failed attempts, accounted by `_budget_b10.py`. Outer process-group timeout stops at the remaining allowance. No retries beyond the cap and no silently reduced query/seed matrix. General lesson writing is not a compute run. The portable notebook executes fresh mechanism checks plus an embedded saved audit replay; it neither downloads model weights nor claims fresh native contexts. Author checks do not establish learner mastery: **PENDING_WRITTEN_DEFENSE**. Live Colab and deployment are separate, unverified/unrun lanes.
