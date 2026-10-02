# L175 RT-v1 F1 Zero-shot Checkpoint Evaluation

Approved 2026-10-02. **Status: INCOMPLETE_TEMPORAL_GATE.** Complete three-seed native-sampler audit; all six checkpoint evaluations NOT_RUN. No measured model AUROC, GPU inference, new pretraining, or new supervised training.

## Named experiment and immutable inputs

Two fixed released checkpoints on the full rel-f1/driver-dnf test population: `pretrain_rel-f1_driver-dnf.pt` and `finetune-from-contd-pretrain_rel-f1_driver-dnf.pt`, seeds0/1/2,702 queries each,1024cells,BFS256. Source `stanford-star/relational-transformer` rt-v1 commit `8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb`; checkpoint repository `stanford-star/rt-v1` revision `299701dedae451f3dfa40717b831d9dc17c0e4e7`; preprocessing `hvag976/relational-transformer` revision `e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee`. Checkpoints were selected by filename/provenance before evaluation and were NOT downloaded or executed after the temporal gate failed. Their bytes therefore are not locally authenticated. Never substitute RT-J or the current preprocessed layout.

Released original source is archived in `sources/l175/upstream/`, including the full forward model, trainer, dataset wrapper, preprocessor and native sampler. Source ledger pins every file. Original model:12blocks,width256,8heads,FF1024,384-wide MiniLM name/text vectors; RMSNorm and column/feature/neighbor/full masked attention; boolean prediction head; bfloat16. No target fitting or new checkpoint selection. Target-task validation was used by the authors for released checkpoint selection, and target labels may enter context. Historical exclusion is release provenance, not independently reproduced training lineage.

The fine-tuned checkpoint is a later released continued-pretraining-initialized comparator; it is not asserted to be the checkpoint behind the original paper's supervised result. Seed0 follows the released sampler seed; seeds1/2 are a declared robustness extension. One database and context seeds do not establish cross-database performance. No numerical tolerance verdict exists because predictions were not run.

## Executed audit and stop decision

Native Rust `common.rs`, `fly.rs`, `lib.rs` ran unchanged on all702test nodes per seed. Metadata locates nodes97606–98307. Batch8 replaces the training example's32 for memory, with per-node RNG seeded independently of batch position. Shuffle epoch0; no context truncation beyond original1024. Inference-only Cargo manifest removed unused preprocessing dependencies (polars,parquet,glob,indicatif,serde_json); source code and original lock remain archived. Rust1.88.0,maturin1.9.4,Python3.12,numpy2.2.6,ml_dtypes0.5.3,huggingface-hub0.34.4. Modal build IDs and compilation receipts are in audit-cloud logs.

Full saved context arrays: `evidence/l175/audit-3/contexts-{0,1,2}.npz`. Independent Python loops rescore2,156,544cell slots, re-identify every composite query key, and reconstruct every query label from26,080original F1 results rows.702unique keys/seed;495positive labels each. Query labels use the actual DNF SQL (`statusId !=1` within `(cutoff,cutoff+30days]`), not the reversed RDB-PFN release labels.

- Exposed query targets:0. Same-time visible labels:0. Visible labels with unfinished30day outcome window:0.
- Historical visible task-label cells:29,988 across2,043contexts.12,351are from earlier test-period task rows across1,901contexts; these meet the horizon-completion test. This is compatible only with a declared rolling-label policy, not a sealed offline test-label embargo.
- Future-dated cells:385 across77contexts (seed0:130/26;seed1:130/26;seed2:125/25). All are five schedule fields of `races` rows. Example query cutoff2011-03-27 00:00UTC encounters race time2011-03-27 06:00UTC. FK→PK traversal has no timestamp rejection.
- These are scheduled-event attributes, not measured future outcomes. We establish a failure of the declared event-time bound, NOT proof that those schedule fields were unavailable historically. No arrival logs establish availability.432,050untimed cells further prevent an availability proof.
- Original source database-column and global datetime fitting use broad table populations; only task validation/test column stats are replaced from train. Source-faithful behavior is separate from a strict training-only reconstruction.

The approved clean temporal gate fails. No GPU timing probe or full inference was launched. `_run_l175.py` rejects the failed receipt before importing the model or loading weights. Repairing the sampler, allowing future scheduled fields with arrival evidence, or recomputing preprocessing would define another protocol and cannot silently retain this experiment's identity.

## Reproduce the completed local audit

From repository root, using the course environment (`ml_dtypes==0.5.3` added):

```bash
.venv/bin/python labs/_budget_l175.py .venv/bin/python labs/_check_l175.py
.venv/bin/python labs/_budget_l175.py .venv/bin/python labs/_verify_l175.py
.venv/bin/python labs/_figures_l175.py
.venv/bin/python labs/_build_l175.py
.venv/bin/python labs/_budget_l175.py .venv/bin/python labs/_execute_l175.py
.venv/bin/python labs/_delivery_l175.py
```

The portable notebook authenticates its embedded packet, runs the live student functions over every saved context, and independently reconstructs all query labels. This is saved-context replay, not a fresh native-sampler or model run. It makes no paid call. The original model, Rust sampler and six-run inference operator are visible in notebook appendices. The student must complete three functions and write a defense.

## Native sampler / gated full inference operators

`modal/l175_repro.py` is the original cloud audit operator. Its completed attempt IDs cannot overwrite evidence; `--phase` selects a new immutable attempt after an aggregate reservation. Image compilation precedes worker execution; reserve/limit the whole CLI using `_dispatch_l175.py`. No runtime source changes are needed to replay the same sampler.

```bash
.venv/bin/python labs/_dispatch_l175.py --phase audit-4
```

For the six-run GPU operator, install the original RT dependencies (PyTorch2.6.0,einops,ml_dtypes,maturin_import_hook,compiled rustler), place the authenticated original data in `~/scratch/pre/rel-f1`, and obtain the two named checkpoint files from the fixed revision. The entry point below is deliberately blocked by the saved audit. Its post-gate GPU path remains UNVALIDATED; it is not advertised as a successfully reproduced model run.

```bash
.venv/bin/python labs/_run_l175.py \
  --source labs/sources/l175/upstream \
  --checkpoints /path/to/rt-v1-checkpoints \
  --audit-dir labs/evidence/l175/audit-3 \
  --out /path/to/new-l175-inference
```

Expected current result: `BLOCKED_TEMPORAL_AUDIT`, before any checkpoint download or compute. Do not edit the receipt to bypass it.

## Budget, attempts and boundaries

USD10absolute cap; USD8planned-work stop; USD2overhead reserve. L4+2CPU+16GiB rate.00028372USD/s; six hours would beUSD6.128352, conditional on a passing audit and timing forecast. Actual work used CPU sampling only. Three900secondCPU reservations at.00006172USD/s (each includes30seconds margin), plusUSD1build reservation andUSD2overhead reserve: conservative reserved ceilingUSD3.1721988, belowUSD8. This is a reservation ceiling, NOT an invoiced bill. Provider invoice NOT_CHECKED. Unused GPU time was not purchased.

Attempts: unsupported image timeout argument failed locally before dispatch; first remote worker import failed at initialization; second worker failed because NumPy's bfloat16 registration was missing; third completed. All retained in the ledger/logs; no failed attempt promoted to evidence. Initial Modal directory download overwrote its destination path; collection was repeated file-by-file without rerunning sampling. Local missing `ml_dtypes` verification failed before work and was corrected. CPU numerical cap3600seconds covers recorded local validation and solution attempts. Cloud apps stopped; no inference retries after the scientific gate failure.

Whole-paper reproduction / fresh pretraining / fresh supervised fit NOT_RUN; historical identity and arrival-time legality NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Delivery, live Colab and deployment are separate statuses. No deployment was requested.
