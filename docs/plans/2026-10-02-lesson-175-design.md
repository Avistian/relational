# Lesson 175 zero-shot evaluation implementation plan

Approved by the user on 2026-10-02 after the scoped reproduction proposal.

**Goal:** teach and execute an auditable unseen-database checkpoint evaluation.
**Architecture:** original RT-v1 and native context sampler, complete rel-f1/driver-dnf test population, frozen pretrained and fine-tuned checkpoints, independently keyed scoring. Separate released protocol, temporal audit, and historical identity claims.
**Tech stack:** Python, PyTorch, Rust sampler, Modal L4, HTML/JS, nbformat/nbconvert.

## Frozen experiment

L175 RT-v1 F1 Zero-shot Checkpoint Evaluation. Source rt-v1 commit 8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb; weights stanford-star/rt-v1 revision 299701dedae451f3dfa40717b831d9dc17c0e4e7. Original preprocessing repository hvag976/relational-transformer revision e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee is the initial compatibility target. Verify binary format/schema before dispatch; do not substitute RT-J or its preprocessing. Source code and byte hashes will be archived in labs/sources/l175.

Two fixed checkpoints: pretrain_rel-f1_driver-dnf.pt and finetune-from-contd-pretrain_rel-f1_driver-dnf.pt. Full official test population, no row cap; context seeds0/1/2, context1024, BFS256; six full evaluations. Same model dimensions12blocks,d256,8heads,FF1024. No gradients, new selection, or tuning. Use source sampling/typing/embedding semantics and record implementation deviations. Seed0 is the reference release lane; seeds1/2 are robustness extensions. Checkpoint comparator is a later released supervised checkpoint, not necessarily the original paper's supervised cell.

The paper permits target-task validation selection and historical labels in relational contexts. Database held out of gradient pretraining does not mean no target-label exposure. Audit per-query label masking, timestamps, preprocessing fit populations, all entity/cutoff keys and complete coverage. Stop any clean temporal claim when historical preprocessing or available-label semantics violate it. Keep released-protocol behavior visible rather than silently repairing it under the same name. Historical identity remains NOT_ESTABLISHED. Pretraining and fresh supervised fitting/whole-paper reproduction NOT_RUN.

## Aggregate budget

USD10 absolute maximum, USD8 planned-work stop, balance reserved for overhead. L4+2physical CPU cores+16GiB at .00028372USD/s: six aggregate container-hours aboutUSD6.13. All builds, preparation, attempts, retries, probes, inference and numerical validation count; credit discounts do not expand cap. Reserve before calls. Initial GPU probe capped600s; dispatch remaining only if conservative measured projection plus all charges/reserves <=USD8. No automatic retry, silent downscaling, or paid fallback. A blocked compatibility/forecast becomes NOT_RUN/INCOMPLETE, while the runnable audit/lesson still ships. CPU numerical verification cap3600aggregate seconds; no cloud/API costs outside this ledger.

## Tasks and checks

1. Archive and inspect original model/data/sampler/selection source and checkpoint/data metadata. Add behavioral tests for composite-key scoring, information-access classification and budget reservation, observe RED, implement labs/relkit/zero_shot_l175.py and verify GREEN. Pin environment and commands in labs/l175-reproduction.md.
2. Implement labs/_run_l175.py and modal/l175_repro.py with immutable output folders, hard resource/time limits, prediction coverage and exception receipts. Probe compatibility and timing before full six-run dispatch; archive remote evidence locally and independently rescore all rows. Reject missing/duplicate keys and altered labels.
3. Build lessons/content/0175-zero-shot-evaluation.md, reusable assets/zero-shot-evaluation.js, architecture/information/scoring figures, reference/zero-shot-evaluation.html, student/solution notebooks. Expose original RT forward implementation and meaningful live TODO/CHECK/EXIT evaluation code. Execute solution and record evidence status accurately.
4. Build/verify via labs/_build_l175.py, _verify_l175.py, _execute_l175.py and _delivery_l175.py. Inspect desktop/mobile, keyboard/reset, print/no-JS and portable notebook figures; test wrong learner implementations. Regeneration deterministic except measured timing.
5. Update manifest, curriculum, source list, prepared-learning notes and Pages workflow. Stage only intended L175 and shared updates, preserving existing index; run full real Git-index Pages build. No push/deployment requested. Learner PENDING_WRITTEN_DEFENSE; live Colab NOT_CHECKED.

Execution proceeds in this session under the user's approval. No additional permission gate or delegation is needed.
