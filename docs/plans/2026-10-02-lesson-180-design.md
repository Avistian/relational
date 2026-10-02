# Lesson 180 · public encoder fine-tuning checkpoint

Approved 2026-10-02. Named target: RT-v1 supervised fine-tuning on rel-f1/driver-dnf, paper v1 §4.1 and Appendix D. Complete selected audit/replay only; full public-encoder training remains NOT_RUN and practical exit INCOMPLETE.

## Frozen scope and limits

Public initialization pretrain_rel-f1_driver-dnf.pt at stanford-star/rt-v1 revision 299701dedae451f3dfa40717b831d9dc17c0e4e7. Source revision 8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb and preprocessed release e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee inherited from L175. Byte authentication of downloaded weights NOT_CHECKED; historical training lineage NOT_ESTABLISHED. Do not download weights after a failed scientific gate.

Paper schedule: approximately 33k steps, 1024 cells, global batch256, AdamW LR1e-4, zero weight decay. Released example: max_steps32769, per-rank batch32 on eight GPUs, seed0. Audit actual stopping, evaluation, checkpoint retention and selection behavior rather than assuming paper/source equivalence. Full validation/test identities required; test must not select checkpoints. No shortened training, different model, or temporal repair is authorized.

Authenticate and replay all L175 saved contexts (3×702×1024 cell slots), target masks, target-window checks and original complete keys. Preserve observed future-schedule-cell failure without asserting that schedule dates prove outcome leakage. This is saved sampler evidence, not new contexts, gradients or inference. Audit source configuration and compute full-run GPU-only price floor with decimal arithmetic. Independent checks must reject corrupt packets and incorrect learner implementations.

## Budget

USD0 new cloud/API. 1800 aggregate local numerical seconds, including preparation, failures, checks and solution execution. Full run is already over the standing USD10 ceiling: approximately USD25–30 GPU-only for one reported 1.5h eight-A100 run. No paid pilot or dispatch. Preserve every blocker and stop at cutoff.

## Implementation sequence

1. Pin sources and inherited evidence, preserve all prior staged work; add an all-attempt runtime watchdog.
2. Test live temporal audit, cost and checkpoint-decision functions; implement independent whole-packet verification and a gated reproduction entry point. Archive original model/trainer visibly; do not claim post-gate training is validated.
3. Connected short HTML lesson, reference, RT-specific architecture and worked trace, reusable interactive checkpoint explorer, self-contained student/solution notebooks with meaningful tasks and explicit mastery gates.
4. Execute solution from empty directory; check source parity, negative cases, deterministic build, desktop/mobile, keyboard/reset, no-JS/print, figures and manifest navigation. Stage only intended deliverables; build real Pages from Git index and hash copied evidence. No push/deployment requested.

The brainstorming-referenced writing-plans skill is unavailable after search; this approved design records the implementation sequence. Public-checkpoint fresh fit and personal mastery must not be awarded by author preparation. L179 is not present; include prerequisite retrieval without inventing completion.
