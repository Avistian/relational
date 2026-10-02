# Lesson 174 — approved fine-tuning protocol

Approved by the user on 2026-10-02. Named experiment: **L174 F1 Temporal Adaptation**.

## Frozen protocol

Reuse all 21 L173 autocomplete tasks, complete authenticated population, original train-only normalization/vocabulary and row/FK context policy. Source training before 2005; source checkpoint selected on 2005 validation only. Source arm is cell-weighted, fixed irrespective of L173 test outcomes. For each seed 0/1/2, load its selected checkpoint. Adapt on calendar 2006 (6,429 cells), validate during 2007 (6,074), test from 2008 (106,757). Reusing later data already evaluated in L173 makes this retrospective same-database, same-task adaptation, not a pristine confirmatory test, unseen-task transfer or forecasting.

Four arms: frozen encoder/trainable original heads; all original parameters trainable; frozen encoder plus residual 32->8->32 ReLU adapter and trainable original heads; same original architecture trained from scratch. Adapter up projection and bias start at zero (identity); down projection uses seed-fixed random initialization. Insert after the shared 32-wide representation, before the heads. Frozen/full/adapter share initial original weights; scratch is a different initialization by definition. Every arm shares each seed's example order. Original heads are retained for pretrained arms; they are not reset. Batch1024, ten complete epochs, Adam lr0.001 default betas/epsilon, no weight decay. Equal-task training weights derive only from 2006 counts. Earliest minimal validation macro task loss selects one of ten checkpoints; test is evaluated only afterward. No learning-rate or adapter-width search; equal updates do not imply equal compute or tuned-optimal arms.

Report all seeds and paired differences, sample SD, macro loss and separate per-task raw MAE or accuracy; do not pool those physical metrics. Save every checkpoint, full keyed logits and targets, train/validation curves, update counts, parameter hashes and trainability. Unchanged source-checkpoint baselines and 2006 training-only numeric-mean/Laplace-frequency baselines share test identities. Timings are measured and not deterministic equality targets.

## Evidence and budget

USD0 new cloud/API. Aggregate numerical cap3600 seconds includes preparation, behavioral checks and failures, twelve author fits, independent verification and separate notebook validation fit. Prior planning inspection used two read-only population-count probes (one failed from timestamp units/padding); account their combined observed .082seconds in the ledger. Rendering, source retrieval and delivery checks are outside numerical execution. No paid fallback or silent data/epoch/seed downscaling. Stop/report INCOMPLETE if cap is reached. Existing L173 training is reused, not new pretraining.

Houlsby et al. supplies the adapter principle; the small pooled MLP and one end-of-encoder adapter are course deviations from BERT adapters. RT supplies a published relational fine-tuning comparison, not architectural parity. Its reported12A100GPU-hours per fine-tuning run exceeds USD10 at checked Modal rates even before other costs. Whole-paper reproduction NOT_RUN; cross-database/unseen-task transfer and historical availability NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. No deployment requested.

## Implementation and acceptance

1. Behavioral tests fail on missing implementation, then verify trainability, identity adapter, nonzero adapter learning, temporal bounds and earliest validation selection; wrong learner policies must be rejected.
2. Authenticate L173 source packet/checkpoints; pin primary readings and runtime. Implement visible model, optimizer policy and full fresh runner with overwrite protection.
3. Run twelve fits under one budget. Independent NumPy scoring, complete identity checks, checkpoint replay and byte-level frozen-parameter audits must pass.
4. Build a 20-minute lesson, model-specific architecture and learning-curve figures, interactive gradient/update exercise, quick reference, student/solution notebooks with live TODO/CHECK/EXIT functions and authenticated portable data. Execute a fresh notebook fit separately from saved author evidence.
5. Validate source parity, browser desktop/mobile, keyboard/reset, no-JS/print, notebook output, deterministic builders, links, manifest galleries and full Git-index Pages build. Preserve all existing staged work; stage only this lesson's additions and shared-file edits.

The writing-plans skill is unavailable in the installed skill directories; this approved document provides the implementation plan directly.
