# Lesson 173 — approved multi-task pretraining

User approved on 2026-10-02. Named experiment: L173 F1 Multi-task Masked-cell Pretraining.

## Frozen experiment

Use the complete authenticated L171 F1 tables and L172 schema. Targets are every observed numerical or categorical feature in dated tables. Train date < 2005-01-01; validation during 2005; test date >= 2006-01-01. Fitting uses training only. Untimed tables, keys, text, and timestamps are not predictive features or targets. Excluded cells and unknown-category targets are counted explicitly. Category UNKNOWN is reserved class 0, never a numeric magnitude. At least one training observation is required per task.

One target cell per example, always removed from context, every eligible target once per epoch. Context: first four other observed eligible feature columns in sorted name order from the same row; first four eligible feature cells from dated FK-parent rows whose dates are <= the query date, sorted FK/column order. Strict nonfuture event time is a course policy, not historical availability evidence. Same-row context defines autocomplete, not forecasting. No reverse links or ICL support contexts. The four KumoRFM axes are conceptual material; row/FK context here does not implement its architecture.

Column and categorical embeddings plus a learned numeric projection form 32-dimensional tokens. Separately average row and parent tokens; concatenate both with the target-column embedding; shared 96->64->32 MLP, then per-column linear prediction heads. Padded context has zero contribution. Huber delta=1 on normalized numerical targets; categorical cross-entropy includes reserved UNKNOWN. No text encoder, attention, global schema semantics, or cross-database transfer claim.

Arms cell-weighted and equal-task-weighted use the same initial weights, seed-specific permutations, masks, and batches. Seeds 0,1,2; 3 complete epochs; batch1024; Adam lr0.001, no weight decay. Task arm multiplies each example by N/(K*N_task), with counts computed only on training data. This is an unbiased minibatch estimator of the equal-task objective; do not renormalize by the tasks present in a minibatch. Select earliest minimal validation macro task loss for both arms, evaluate test only after selection. Save all epoch checkpoints, per-task curves, selected predictions, baseline predictions and full manifests. Report each task separately, aggregate loss explicitly, and do not combine MAE with accuracy.

## Budget and evidence

USD0 cloud/API; 3600 aggregate seconds for numerical preparation, test failures, training, verification and solution execution. Enforced subprocess-group timeout, no paid fallback or silent downscale. Delivery rendering/source retrieval excluded from numerical runtime. Stop and record INCOMPLETE on cutoff. Whole-paper training NOT_RUN; course fresh training measured; historical availability and cross-database transfer NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. No deployment requested.

RT v1 §3.3 supplies a comparison (numeric Huber and boolean BCE; mean masked-cell loss); multiclass categorical heads and task balancing are course deviations. RT's reported single pretrain takes ~16 A100 GPU-hours, above USD10 before overhead. KumoRFM-2 supplies the four-axis conceptual description, not a released executable training protocol used here.

## Implementation and acceptance

1. Write adversarial behavioral tests, observe missing implementation failure; build budget wrapper.
2. Authenticate source and inputs, implement target erasure, context admission, task weighting and model/training visibly.
3. Run six complete fits under the cap. Independently reconstruct data identities/context and recompute losses from saved logits; replay selected checkpoints and check initialization/permutation pairing.
4. Produce explanatory figures, interactive loss weighting, lesson/reference, portable student and solution notebooks with live TODO/CHECK functions and a gated fresh run. Execute solution including a fresh selected fit, separately label replay.
5. Check sources, controls, portable figures, notebook output, determinism, local links, navigation and actual Git-index Pages build. Preserve prior staged work.

The writing-plans skill was not found in available skill directories; this document provides the implementation plan directly.
