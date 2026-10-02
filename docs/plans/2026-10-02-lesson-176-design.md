# Lesson 176: few-shot ICL

Approved by the user on 2026-10-02 following the scoped two-track proposal.

## Objective and frozen experiment

Teach how to evaluate additional labeled context with unchanged pretrained parameters. Preserve L175's failed temporal audit and distinguish it from this separate RDB-PFN pipeline.

Track A: independently replay the full L169 selected published experiment (300 saved evaluations, 229050 predictions). This is reused evidence, not new inference. Source: RDB-PFN arXiv2603.03805v5 Appendix A.3/Tables6–10.

Track B: L176 Nested-Support ICL Evaluation: rel-f1/driver-dnf (702 test queries), rel-trial/study-outcome (825), RDBPFN/RDBPFN_single/TabICLv1.1, contexts64/128/256/512/1024, seeds0–9:300 fresh evaluations. Source a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL eaf789a9b25ee8486d6f48997ba076f850bbc30b. Reuse L169 authenticated checkpoint/data/runtime pins and label orientation. For each task/seed, use the original SHA256 seed derivation and NumPy1.26.4 default_rng choice of1024without replacement once; prefixes produce all smaller contexts. Identical ordered support for every arm. Source imputation remains support-dependent, and TabICL preprocessing/ensembling may depend on k; this estimates the response of the complete inference pipeline, not the isolated causal effect of labels alone. No test-driven selection, new fitting of pretrained weights, or silently repaired source pipeline. No k=0 claim.

Audit complete composite query and support keys, support uniqueness/nesting, train-only support membership and completed label horizons before the earliest query. Retain source DFS/historical arrival/lineage limitations; original DFS regeneration NOT_RUN. Independently rescore every prediction and paired seed difference. Changed sampling is a declared course intervention, not a new paper-table reproduction. Exact TabICL repeatability previously FAIL; do not upgrade it without evidence. Whole-paper reproduction and fresh pretraining NOT_RUN, learner PENDING_WRITTEN_DEFENSE.

## Budget and execution

USD10 absolute aggregate cap, USD8 planned stop, USD2 protected margin. Initial total estimate aboutUSD6. ReserveUSD3 for build/startup/storage/other overhead; L4+2physical CPU+16GiB rateUSD.00028372/sec verified against Modal pricing. Pilot600s maximum, then reserve up to7200s for main only if measured worst-case per-run extrapolation with3xmargin plus all reservations remains belowUSD8. All attempts/retries/validation count, no automatic retries, no silent downscaling. Remote nonblocking submission avoids prior synchronous RPC cancellation. Per-job outputs immutable; source/data fingerprints checked; no claims from partial grid. Local numerical aggregate cap3600s. Stop/report INCOMPLETE if budget or scientific gate fails; ship candid lesson and executable lane.

## Implementation and delivery tasks

1. Add meaningful failing tests for nested support, keyed scoring, paired completeness and cost admission. Implement visible relkit module, verify independent and wrong-solution cases.
2. Authenticate all inherited inputs; freeze nested schedule and source ledger. Build bounded immutable worker, run new pilot, audit timing and dispatch full remaining grid only on passing budget decision. Save raw predictions, receipts, failed attempts, and costs.
3. Replay all300L169evaluations independently; independently audit new predictions/supports and produce deterministic report. No new score used to select k or model.
4. Author connected HTML lesson and quick reference, model-specific support/query architecture and portable figures, live context explorer, student/solution notebooks with three live TODO/CHECK functions and explicit EXIT defense. Expose actual predictor/model/runner source, not just wrappers. Default notebook replays all saved evidence; paid fresh inference explicit.
5. Run standalone notebook in empty directory, browser1200/375, keyboard/reset, no-JS/print, figure visual inspection, source-inline parity, deterministic regeneration, local links and manifest galleries. Update resources/curriculum/notes and publication copy rules.
6. Stage intended deliverables preserving existing staged lessons171–175. Build real Pages from Git index and authenticate copied evidence. No push/deployment requested; live Colab NOT_CHECKED.

The writing-plans skill referenced by brainstorming is not installed. This approved implementation plan records the workflow directly. No additional approval or delegation is needed within this scope.
