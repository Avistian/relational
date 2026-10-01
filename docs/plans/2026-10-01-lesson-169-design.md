# Lesson 169 — Scaling laws and open questions

Approved by the user on 2026-10-01. Implement in this session.

Goal: distinguish context scaling evidence from pretraining scaling laws and produce a defensible scaling-gap map.
Architecture: reuse authenticated L166/L168 task packets and frozen released predictors; extend only the context-size axis. Portable notebook independently replays all predictions and exposes live learner contracts. Shared course components provide retrieval, prediction, teachback and navigation.
Tech stack: Python/NumPy/PyTorch/TabICL, Modal L4, Markdown/HTML/JavaScript, nbformat/nbclient, matplotlib, Playwright.

## Frozen experiment
RDB-PFN v5 Tables 6–10, rel-f1/driver-dnf and rel-trial/study-outcome; contexts 64,128,256,512,1024; RDBPFN,RDBPFN_single,TabICLv1.1; seeds0–9; full702/825 query populations. 300 evaluations:60 authenticated reused512-context runs and240 fresh runs. Original stable seed and uniform without-replacement sampling independently per context (do not invent nested prefixes), source preprocessing, fixed weights,32 TabICL estimators. No tuning on test. Descriptive paper-distance tolerance0.02 AUROC; means/SD and paired effects, no claim of statistical equivalence.
Verify data/source/checkpoint hashes, query keys, labels, support readiness, released feature ordering and original receipts. Preserve reversed released labels, DFS regeneration NOT_RUN, historical availability/identity and exact schema exclusion NOT_ESTABLISHED. Whole-paper/fresh pretraining NOT_RUN.

## Budget
USD10 aggregate. USD3 overhead reserve. L4+2 physical CPU+16GiB USD0.00028372/s; at most21600 aggregate worker seconds, including retries and validations. Every reservation includes30s lifecycle margin. Pilot: both tasks, all3 models, context1024,seed0; timeout600s. Continue only if measured projection with3x margin fits remaining budget. No automatic retries. Local jobs bounded600s and thread-limited where practical. Stop rather than shrink protocol.

## Implementation and checks
1. Create labs/_check_l169.py with failing behavioral checks for exact run-grid completeness, independent context sampling, per-doubling gains and conservative claim classification; then implement labs/relkit/scaling_l169.py. Reject wrong learner functions and corrupt evidence.
2. Create labs/_prepare_l169.py, _run_l169.py, _fetch_l169.py and modal/l169_repro.py. Pin original sources and paper tables, build packet, run source sampling oracle and predispatch audit. Freeze immutable budget source hashes. Execute pilot; collect receipts and make measured cost decision; execute remaining234 fresh evaluations if gate passes.
3. Create labs/_audit_l169.py to independently authenticate original/current manifests, rescore all300 full predictions, compare paper values, quantify per-doubling gains/paired seed changes and record caveats. Record cost and stopped apps.
4. Create lessons/content/0169-scaling-laws-open-questions.md; labs/_figures_l169.py, _build_l169.py; interactive assets/scaling-context.js and assets/l169-lesson.js; reference/scaling-laws.html; student/solution notebooks and scaling-gap template. Show held-fixed/varied/measured matrix, full curve+SD, context versus pretraining distinction and extrapolation limits.
5. Execute standalone solution from empty directory. Independent metric/support/claim oracles; mutation checks; visible inline source parity; deterministic regeneration. Browser desktop/mobile all states, keyboard/reset, no-JS/print; inspect figures/screenshots. Update manifest, curricula, resources, preparation notes and record without claiming learner mastery.
6. Stage only intended artifacts, preserving existing index; clean Git-index Pages workflow check. No push/deployment requested. Final separates selected reproduction, delivery, pretraining and learner status.
