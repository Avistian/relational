# Lesson 135 — Hyperparameter tuning on the REG

Approved by user 2026-09-27. Deliver a full tuning study and fresh selected published-baseline reproduction, within USD10 aggregate including all checks/retries.

Learning outcome: freeze a search space and budget, select using validation only, and compare the frozen winner with the default over fresh paired seeds. L134 measures sampled computation; L135 allocates it without letting test labels steer choices. L136 handles leaderboard claims.

Named baseline: RelBench v1 Table7 rel-f1/driver-position, full released data, ten epochs, five seeds0–4; pinned release9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Historical training commit is NOT_ESTABLISHED. Preserve source fanouts[128,64] and full-database preprocessing with documented paper/source distinctions.

Search: six fixed configurations, learning rates .001/.005/.01 crossed with fanouts[32,16]/[128,64]; seeds100,101; ten full epochs each, minimum validation MAE checkpoint (first on ties). Rank mean selection MAE over both seeds, tie-break by stable config ID. No test tables/labels/evaluation in search mode. Freeze winner file before final fits. Final seeds0–4, default and frozen winner, validation-selected checkpoints, one final test evaluation per fit. If winner equals default reuse identical paired condition explicitly. Single-task result, no architecture-superiority claim.

Budget: current Modal T4 .000164/s +2 cores*.0000131/s +16GiB*.00000222/s = .00022572/s. Twenty-three fit slots including pilot, each900seconds, reserve <=USD4.673004; allow separate checks with cumulative reservations <=USD7, retaining USD3 overhead. Estimate USD2–5; timed pilot before full dispatch; no automatic retries; no local full-data training. Every dispatch reserves worst-case worker time. Persist failures and no overwriting completed runs.

Deliver HTML/reference, visible canonical mechanism and inherited model/trainer, student/executed solution/prepared HTML, portable figures, live selection/budget/paired-difference exercises, frozen source/config/data provenance, exact operators and independent audits. Browser desktop/mobile/keyboard/reset/noJS/print and copied Pages validation. Do not infer learner mastery or live Colab/deployment from local execution.

Plan: contract tests then implementation; pin protocol and source; pilot; validation-only search; freeze; fresh final comparison; independent rescore/source/parity checks; build narrative/figures/notebooks; execute solution and validate browser/staging. No subagents requested. Referenced writing-plans skill is unavailable; implementation steps are recorded here.
