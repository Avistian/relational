# Lesson 154 approved design and implementation plan

Approved in chat 2026-10-01. Named experiment: L154 RelBench portfolio evidence replay. Synthesize L151 study-outcome, L152 driver-position and L153 site-sponsor-run against RelBench v1 Tables 6–8. No new model or paid training. Additional cloud spend USD0; standing aggregate cap USD10 is not permission to restart L153. Its full recommendation run stays INCOMPLETE and test NOT_RUN.

Goal: reproduce the report from saved predictions, frozen source/protocol hashes and complete seed sets; distinguish published baseline context from fresh matched comparisons. Classification reference seeds0–4 and course-selected seeds10–14 stay separate; regression seeds0–4; recommendation pilot remains validation only. Do not average heterogeneous metrics or count absent results as zero. Missing fresh FE/tree comparators prevent a local superiority claim. Whole paper and historical identity remain unestablished; author audit is not learner mastery.

Architecture: pure reporting functions validate complete runs, orient within-task differences and gate portfolio claims. A read-only replay adapter independently rescores saved predictions using complete query keys. A frozen manifest pins all consumed inputs; a portable notebook embeds the same packet and visible executable code. Generated HTML/report/notebooks share one builder and native shared styles.

Tech stack: Python stdlib/numpy, nbformat/nbclient, existing lesson CSS, JavaScript widget, Playwright.

1. Create labs/_check_l154.py with rejection checks for mixed protocols, duplicate/missing seeds, nonfinite scores, mismatched metric units, pilot promotion and unsupported aggregate claims. Run against stub module labs/relkit/portfolio_l154.py; verify failures, then implement and rerun.
2. Create labs/_replay_l154.py and labs/evidence/l154/input-manifest.json. Pin actual inputs once, then reject any altered bytes. Rescore complete L151 reference and selected tracks, L152 five seeds and L153 pilot; preserve all population/selection boundaries. Generate report.json and report.md locally. Cross-check metrics using independent sklearn/vectorized oracles and corruption tests.
3. Create lessons/content/0154-portfolio-synthesis.md, labs/_build_l154.py, assets/portfolio-evidence.js and assets/portfolio-evidence.css. Deliver lesson, reference, portable student/solution notebooks with three live functions and an exit defense; generate report from learner functions. Use the existing component library.
4. Run labs/_execute_l154.py in an empty directory, export HTML; build twice to verify determinism. Test notebook source parity, real evidence results, desktop/mobile widget interventions, keyboard/reset/noJS/print, and local links.
5. Update manifest, plan/year-4.md, labs/README.md, RESOURCES.md and NOTES.md; no mastery record. Extend Pages copying, stage only intended deliverables while preserving existing staged changes, and run labs/_check_pages_checkout.py from the Git index. No deployment requested.

Implementation proceeds in this session under the approved scope. No further design handoff or approval is needed. The local writing-plans skill's suggested worktree would omit the staged L151–153 inputs; preserve and use this authorized workspace instead.
