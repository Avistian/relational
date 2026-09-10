# Individual lesson audit and repair, 2026-09-10

User requested an in-depth audit and repair of lessons 047–070, using 042–045 as the quality baseline, with a fresh subagent for each lesson, sequentially. This is authorized repair work; implement fixes, not merely a review or a proposal. Do not change the mission or record learner mastery.

The user subsequently explicitly authorized **commit and push each lesson separately**. The parent agent owns commits, pushes and publication verification after each lesson's review/checks. Subagents should finish their scoped repair and report, then return to the parent without committing. This supersedes earlier no-commit/no-publication wording in assignments.

User emphasis added during execution: **pay particularly close attention to lessons 055–070, which the user finds shorter and lower quality. Extensive reading of the primary papers and substantial expansion may be needed.** For these units, read the relevant full paper sections and appendices, not just abstracts or current lesson citations. Develop richer motivation, derivations, worked examples, experiment explanations and lab guidance. Audit whether reduced/surrogate mechanisms actually teach the load-bearing paper contribution; improve alignment when feasible. More words alone are not the outcome. Do not restrict the repair to small additions or evidence disclaimers when the lesson itself remains thin.

The user explicitly reaffirmed: continue in numerical order; the 055–070 emphasis is not a request to jump ahead.

## Working method

Read MISSION.md, relevant NOTES preferences, .agents/skills/{teach,lesson-pedagogy,lab-authoring,lesson-visuals}/SKILL.md, and representative portions of lessons/labs 042–045. Inspect the assigned lesson completely, its depth/content source, builder, implementation, student and solution notebook, reference, visuals, provenance, evidence and checkers. Compare actual explanatory quality, not section counts or length. Existing reviews/lessons-047-070-depth-and-architecture.md describes an earlier prose/visual pass; it does not certify paper fidelity.

Use primary paper and official implementation sources, browse to verify sources. Pin version/section/function for substantive claims. Inspect all central theoretical, architectural and empirical claims; qualify/remove unsupported claims. Distinguish code parity, procedure fidelity and result reproduction. A `paper` preset or matching-looking number is not reproduction. If exact reproduction is impractical, explain precise deviations and improve runnable reproduction alignment where feasible. Do not launch paid/cloud runs without existing authorization; local bounded checks and runs are authorized.

## Review rubric

1. Problem, motivation, prerequisite recap, precise learning outcome and relation to the relational mission.
2. Define terms/symbols; explain why each operation exists; derive equations; coherent numerical traces and edge cases; no unsupported causal/general superiority claims.
3. End-to-end model diagram where a model is introduced: preprocessing, shapes, connections, repeated blocks, head, train/inference differences, exact pictured variant. Otherwise show the evaluation protocol. Confirm prose/diagram/code agreement and actual readability.
4. Paper versus implemented scope: architecture, objectives, optimization, preprocessing, datasets/splits, selection, repetitions, metric, compute and released version. Correct mechanisms where wrong; candidly state remaining omissions.
5. Standalone lab explanations beside coherent visible code chunks, paper-to-code mapping, substantive live TODOs with goal/why/hint, diagnostic CHECKs, no solution leaks, runnable EXIT artifact and interpretation tasks. Verify student functions are used by the experiment.
6. Evidence and reproduction: reconcile published statements with primary sources and saved measured artifacts; appropriate uncertainty/unit of analysis; available executable reproduction track; exact status and unresolved gaps. Never fabricate runs or preserve stale outputs after code changes.
7. Bigger picture: compare mechanisms with the closest earlier/later architectures and explain assumptions/tradeoffs; use concrete examples, not a list of model names.
8. Retrieval, prediction before results, teach-back, meaningful failure diagnosis, clear next step.
9. Delivery: canonical sources/builders survive regeneration; student/solution/prepared HTML/reference agree; links, image payloads, browser desktop/mobile and relevant behavioral/source checks. Distinguish live Colab/deployment untested.

## Deliverable for the assigned lesson

Repair the full package in place. Keep changes scoped to the assigned lesson; shared helper changes only if needed, and report them. Use .venv/bin/python; inspect CLI before running broad generators. `labs/_revise_lesson_depth.py N` and `labs/_lesson_depth.py N` support targeted enrichment. Never overwrite another lesson or erase already repaired work. Preserve saved outputs only if code is unchanged; otherwise execute or honestly clear/relabel.

Current browser environment: `PYTHONPATH=/tmp/relational-access-browser LD_LIBRARY_PATH=/tmp/relational-access-libs/unpack/usr/lib/aarch64-linux-gnu .venv/bin/python ...` loads installed Playwright and Chromium dependencies. The default venv has no Playwright. `labs/_check_quality_audit.py N` is a read-only consistency check for repaired packages; unlike the earlier depth checker, it allows intentional code repairs. The older depth revision/check scripts assume unchanged executable code and are unsuitable as final acceptance gates after code changes.

Write `reviews/lesson-quality-audit-047-070/NNN.md`: verdict before/after with concrete findings, claim/source/code/evidence mapping, repairs, checks actually run with outcomes, remaining fidelity gaps, and whether paper results were reproduced. Include source URLs and precise local paths. Do not claim every claim checked without a substantive inventory. Report completion to parent with changed paths, checks, blockers and residual limitations. Do not spawn additional agents.
