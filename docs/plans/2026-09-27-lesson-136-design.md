# Lesson 136 Implementation Plan

Approved combined scope: leaderboard evaluation replay plus five fresh full-data historical RDL fits, USD10 aggregate including all validation and retries.

**Goal:** Teach the learner to reproduce a reported score, validate complete board coverage and defend a configuration/protocol difference report.

**Architecture:** Freeze RelBench commit 584a03d518b2b655580ea8e1cfbbb26bec0a2841, entry393 (Kapso), entry380 (GNN), entry397 (RT-PluRel fine-tuned), official prediction archives, official evaluator and task data identities. Independently align query keys and rescore all nine regression tasks. Keep this evaluation replay separate from fresh historical v1 Table7 rel-f1/driver-position RDL training, and both separate from top-entry search/training reproduction.

**Tech Stack:** Python/numpy/pandas, pinned RelBench/PyG/Frame GPU runtime, Modal T4, HTML/JS course widgets, nbformat/nbclient, matplotlib and Playwright.

## Tasks and checks

1. Archive source/data provenance in labs/sources/l136 and labs/_sources_l136.json; inspect task label schema and normalized-MAE denominator. Pin attached predictions and full current evaluator. Missing historical training provenance remains NOT_ESTABLISHED.
2. Add failing behavior checks labs/_check_l136.py for keyed alignment, normalized metric and exact board coverage. Implement visible functions in labs/relkit/leaderboard_l136.py; reject missing/duplicate/foreign query keys, nonfinite predictions and incomplete coverage. Cross-check independent metric with pinned upstream evaluator.
3. Write labs/_replay_l136.py: cache official label tables only, rescore all nine task files for three entries, reconstruct aggregate and record hashes/row counts/errors. No score substitutions. Treat incomplete replay as INCOMPLETE.
4. Reuse unchanged L135 data/model/trainer through modal/l136_repro.py and labs/_run_l136.py. Freeze source hashes before pilot. One epoch seed999 pilot, then seeds0-4 ten full epochs, validation checkpoint, test once. Collect checkpoints/results/predictions and independently audit all outputs. Current resource rate .00022572 USD/s; timeout900; automatic retries0. Reserve at most USD7 workers plus USD3 overhead. Expected USD1-3. Additional notebook checks share same budget; no new independent cap.
5. Build lessons/content/0136-leaderboard-literacy.md and HTML, reference, labs/0136-leaderboard-literacy.ipynb and executed solution, portable figures. Live exercises: query alignment, normalized MAE, complete-board aggregation. Include source/config diff, prediction-before-reveal, retrieval and written defense. Inline model/trainer and optional full five-seed gate remain visible.
6. Verify scientific contracts, full-data evidence, notebook execution in isolated directory, meaningful mutation rejection, deterministic rebuild, browser desktop/mobile/keyboard/reset/no-JS/print, copied Pages assets. Update manifest, curriculum, plan, resources, notes and thesis ledger without claiming learner completion.

## Boundaries

Fresh RDL reproduction is not the current GNN entry's exact training run. Kapso archive replay is not its search/training reproduction. Whole-paper/all-board training NOT_RUN. Snapshot ranking is conditional on captured entries and source schema. Submission validation does not establish fair compute or legal training features. No submission, push or deployment requested.
