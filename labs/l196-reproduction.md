# L196 — Complete original-preprocessor diagnostic and community question

Approved experiment: L196 complete original-preprocessor four-case diagnostic. USD0 cloud/API; 1800 aggregate local execution seconds including installation, retries, verification and delivery. A full-model experiment remains outside this diagnostic and stopped by the existing source gate.

## Frozen protocol

RDBLearn v0.1.2 commit b5b03ebf8091547285a6e06cba53d2d1a40cb171, unchanged full TabularPreprocessor, including AutoGluon. The source packet copies 62 files byte-for-byte from the authenticated L193 ledger. The preprocessing file was additionally downloaded from the exact upstream commit and matched SHA256 87fa6f9fe7d441fbcd5fd4e6119fd97b35a13c6d2ad6dd3e4195beb9c3229371.

Fit twelve rows: categories b/c/d repeated four times, numeric column 0–11. Cache support codes; transform known b/c/d with numeric 1/2/3; transform one unseen row with numeric 1; transform known rows again. Repeat independently for a, z, 0, e. Each case also compares query b alone against b batched after the unseen row using separate newly fitted preprocessors. No randomness or selected subset; all four predeclared cases are required.

Independent oracle uses list sorting/indexing, not the original encoder, to check every expected code. Numeric columns must remain 1/2/3 and cached support 0/1/2 repeated four times. Known codes shift for a/0 but not z/e. This is an observed consistency counterexample, not a metric, prevalence estimate, backend comparison or statistical population claim.

## Portable fresh execution

Download and extract [reproducer.zip](evidence/l196/reproducer.zip). Its README gives standalone Python 3.12 commands to create a virtual environment, install the exact dependency freeze, run pip check, and execute diagnostic.py. Every included source file is hash-checked before import.

The [solution notebook](solutions/0196-community-engagement.ipynb) embeds the entire packet, displays the original preprocessing source and diagnostic, and defaults to MODE=fresh. It creates a new temporary environment, installs the pinned CPU dependencies, runs the original full diagnostic in a fresh child process, and checks exact agreement with saved output. Installation requires internet but no paid services or model checkpoints. MODE=replay is explicitly offline saved-evidence practice and cannot establish a fresh run. Live Colab is NOT_CHECKED.

The student notebook exposes three meaningful contracts: complete-case evidence admission, owner routing, and feedback state. Passing them does not certify participation or mastery.

## Repository commands

From the repository root, use .venv/bin/python labs/_budget_l196.py before each execution command. Example:

    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_verify_l196.py
    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_build_l196.py
    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_execute_l196.py
    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_delivery_l196.py
    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_seal_l196.py
    .venv/bin/python labs/_budget_l196.py .venv/bin/python labs/_checkout_l196.py

The preparation script records the author environment using /tmp/l192-repro-env and copies L193 source; it is an authoring helper, not the portable installer. Use the ZIP or notebook to reproduce independently. Running the builder resets solution outputs; execute the solution again before sealing. The budget ledger records failed attempts and cuts off at the aggregate limit.

## Boundaries and deviations

This lesson teaches community engagement; it introduces no new model or new paper-table experiment. The exact original preprocessing diagnostic is complete. It uses a declared contemporary CPU environment (AutoGluon 1.5.0), not an authenticated historical GPU environment. Benchmark prevalence, model-score effect and historical paper behavior remain NOT_ESTABLISHED. Full RDBLearn model reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. No repaired source is silently substituted.

The local question draft asks how support/query category alignment is intended to work and which historical revision/environment to reproduce. Community source snapshots are current routing evidence, separate from the frozen historical implementation. Availability of a public page does not establish posting permissions. The actual question is DRAFT_ONLY; no message has been sent. A response and checked follow-up are required to finish the community milestone. Learner PENDING_WRITTEN_DEFENSE.

No deployment is requested. Browser, notebook and Git-index build results are separately recorded; no live deployment or live Colab claim follows from local checks.
