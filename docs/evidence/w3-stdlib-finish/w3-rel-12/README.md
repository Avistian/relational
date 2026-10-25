# REL-12 builder evidence

Base: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
Tests-only RED: `e6a01813ac74f7f9c65737232098b22d4ad4e767`.
Minimal GREEN: `2ab9e215e8f9792e9ccf14c28d343cdb55c94851`.

The progress CLI accepts `--check` alongside the existing root, date and output
options. Both files must match the deterministic UTF-8 output for the selected
date. Formatting differences also count as stale. Missing, unreadable or stale
outputs produce path diagnostics and exit 1; current outputs exit 0. The check
branch performs no writes and checks both outputs before returning.

The frozen CLI acceptance matrix covers 18 independent fixture cases: default
and custom output paths, current outputs, either output stale or absent, both
absent, a changed reference date, and invalid bytes in either output. Snapshots
compare every fixture file's bytes and every entry's modification time and mode
before and after checking. Existing generation behavior is also exercised.

Raw RED/GREEN and progress-suite logs are under `evidence/postrollback-rel12/`.
RED exited 1 with 18 genuine assertion failures because `--check` was absent.
GREEN exited 0 with unchanged test bytes. The progress suite passed 14 tests.
Doctor passed using the existing parent-provisioned runtime; no installation or
environment repair was performed.

The first `/verify-project` run passed 68 tests at GREEN SHA with `dirty:true`
because these evidence files were untracked. Final exact-head verification and
the SOURCE_READY receipt are retained outside the checkout at
`/home/avist/agent-stack/lead/.lead/tasks/w3-stdlib-finish/evidence/postrollback-rel12/`.
The receipt is written last after the evidence commit and verification.

Builder handoff only. Separate read-only review and canonical replay remain
pending with the lead. Parked before canonical replay; no push or merge.
