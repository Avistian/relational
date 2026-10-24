# REL-15 fresh builder evidence

LEAD_RESULT: SOURCE_READY

Base: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
Tests-only RED: `7bf494d5f12d45cc42970e33aaacf00f973f6df9`.
Minimal GREEN: `15c60c7c4b8cff09a04ded789bd0d2558dc20981`.
Actor: builder in `/tmp/lead-wt/rb2-rel15`; no delegated agents.

All four browser scripts accept `--out` as the exact review/report root.
It wins over `--report-dir` in either flag order, including live URL mode.
Defaults and previously supported `--report-dir` suffixes are preserved.
Fallback screenshots retain the `after/` layout under the selected review root;
the fallback creates this directory before writing. Course visual exports remain
at `assets/course-visuals`. Review and scratch ignore rules retain tracked artifacts.
No course content or existing review/export artifacts changed.

Frozen stdlib fixtures execute actual argparse/output assignments extracted via AST;
they never import Playwright or start a server/browser. They check all four scripts,
precedence, default and legacy paths, live paths, review literal placement, the
export destination, ignore behavior and retention of tracked PNGs. These are local
CLI-prefix and static checks; browser capture and export execution are NOT_RUN.

RED: 7 tests, 18 assertion failures, no errors/skips, exit 1.
GREEN: 7 tests pass, exit 0. Frozen test SHA256:
`36d98422b503ddb39a4575844e395462bad96a1c71f1cddd53c6dd579c910804`.
Configured `/verify-project`: PASS, exit_code 0, 74 tests at GREEN head,
source.dirty false. Existing notebook missing-ID warnings and negative-fixture
messages remain in raw output; the overall unittest result is OK.

Raw evidence lives at:
`/home/avist/agent-stack/lead/.lead/tasks/w3-stdlib-finish/evidence/postrollback-rel15/`.
It includes precommit and committed RED/GREEN logs and metadata, Doctor output,
scope/freeze checks, configured verification output and a final handoff index.
The first Doctor invocation had a command quoting error; its raw diagnostic is
preserved as `doctor-command-error.*`. The corrected direct-argv Doctor passed
before test authoring. Historical evidence under `evidence/w3-rel-15` is untouched.

Separate REVIEW: pending, lead-owned. Canonical replay: NOT_RUN, parked.
Hosted CI/publication/push/merge: NOT_RUN. No installs, spend, browser/model runs,
profile changes, gate changes or meta repair.
