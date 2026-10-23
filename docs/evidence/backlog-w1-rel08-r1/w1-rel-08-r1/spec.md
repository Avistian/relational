# REL-08 R1 spec

## Problem
Separate reviewer reproduced duplicate cell IDs returning errors: [] because nbformat validates after repairing IDs in memory. Original410 notebooks have no duplicate warning;464missing-ID warnings are explicitly retained.

## Contract
1. Base fresh branch on ae27156b5b27a583de80c95be1c3bcfa8eec3538; preserve all original REL-08 history and source. One builder then a distinct verifier, shared cap4.
2. Add permanent duplicate-ID regression and missing-ID warning-policy test in tests/notebooks/test_duplicate_ids.py. Meaningful assertion RED against existing real checker before any implementation edit; no import errors or artificial breakage.
3. Pin tests/notebooks/check_notebooks.py as exact IMPLEMENTATION_FILES in this follow-on intake before RED. Existing original seven tests and new canonical test file stay byte-identical through GREEN; only declared checker may change. No old pin/receipt edits.
4. Reject duplicate IDs, retain missing-ID warnings, keep all410 notebook bytes unchanged. Keep exact intentional pairing exceptions and original CI dependency setup. No notebook/workflow/content edits in this follow-on.
5. Separate verifier covers full REL-08 original diff and R1, actual raw RED/GREEN, mutation repro, fresh env/project verification and final human proof/code equivalence. Draft publication through managed lead-publish after acceptance; never merge.
6. Commit readable current human proof under docs/evidence/backlog-w1-rel08-r1/w1-rel-08-r1; retain original blocked review/probe and original canonical receipt as historical evidence. Main owns IPC/publication bridge, actor provenance explicit.

FEATURE_ESTIMATE: w1-rel-08-r1 | 30

## Test seams
Existing check_notebooks.py actual audit path and nbformat library, disposable complete paired fixture, new regression module added in tests-only RED.

## Out of scope
Teaching notebook repairs, broad schema expansion, missing-ID rejection, exception allowlist expansion, local installs, new remotes/providers/spend, Lead push, merge, or edits to any original W1 intake/pin/receipt.

## Done when
Duplicate IDs fail the real checker; existing missing-ID warnings remain nonblocking; all410 notebook hashes stay unchanged; separate verifier PASS and actual final-head draft PR CI are required.
