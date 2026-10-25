# REL15 history-independent CI
FEATURE_KIND: non-ui
FEATURE_ESTIMATE: rel15-ci-r1 | 45
## Problem
PR35 run40192853147 fails test_generated_directories_ignored_and_tracked_pngs_retained at git ls-tree503704f exit128; CI lacks historical object, full local history masks it.
## Contract
1. Reproduce actual test in a temporary single-commit repo without adding remotes. Record trigger/mask/symptom and compare to proven full-history path.
2. Replace runtime old-object requirement with truthful self-contained original PNG baseline derived from actual503704f bytes, preserving PNG retention and ignore checks; no skip or weaker current-only expectation.
3. Fresh regression test exercises actual old unittest path plus deletion counterfactual. Existing test_output_paths.py is declared implementation; freeze selected new regression test.
## Test seams
Public unittest behavior with temporary Git repository and controlled ROOT, genuine RED/GREEN. No browser/model execution.
## Out of scope
Browser scripts, course/PNG content, CI permissions, local/global Git config changes, installs/spend and second repair attempt.
## Done when
The real PNG-retention unittest succeeds in a fresh single-commit repository and still fails when an original tracked PNG is removed, without consulting unavailable history.
