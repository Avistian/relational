# REL21 grouped rows — builder proof

Authority: direct Piotr morningdecision3, as recorded in the new task's grill/spec/issue. Historical strict-row contracts were not read for implementation, reused, or rewritten.

Baseline: `503704f30fa28e9c8896302d1423f0fcca9b539c` (clean).
Tests-only RED: `06e82129` (full SHA in source-inventory.json).
Implementation GREEN: `beda5136f63ea13c78634c03756b556ab4360413`.
Supported declaration: `lead-guard declare-implementation --task rel-21-grouped --slice rel-21-grouped --worktree /tmp/lead-wt/night-rel21-grouped`, before the RED commit. The immutable declaration remains in the task's `tdd-declarations/rel-21-grouped.json`.

Focused command (existing interpreter, stdlib fixtures only):

```
/home/avist/Projects/relational/.venv/bin/python -B -m unittest discover -s tests/curriculum -p 'test_*.py' -v
```

RED: exit 1, 8 assertion failures, no errors/skips, missing checker. GREEN: exit 0, same 8 unchanged tests pass. Raw stdout/stderr and exit files are retained. The user explicitly required immutable new tests before implementation; the tests were frozen together under that instruction. No canonical replay was run.

Checker command:

```
/home/avist/Projects/relational/.venv/bin/python -B tests/curriculum/check_curriculum.py
```

Live row audit: exit 0, PASS; 228 manifest pages, 228 covered, zero zero-row pages, zero planned-row disagreements. The three manifest-covered grouped ranges are 111–114, 151–154, 192–194 (11 pages). B19a/B19b have their own explicit rows. Planned row 179 has no published manifest page and is consistent. Complete page-to-line inventory is in live.stdout.json. source-inventory.json confirms all 228 HTML pages exist and no HTML pages are unlisted. Fixtureproof JSONL retains input and raw CLI output for intentional grouping, zero coverage, planned disagreement, and a consistent planned unpublished page.

Intent comes only from explicit first-cell IDs, comma/slash lists, or closed numeric ranges; parent IDs, headings, prose, and incidental links do not cover suffix pages. Every manifest page needs coverage, including unpublished entries. Planned rows contradict published entries; a planned future row without a manifest entry is allowed. No source exclusion list or lesson edits were introduced. This checks curriculum row coverage, not scientific reproduction or learner mastery.

CURRICULUM.md and lessons/manifest.json are byte-identical to baseline; all implementation changes are within tests/curriculum. Frozen tests are byte-identical to RED. source-inventory.json binds hashes.

Configured verify-project suite: NOT_RUN. Its configured discovery includes model fitting; the user's restriction prevents that invocation. configured-suite.NOT_RUN.json preserves its command and reason. The profile, command, permissions and gates were not changed. Focused PASS does not replace configured verification.

Builder source/fixtures: PASS. Live row data: PASS, no observed gaps. Configured suite/canonical replay: NOT_RUN. Acceptance: pending separate parent-launched read-only verifier. This is a builder handoff, not DONE, review, acceptance, or experiment success. No agents launched, pushes, merges, installs or spending.
