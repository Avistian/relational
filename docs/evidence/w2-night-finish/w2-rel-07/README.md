# REL-07 fresh builder handoff

Status: SOURCE_READY_WITH_INVENTORY_GAPS; independent review and acceptance remain with the parent.
Builder: night-w2-rel07, branch lead/night-w2-rel07, worktree /tmp/lead-wt/night-w2-rel07.

Verified implementation source: `8b961503896e9fb9ed4cc78235c60d2c11eeee85`.
Tests-only RED: `de7887c54a76dccba2e69c905036e2104fee3f5e` (parent 503704f30fa28e9c8896302d1423f0fcca9b539c).
This evidence-only commit follows that implementation; it does not change executable source or tests.

## Frozen RED / GREEN

Fresh CLI contract executed locally with existing ignored .venv, without installation.
RED exit 1: one assertion failure, zero errors, because the published checker CLI was absent.
GREEN exit 0: the same integration test passes with identical test bytes
(SHA256 c36e71299d71cb0bf1129f18149239f6055ee695056ad73c3122e165b96f9386). No fabricated fail, skip or dependency error.
The test exercises good/broken file and fragment targets, percent-encoded literal and decoded
anchors, legacy names, case sensitivity, queries, directory indexes, root-relative paths,
non-HTML fragments, external/vendor/archive exclusions, hidden published directories,
symlink root escape, deterministic output, unchanged fixture bytes and CLI exit 0/1/2.
This is one CLI integration contract; no claim of individual RED proof for each branch.

After fresh RED, reused reviewed logic from historical source
02e780643373902d83f737bfba1ea14ed0cbe9d3. Removed its extra hidden-directory exclusion
so only vendor/archive categories are excluded. The fixture explicitly includes a hidden
published directory. Previous source, contracts, attempts and ledgers remain untouched.

## Actual inventory

CLI exit 1, LEAD_RESULT: FAILED.
672 published HTML sources, 20,179 checked local targets, 6,006 excluded URL occurrences.
111 broken occurrences: 110 missing-file, 1 missing-fragment.
Full broken records are in broken-links.json; raw full inventory includes every page,
checked target and exclusion. The missing fragment is
reference/solution-map-atlas.html -> ../lessons/0058-surveys-meta-benchmarks.html#solution-evidence-funnel.
These are unresolved content findings; no source content was repaired.
All 21,391 tracked file hashes matched before and after the real CLI.

## Verification

Requested verify-project skill executed the unchanged configured relational command:
`/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/night-w2-rel07`.
Durable verdict PASS, exit_code 0, source.head 8b961503896e9fb9ed4cc78235c60d2c11eeee85,
dirty false; 68 tests in 7.437s, OK.
Original evidence: /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261024T202706Z.
No model or protocol execution was required by that configured unittest suite.

## Raw proof and limits

Raw stdout/stderr, per-command provenance, complete inventory, tracked hashes and frozen-test
proof: /home/avist/agent-stack/lead/.lead/tasks/w2-night-finish/evidence/w2-rel-07.
raw-sha256.json binds raw artifacts. Parent task READY.json is written last after the evidence commit.

Filesystem HTML href/src/poster inventory only: root *.html, lessons/, labs/html/, reference/.
Vendor/archive trees and external schemes/hosts are excluded; no network checks.
CSS/JS-generated links, srcset, deployed Pages base URLs, browser rendering and non-HTML
fragments are outside this checker contract. Source symlinks escaping the selected root
are not read; referenced escaping targets are reported outside-root. Existing HTML parse/IO
errors fail the CLI rather than inventing exclusions. No content completeness, browser,
scientific-reproduction, or deployed-site health claim.

Canonical execution: NOT_RUN (explicitly prohibited). Separate review: PENDING_PARENT.
No canonical receipt, old-attempt retry, push, merge, install, spend, gate/profile change,
subagent launch, or builder-done/acceptance transition performed.
