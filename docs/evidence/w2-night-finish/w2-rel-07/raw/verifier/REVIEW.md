# REL-07 independent read-only review

Reviewer: `verifier:w2-rel-07`; worktree `/tmp/lead-wt/night-w2-vrel07`.
Reviewed delivery SHA: `758820f0fc221065a5c5be0ff4337b609d4c0d4c`, clean before/after.
Implementation SHA: `8b961503896e9fb9ed4cc78235c60d2c11eeee85`.
RISK_FLAG: DATA_GAPS_AND_CONFIGURED_NOT_RUN.

Source review PASS within the published filesystem HTML contract. Focused execution PASS.
Data verdict FAIL_EXISTING_LINKS. Acceptance NOT_ACCEPTED; no DONE transition.
Delivery claims QUALIFIED: the retained full-suite PASS is real, but does not establish a
verification run without model fitting. No implementation or content changes were made.

## Scope and blast radius

The new standard-library CLI scans root HTML, lessons/, labs/html/, reference/ and
their href/src/poster attributes. It checks local files and HTML fragments, handles
directory index.html, and excludes external URLs and vendor/archive categories.
`scripts/check_links.py:54-57` selects sources, `:80-88` excludes/resolves URLs,
`:90-106` bounds reads and checks files/anchors, `:122-130` maps audit outcomes to
exit 0/1/2. Non-HTML fragments, CSS/JS/srcset, browser execution and deployed routing
are outside this contract. No new dependency or runtime integration is introduced.

The safety fact is that running the CLI inventories the site without changing its
bytes or reading referenced symlink targets outside the root. This reached skill
level 4 (real code execution): all 21,395 tracked file hashes match before/after,
and disposable probes exercise direct traversal, symlink escape, and a directory
whose index is an escaping symlink. Expected outside-root outcomes passed, and the
external sentinel and fixture bytes were unchanged. This proves the exercised
stable-tree behavior, not immunity to concurrent filesystem replacement races.

The consumer census finds the new CLI used by `tests/links/test_links.py:11,65`.
`tests/links/__init__.py` includes it in recursive unittest discovery, also used by
`.github/workflows/pr-checks.yml` (Run unit tests). No other executable caller was
found in scripts/, tests/, or .github/. The whole delivery delta after code SHA is
four docs/evidence files; executable and test blobs are unchanged. Hosted CI was
not invoked and is not claimed.

## Independent execution

`reproduce.py` is the retained external verifier driver. No project files are
generated; frozen minimal exports and fixture sites use external TemporaryDirectory.

- Frozen RED `de7887c54a76dccba2e69c905036e2104fee3f5e`: exit 1, one assertion
  failure because the checker CLI is absent; no import/dependency error.
- Frozen GREEN `8b961503896e9fb9ed4cc78235c60d2c11eeee85`: exit 0, one test passes.
  Test bytes at RED/GREEN/delivery match SHA256
  `c36e71299d71cb0bf1129f18149239f6055ee695056ad73c3122e165b96f9386`.
  This independently verifies the existing frozen contract; no new TDD work,
  canonical execution, declaration or ledger rewrite was performed.
- Current delivery focused test: exit 0, one test passes. See focused.stderr.
- Two current CLI inventories: exit 1, identical stdout/stderr. Independently
  implemented parser/resolver imports no checker code and matches every page,
  checked target, broken record and exclusion, not just summary counts.
- 23 additional URL probe cases PASS: Unicode/literal/decoded/double-encoded
  anchors, case sensitivity, legacy name, text directive, self/top, query,
  directory/root paths, non-HTML fragments, encoded exclusions, external schemes,
  malformed URL and root escapes. src/poster and hidden published sources also
  checked. These are verification probes, not repository tests or new RED claims.
- Every listed fresh builder raw artifact hash and its raw-sha256 manifest match.
  Old deleted worktree proof was not used to establish any verdict.

## Confirmed risks and qualifications

1. Existing published navigation is broken (confirmed, user-visible data impact):
   672 source pages, 20,179 checked targets, 6,006 exclusions; 111 broken occurrences,
   comprising 110 missing-file and one missing-fragment. Full current findings are
   in broken-links.json. The fragment failure is
   `reference/solution-map-atlas.html:1` to
   `lessons/0058-surveys-meta-benchmarks.html#solution-evidence-funnel`.
   These are content followups; no permission to repair them is inferred.
2. Verification-scope wording needs qualification (confirmed reporting risk): the
   retained builder README says no model execution was required by the full suite.
   Its configured discovery includes `tests/relmini/test_relmini.py:69-78`, calling
   `labs/relmini/baseline.py:72-79`, which creates LogisticRegression and calls fit
   on synthetic fixtures. Synthetic fitting is not a real-data scientific result,
   but it is model execution. This verifier avoids that ambiguous execution boundary.
   Original reports and PASS evidence remain untouched.

Cleared by execution: source selection, external/vendor/archive exclusions,
percent decoding, relative/root paths, fragment/name matching, non-HTML fragment
handling, stable-tree symlink containment, deterministic output, exit behavior and
read-only bytes. No source defect was found in the reviewed contract.

## Verification and acceptance boundaries

Installed verify-project skill and unchanged configured argv were inspected.
Fresh configured `/verify-project` is NOT_RUN: recursive discovery reaches model.fit;
no invocation, override, skip injection or config change was made to bypass this.
Focused PASS must not be read as full configured verification.

The retained durable builder status at
`/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261024T202706Z`
is PASS, exit_code 0, source.head 8b961503896e9fb9ed4cc78235c60d2c11eeee85,
dirty false; stderr records 68 tests, OK. This was opened and hash-checked as fresh
builder evidence, not relabeled as an independent verifier run or a delivery-SHA
full-suite run. The evidence-only descendant has only focused independent execution.

Before any later acceptance, preserve these boundaries and address or explicitly
dispose of the broken-link data through the owning lead's authorized workflow.
The cheapest source regression check is the already-run focused links unittest
and retained external probe driver. Canonical execution NOT_RUN by explicit user
prohibition. No model/protocol, push, merge, install, spend, workers, permissions,
profile, gates, shared ledgers or acceptance transitions were changed here.
