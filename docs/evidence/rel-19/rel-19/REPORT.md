# REL-19 cached audit package: awaiting separate package verifier

Audit source: `dbd7df5b905f2f79d4c56532c65bedd987899cc1`.
Frozen tests-only commit: `e78e3a2a4872e44d92470feb9f3e325aa087271a`.
Pre-audit baseline: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
Delivery is the commit containing this report; its exact SHA is recorded after
commit in the task's `evidence/current-finish-20261025/REPORT.md` and final READY.json.
Actor: builder:rel-19 (Codex /root), dispatch pane w1:p56.

Production and frozen tests are unchanged. This package commits the authentic
historical builder READY receipt and all separate verifier files, including raw
stdout/stderr, original failed checks, reports and source identities. Files under
`evidence/verifier/` remain the old-source review, not a review of this delivery.
The old `report.md` and old READY verdicts remain historical and unchanged.
`evidence/current-finish-20261025/origins.json` binds copied bytes to origin paths
and SHA256; already committed builder proof also has original Git blob identities.
`package-sha256.json` covers all packaged files except itself.

Herder's delegated decision (`evidence/current-finish-20261025/authority-decision.json`)
waives genuine RED for this unchanged-production audit and approves cached cutoff
tests plus existing non-training sensitivity checks in place of configured discovery.
This is manual-human authority, not a new personal Piotr answer, engine waiver,
canonical replay, or global guard change. No genuine RED/fix/GREEN exists; sensitivity
failures never become RED. The authority file is retained verbatim, including its
other task decisions, which are not execution authority for this builder.

Current builder checks ran against the audit source plus uncommitted evidence
(`dirty:true`), using the existing absolute interpreter and cached data only:

| Check | Actual result |
| --- | --- |
| Frozen test and five cache SHA256 values | All six match |
| Cached cutoff audit and future fixture | Exit 0; 2 tests, zero skips |
| Train / val / test labels | 11,411 / 566 / 702; total 12,679 |
| Labels with history | 10,636 / 546 / 685; total 11,867 |
| Missing-cache probe | Exit 0; one explicit NOT_RUN skip; not audit completion |
| Existing count-corruption sensitivity | Exit 0; expected underlying assertion failure; not TDD RED |
| check-skills | Exit 1: missing historical grill invocation |
| Configured full suite / lead-verify relational | NOT_RUN; discovery calls model.fit |
| Canonical replay | NOT_RUN |
| Final delivery package review | NOT_RUN; lead must launch separate verifier /blast-radius |
| PR / hosted exact-head CI | NOT_RUN; no push or PR created |

Exact argv, exit codes, times and dirty audit source are in current `checks.json`;
raw output accompanies it. `/verify-project` was read and applied with the explicitly
authorized scoped substitute; no full-suite PASS or successful engine run is claimed.
Own current `/tdd` and `/verify-project` invocations were recorded in the task ledger;
the snapshot retains existing history without inventing missing rows. No historical
review invocation is backfilled. The supported skill check still fails at missing
grill; human authority does not make that engine gate PASS. No acceptance call made.

The prior separate verifier's SOURCE_RESULT PASS / PACKAGE_RESULT BLOCKED /
RISK_FLAG YES remain bound to `dbd7df5b...`. This package addresses external proof
packaging, but it does not promote that verdict to delivery approval or acceptance.
Full discovery is prohibited because Determinism tests call fit_predict/model.fit;
the configured profile was not altered. Raw unittest whitespace is preserved, so
`git diff --check` failures on proof logs remain genuine packaging caveats.

The oracle independently filters each driver's results to date < label.date and
checks alignment, all six features and cutoff metadata. Future/equal-time fixtures
are synthetic guard evidence. DOB is undated: age/join correctness does not prove
historical availability of driver metadata. Coverage is one cached rel-f1 snapshot;
missing-cache unittest exit 0 must always be interpreted with skip/NOT_RUN status.
Frozen test, cache and production hashes are retained; no new general safety claim.

Builder package preparation is complete only when committed. Task acceptance and
verified DONE remain pending. No push, merge, deployment, training, downloads,
installs, spend, new remotes, profile/permission answers, global guard changes,
model/protocol campaigns, or agents/workers were launched. Dirty canonical source
was inspected read-only and preserved. Separate package review is required before
human acceptance/publication; actual exact-head hosted CI is required before merge.
