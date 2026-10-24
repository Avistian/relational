# REL-16 R1 independent verifier

Actor: separate read-only verifier in /tmp/lead-wt/night-vrel16-r1.
Exact reviewed SHA: 86fe6e7284421f128686da00240aeddb5595cbab; dirty:false.
Base reviewed SHA: fa4521280ab256beb622bfa3e0c8ec66e2ef68b7.
RISK_FLAG: YES.
Source/code verdict: PASS for the bounded R1 repair; REL16-V1 and REL16-V2 resolved.
Configured verification: NOT_RUN. Acceptance: NOT_DONE.
No source edits, subagents, workers, training, downloads, installs, spend, push,
merge, publication, permission/profile/gate changes or new canonical replay.
Only new verifier evidence was written in this external directory.

## Independent execution

- Focused command: .venv/bin/python -B -m unittest discover -s tests/pages -v.
  PASS, exit 0, ten methods at the exact clean reviewed SHA. The unchanged seven
  original methods and three new regression methods run the public CLI.
- Quoted wildcard suite covers single/double quotes, partial quoting, question
  mark/bracket patterns, and adjacent quoted strings. Wildcard-removal suite
  covers *, ?, brackets and *.txt. Root-removal suite covers public, public/,
  public// and ./public//. Each synthetic plan executes successfully in Bash,
  produces the demonstrated changed file set, and is rejected by the CLI with
  exit 2, empty stdout and an unsupported diagnostic in all three positions:
  generation, compared workflow, and generating workflow with comparison.
- Additional independent blast-probes.py repeats the original quoted-glob and
  wildcard-removal findings plus all four normalized root forms, with 18 CLI
  rejections. Literal quoted filename containing a space and literal subtree
  removal remain supported and match actual Bash. See blast-probes.json.
- Independent current-workflow inventory: Bash/GNU cp executed every copy,
  conditional, numbered loop and removal against this checkout into disposable
  scratch. Exactly two existing HTML byte-version mutations were omitted.
  Shell exit 0; CLI --compare exit 0; 18,497 actual = 18,497 generated = retained
  original inventory, zero membership differences. All three diagnostic lists
  are independently empty. See parity.json and actual-files.json.
- Initial inventory harness attempt failed before Bash launch with OSError
  errno 7 (script too large as one bash -c argument). Its script, diagnostic
  and traceback are preserved in parity-attempt-1.*. Passing the same script
  through stdin completed once. This is harness provenance, not a source/test
  failure, canonical replay or permission change.
- Source parses with ast.parse; git diff --check passes; pages.yml is unchanged.
  R1 changes only scripts/pages_publish_list.py, the new regression test file,
  and the owned README. Builder source/artifact hashes all match.

## Blast radius

The repair rejects quoted wildcard syntax before shlex loses quoting at
scripts/pages_publish_list.py:993-998, rejects wildcard removals at :1006-1009,
and rejects normalized public-root removals at :1010-1012. It retains literal
subtree removal. No supported workflow operation or frozen default plan changed.

The safety fact is proven at skill level 4 (executed real code): these rejected
forms raise before the single JSON print at :1126, including when encountered
on the comparison side, so they cannot produce the prior equal:true result.
The public CLI and actual Bash probes establish this; focused fixtures broaden
coverage. Current-workflow Bash inventory independently proves no membership
regression. This is not a deployed browser/app validation.

Consumer census across .github, scripts, tests and docs finds the executable
reference only in the two Pages test modules; the owned README documents the
manual CLI. pages.yml does not invoke this generator. Exit/JSON behavior is the
external contract and the changed rejection is intentional. There is no new
runtime dependency: glob, re, shlex and PurePosixPath were already used; the
existing ignored .venv and installed yaml were reused without installation.

Cleared: original quote/removal false equalities, valid unquoted globs, literal
quoted paths, literal subtree removal, current frozen plan, direct manifest
diagnostics, unchanged workflow, deterministic sorted file membership, and
retained source/evidence hashes. No remaining source finding in this R1 review.

Remaining risk: this is a bounded shell interpreter. Quoted literal wildcard
filenames are deliberately unsupported; arbitrary Bash is outside its contract.
Membership parity does not prove published bytes, empty-directory behavior,
symlink target validity, transitive link integrity or deployment correctness.
The cheapest future regression check is the focused Pages command above plus
blast-probes.py; no source repair or extra replay is requested by this review.

## TDD and frozen history

Installed TDD guidance was read for audit only; this verifier performed no TDD
implementation. static-history.json and retained-audit.json inspect Git objects,
raw logs, original frozen hashes and receipt bindings without replaying history.

Original cycle: tests-only 9e09a39b909a445b420ca4ff0fbcb5fca8c7aff1 -> production
1ba8e647d4ef15eb573b8caa13033cdad5e3b9ff. Seven public CLI tests, six assertion
failures and zero unittest runner errors at RED (missing executable script);
the escape test already passed. Seven GREEN. Tests are byte-identical from RED
through current HEAD. Canonical receipt/report/raw-log hashes match retained
artifacts. That PASS is historical at 1ba8e647, not R1 acceptance.
Methodology limitation: all seven original tests preceded one implementation
commit, rather than the installed skill's one-test-at-a-time cadence. Preserve
that distinction; do not represent historical receipt integrity as full process
conformance or rewrite its evidence.

R1 vertical cycles, each one new behavior method before its minimal source fix:

| Behavior | Committed clean RED | Production GREEN | RED assertion failures / runner errors |
| --- | --- | --- | --- |
| REL16-V1 quoted wildcard | b5fdb10040d19b5230c35ce8dd161e70cd6eb724 | 5da4d654763e1fd5364a20c908a5446b07f185c5 | 21 / 0 |
| REL16-V2 wildcard removal | 27985b26853357f7812896eef5aaab31a521b2e1 | 26fad3df990bce57b2e8692940d4a1453c54824f | 12 / 0 |
| REL16-V2 corrected root removal | c1a98c2e9e6067aed6dc1197f20563e751f6cf57 | dd7580a7e283c4ceff0aca6a9a41a661e47fd0c2 | 12 / 0 |

Git confirms each RED changes tests only, each GREEN changes production only,
and each pair retains byte-identical tests. Counts are subtest assertions, not
method counts. Intermediate GREEN receipts honestly bind the preceding RED HEAD
with dirty:true production changes; they are not clean GREEN-SHA executions.
The builder's final ten-method receipt and this verifier's fresh focused receipt
bind clean final HEAD. All historical raw files remain unchanged.

Invalid fixture preserved: 3a628006af7cf984832ef3f187bf79355d54a257 recorded ten
failures: nine real parser assertions and one invalid shell expectation for
GNU rm public/., which refuses '.'. Separate c1a98c2e replaces that operand with
public// before any root production fix, establishing twelve genuine assertions.
fixture-correction.diff.txt and the original raw stderr preserve this history.
No new canonical replay, receipt admission or lead-guard validation was run.

## Configured verification and handoff limits

Applied installed /home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md
and blast-radius/SKILL.md directly. Actual lead-verify implementation and
config/verify-projects.json were inspected. The relational mapping runs absolute
existing-venv Python -m unittest discover -s tests -p test_*.py (timeout 600).
tests/relmini/test_relmini.py:69-78 calls baseline.fit_predict, which builds
LogisticRegression and invokes model.fit at labs/relmini/baseline.py:77-78.
User's no-training limit therefore leaves /verify-project NOT_RUN, exit_code:null,
source.head:86fe6e7284421f128686da00240aeddb5595cbab, dirty:false.
The focused run is not substituted into project configuration or called full
software verification. Historical 74 PASS belongs to earlier SHA(s), including
the retained exact-head fa452128 receipt, and supplies no full-suite R1 PASS.

Packaging provenance is now accurately documented, but historical committed
READY.json/report.md remain absent. Git trees at base and current SHA contain
README.md as the narrative report; the original READY was intentionally untracked
in its original worktree and is byte-identically preserved at
/home/avist/agent-stack/lead/.lead/tasks/rel-16/evidence/verifier/original-builder-READY.json.
All its recorded artifact hashes still match. Neither this verifier nor the R1
builder invented a committed READY/report. Current R1 evidence and this handoff
are external task artifacts. Parent must resolve any requirement for a committed
package; this review does not waive or fulfill it.

The checkout-relative bin/lead-link-skills is absent; the installed absolute
/home/avist/agent-stack/lead/bin/lead-link-skills exists. No linker was copied or
executed. This does not block direct reading of installed skills.

Acceptance remains NOT_DONE: configured verification is NOT_RUN, any committed
package requirement remains parent-owned, and no new canonical admission, hosted
CI, push or merge is claimed. Parent owns acceptance. READY.json is written last
and hashes this report and every other new verifier artifact; earlier builder,
verifier and R1 receipts remain untouched.

LEAD_RESULT=REVIEW_COMPLETE_SOURCE_PASS
OK exact SHA clean
OK REL16-V1 and REL16-V2 resolved
OK focused Pages tests 10
OK current publication membership 18497
OK frozen tests and historical evidence integrity
NOT_RUN configured verify-project: model fitting prohibited
NOT_RUN new canonical replay, hosted CI, push, merge
NOT_DONE acceptance; external handoff does not fulfill committed-package requirements
