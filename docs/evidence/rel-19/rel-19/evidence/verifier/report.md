# REL-19 separate verifier: source PASS; acceptance BLOCKED

Exact reviewed SHA: `dbd7df5b905f2f79d4c56532c65bedd987899cc1`.
Baseline: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
Tests-only commit: `e78e3a2a4872e44d92470feb9f3e325aa087271a`.
Actor: Codex `/root`, separate verifier, worktree `/tmp/lead-wt/night-v-rel-19`;
dispatch status records pane `w1:p3V`. Source was clean before and after checks.

`LEAD_RESULT: BLOCKED` (acceptance); `SOURCE_RESULT: PASS`;
`PACKAGE_RESULT: BLOCKED`; `RISK_FLAG: YES` (process/package/acceptance gaps).
This is a completed bounded source review, not task DONE or acceptance.

## Source and independently executed evidence

The change adds a cached-data audit and retained evidence; production code is
unchanged. The audit compares every label with its driver's strictly earlier
source results, independently of production's cumulative sums and asof join.
It checks row alignment, cutoff, count, history flag, DNF rate, grid, points,
and age. No source defect was found within this cached snapshot contract.

Executed once with the existing ignored `.venv`, `python -B`:

| Check | Observed result | Evidence |
| --- | --- | --- |
| `-m unittest tests.relmini.test_real_leakage -v` | Exit 0, 2 tests, no skips/errors/failures | `cutoff.stdout.txt`, `cutoff.stderr.txt`, `focused-status.json` |
| Cached train | 11,411 labels; 10,636 with history | `cutoff.stdout.txt` |
| Cached val | 566 labels; 546 with history | same |
| Cached test | 702 labels; 685 with history | same |
| Builder missing-cache probe | Exit 0; expected NOT_RUN skip, not audit completion | `missing.*.txt` |
| Builder count corruption probe | Exit 0; expected underlying assertion failure at audit line 46 | `oracle-sensitivity.*.txt` |
| Frozen test and five cache hashes | Exit 0; all six match | `hashes.stdout.txt` |
| Builder receipt artifact hashes | All 10 match reviewed checkout | `builder-artifact-hashes.json` |

Total: **12,679 labels**, 11,867 with history. The future fixture rejects both
equal-time and future-time metadata (`tests/relmini/test_real_leakage.py:63`).

## Blast radius

The safety fact is that strict temporal selection and the independent aggregate
oracle agree, rather than a legal cutoff timestamp alone proving feature safety.
This reaches skill evidence level 4: real code executed.

`blast-probe.py` independently calls real `baseline.build_features` with one
historical, one same-day, and one future row. Only the historical row contributes:
count 1, mean grid 2, mean points 3. The public guard rejects equal/future cutoffs.
An in-memory corruption of mean points keeps legal cutoff metadata but causes
the frozen cached audit to fail at line 56. The expected raw failure is preserved
in `blast.stderr.txt`; `blast.stdout.txt` confirms all probe assertions completed.
These are sensitivity probes, **not TDD RED**, source changes, or canonical replay.
An independent missing-cache probe also traps `load_real`: it is never called,
and the test skips with explicit NOT_RUN naming all five required files.

Consumer census: production `fit_predict` calls `build_features` and the guard
(`labs/relmini/baseline.py:72`); `run` calls fitting (`:82`); verification calls
the baseline run (`labs/relmini/verify.py:47`). Existing test consumers include
LeakageGuard and RealData in `tests/relmini/test_relmini.py`. None changed.
No model fitting consumer was executed. The added test uses the existing
baseline import path convention. The independent oracle covers all six declared
features on the retained snapshot, without calling fitting or network APIs.

Cleared: same-day/future joins, aggregate corruption with legal metadata,
missing-cache accidental pass interpretation in the handoff, frozen file/cache
identity, task/feature alignment, changes escaping the two owned paths.

Limits: DOB is undated metadata; this verifies its join/age computation, not its
historical availability. This is one cached rel-f1 snapshot, not other datasets
or a general proof of feature provenance. Unittest returns zero for missing-cache
skips; consumers must inspect NOT_RUN and skip count before claiming completion.

## Process, package, and acceptance gaps

1. **TDD BLOCKED.** The first genuine execution passed unchanged production.
   No genuine RED/fix/GREEN pair exists. Neither retained mutation failure nor
   verifier sensitivity failure supplies that pair. No implementation, /tdd,
   manufactured RED, or rework loop was attempted.
2. **verify-project NOT_RUN.** Read the installed skill, actual `bin/lead-verify`,
   and configured relational command. It discovers all `test_*.py`; Determinism
   tests call `fit_predict` (`tests/relmini/test_relmini.py:69`, `:78`), which calls
   `model.fit` (`labs/relmini/baseline.py:78`). The task prohibits training and
   profile changes, so the configured command was not executed or replaced.
   `verify-project-static.json` records the actual configuration and reason.
   Inspected retained PASS records at `20261024T191321Z` and `20261024T191535Z`
   bind to REL-16 heads, the first dirty; neither establishes this SHA's PASS.
3. **Package mismatch.** The specified committed builder READY.json does not
   exist at this SHA. The original builder worktree contains an untracked receipt
   that explicitly describes itself that way. Its exact SHA and all ten artifact
   hashes agree; preserved as `builder-READY-retained.json`. Actual committed
   report is `docs/evidence/rel-19/rel-19/report.md`, not `evidence/builder/report.md`.
   No files were copied into the source checkout or committed to repair this.
4. **Whitespace failure retained.** `git diff --check 503704f3..HEAD` exits 2 for
   trailing whitespace in the raw builder mutation stderr at line 1. Preserved
   verbatim; not a production failure, not repaired or overridden.
5. **Authority provenance retained.** Answered grill names Piotr under Herder's
   chat standing order (~21:00 Warsaw on 2026-10-24). Its note explicitly says
   this is not a lead-guard delegated overnight window. Earlier admission failure
   remains preserved alongside that later note; this review does not certify gate
   admission or rewrite the answer as a direct personal response.

Canonical replay: **NOT_RUN**. Hosted CI: **NOT_RUN**, no exact-head retained
hosted proof supplied. Separate verifier review: **performed here**; it does not
replace builder VERIFY, TDD admission, or acceptance. Before acceptance, the lead
must resolve the existing process/package requirements; no retry or scope expansion
is requested or implied by this handoff.

## Skill-link warning (separate operational issue)

Dispatch recorded `SKILL LINKS INCOMPLETE`: target `bin/lead-link-skills` absent.
No linking success is claimed. Used the installed absolute
`/home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md` and
`/home/avist/agent-stack/lead/.agents/skills/blast-radius/SKILL.md` directly.
The blast-radius instruction references `unslop`; no installed entry was found
at the corresponding lead skill path. Plain-language editing was performed
directly; no unslop tool execution is claimed. No linker copies, installs, or
permission/profile/gate changes were made. No agents were spawned.
