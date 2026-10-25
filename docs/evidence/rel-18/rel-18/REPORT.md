# REL-18 builder handoff

Status: SOURCE_READY; separate verifier required. No DONE/acceptance/PR/push/merge claim.

Actor `builder:rel-18`; actual Herdr pane `w1:p5C`, tab `w1:t5C`.
Builder role only. No subagents, workers, reviewers or verification roles launched.
Started after 2026-10-25 11:50 UTC; handoff prepared at approximately 12:00 UTC,
within the 45-minute timebox. First progress was delivered before 15 minutes.
USD 0 local CPU; no install/download/cache mutation/provider change/new remote.
Piotr time NOT_MEASURED. Task decision comes from Herder's delegated grill, not a new direct Piotr quote.

## Source identities

- Base REL19: `bd1a05ff1ce409195013093840be8ec05213aa2e` (PR32).
- Final implementation/GREEN/scientific execution source: `557fa7ba846588ec182ec2fd4b51fc98ae3a1a57`.
- Canonical tests-only RED: `42f90b70` (full SHA in `tdd-report.json`).
- Clean configured software verification source: `054b684641ce30fbdc0d3e85abad3bf34ddc1b0b`.
- Changes from implementation source to verification source contain only `docs/evidence/rel-18/rel-18`.
- Final delivery may append only this report and verification/environment evidence; READY.json binds that delivery SHA.
- Frozen canonical test file sha256: `2757ef25bfc31853d26f1734bfda07f66e4bb6718da27bd636dfbf85d2cc27ab`.
- Every source/RED/GREEN/output SHA256 is in `artifact-manifest.json`; commit history is reproduced below.

## Behavior and decisions

Preserve the original driver features, cached train/val/test row order and membership,
seed 0, StandardScaler and LogisticRegression(max_iter=1000, random_state=0).
Constructor features are prior constructor-results record count, mean points, history indicator.
Resolve the latest strictly prior driver's constructor; aggregate that constructor's records
strictly before the label date. The historical constructor is a proxy for the label-day team.
Historic shared drives have conflicting teams on the same date: mark the latest team unknown,
without file-order choice or fallback to older teams. Both assignment and feature timestamps are checked.
No history uses fixed zeros. Original missing-history driver priors remain unchanged.

The original baseline without constructor features is the constructor ablation of the augmented model.
A focused synthetic test proves disabling constructor features reproduces original predictions exactly.
Existing `verify.py` still checks original baseline double-run metrics, now also pinning constructor input.
`ablation.py` independently double-runs each variant/split and saves actual probabilities and receipts.
Nonempty output directories are refused so historical evidence cannot be overwritten.
Missing/mismatched input prerequisites yield NOT_RUN before fitting; no fixture metrics substitute for results.

## Actual cached results

Final source smoke: PASS, first 128 original rows per split, 0.902 seconds,
externally bounded by `timeout --kill-after=5 120`. Complete cached histories were used.
Final source full pair: PASS, 1.135 seconds; seed 0; train 11411, val 566, test 702.
All variant/split probabilities exactly matched their repeated fits.

| Model | Val AUROC | Test AUROC |
|---|---:|---:|
| Original baseline / without constructor | 0.7251156462585033 | 0.7885912262723856 |
| With constructor | 0.740607709750567 | 0.7915385741472696 |

LOCAL reproducibility evidence only; **not RelBench-comparable**. No significance,
scientific improvement, causal benefit, official benchmark or multi-seed claim.
Smoke metrics are runtime checks only and differ from full evaluation.
`source-full/receipt.json` includes actual input/split/artifact hashes, model/feature settings,
source file hashes, software versions, actor/pane, seed and elapsed time.
`source-full/predictions.csv` retains both actual prediction columns with ordered label IDs/dates/targets.
Input parquet files remain read-only in their existing cache; ordered evaluated rows and cutoff/features are retained.

## TDD and cutoff evidence

Seven actual incremental RED/GREEN cycles. `red-N.txt` and `green-N.txt` are raw outputs.
The first API-introduction REDs include actual missing-interface errors, not fabricated assertion failures.
Early RED commits also contain their raw log artifacts and therefore are not claimed as canonical tests-only commits.
The seventh RED is tests-only, has an actual assertion failure and unchanged tests through final GREEN.
Canonical engine replay: PASS (`tdd-replay.stdout.txt`, `tdd-receipt.json`):
RED run=1 failed=1 errors=0 skipped=0; GREEN run=1 failed=0 errors=0 skipped=0.
The replay covers missing cutoff provenance only; it does not misrepresent the other focused cases as canonical replay.
All seven focused tests pass, including strict constructor history, guard timestamps, same-model ablation,
NOT_RUN, ambiguity, evidence preservation and missing provenance.
Previously existing tests/relmini files are untouched.

Independent cached strict-filter oracle (`audit_cached.py`, `source-oracle.stdout.txt`):
train=11411 (9699 with constructor history, 12 ambiguous latest teams),
val=566 (546 with history), test=702 (685 with history), all PASS.
It compares each label to independently filtered source history and checks original driver features unchanged.
This is a builder's additional check, not the separate verifier role.

Initial tiny smoke failed before fitting on the over-strict duplicate-team check; stderr and partial rows
are retained under `smoke/` and `smoke.stderr.txt`. No PASS receipt or metrics are claimed for it.
The corrected `smoke-2` passed in 0.932 seconds before the first full pair.
Later `final-*` and `source-*` runs refresh proof after production changes; earlier actual results remain retained.

## Configured verification and exact remaining gates

`/verify-project`: actual configured command via
`lead-verify relational --worktree /tmp/lead-wt/build-rel-18-20261025`.
PASS exit=0 timed_out=false dirty=false on `054b684641ce30fbdc0d3e85abad3bf34ddc1b0b`;
95 tests, raw configured stderr reports 15.406 seconds.
`configured-verification/{status.json,stdout.txt,stderr.txt}` is copied unchanged from `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261025T115833Z`.
Earlier clean-source suite PASS (94 tests at 353a7c84) is historical, not final-source acceptance.
`source-verify-py.stdout.txt`: actual verify.py PASS; labs/relmini/receipt.json updated from real execution.
Environment check PASS on verification source; original evidence path is in `environment.stdout.txt`.
Environment readiness expires and must be refreshed for later admission.

Only the builder's actual `/tdd` invocation (source 557fa7ba) and `/verify-project` invocation
(source 054b684641ce30fbdc0d3e85abad3bf34ddc1b0b) were recorded to the existing task skills-used.jsonl using actual actor/pane/output hashes.
No blast-radius/review/acceptance/closure invocation was recorded by this builder.

Remaining: parent launches a separately counted read-only verifier for `/blast-radius`, scoped configured
verification and Feature Map proof (`labtask`). Parent must preserve source/evidence-delivery lineage
and align source-bound admission/skill/replay records if a single report SHA is required;
canonical receipt covers implementation source, configured software check covers its evidence-only descendant.
No current-source acceptance was attempted or claimed. No reservation release, PR, push or merge by builder.
After successful distinct review and remaining admission/CI gates, only the authorized owner handles PR/publication/human merge.

## Actual commit history

```text
aefa262fc296124b1fba67275c5c600c76eff67c test(relmini): require strict cached constructor history (RED)
7d5a1d9c8474629fc15441882e8fec86926b0fa8 feat(relmini): aggregate constructor points strictly before labels
ae6a6a1a7fa5ce4869bd0e58a7aa43338613b210 test(relmini): require constructor cutoff guard (RED)
e2250f00dfe8610a0db7f2654126f573204a88f2 feat(relmini): enforce both constructor cutoff timestamps
f92a13bba40b6ff02412a3d8c02cf6f12c33c0b1 test(relmini): require paired feature ablation with identical model (RED)
673f0ab569ae91d2bfe29277f63871c3f0b36112 feat(relmini): toggle constructor features while preserving seed0 model
abf9ecbd0063adba5287c1c6f086fac93efc9f59 test(relmini): require honest NOT_RUN for absent cache (RED)
49dba626ca9f9122a91467905d86671dee3547e5 feat(relmini): retain actual cache-only paired predictions and receipts
b160115fd22259252ccd2922c65b3235672a97d3 test(relmini): make ambiguous historical team explicitly unknown (RED)
b527710300df6081ea42b1c3ab5be5ec6195d7ed feat(relmini): resolve shared-drive ambiguity without arbitrary team choice
1c73a0d13c448daf02e973bfe4b1687878de65ec test(relmini): preserve historical evidence on rerun (RED)
e252103b59c64782e5c6060b686164373a189942 feat(relmini): refuse evidence overwrite and document local paired protocol
353a7c84872add128595cc493556b4ed4dbbd1e8 evidence(relmini): retain actual seed0 pairs, smoke attempts and cutoff oracle
42f90b703c2a776f1cc85b96533a32888601ed5f test(relmini): reject missing cutoff provenance (RED tests only)
557fa7ba846588ec182ec2fd4b51fc98ae3a1a57 fix(relmini): fail closed when cutoff provenance is absent
054b684641ce30fbdc0d3e85abad3bf34ddc1b0b evidence(relmini): bind final source smoke, pair and canonical cutoff replay
```
