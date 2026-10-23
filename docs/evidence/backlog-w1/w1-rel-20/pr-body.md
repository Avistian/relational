Adds a read-only requirement/lock coverage check. Package names are normalized; the 17 existing gaps are listed explicitly with `refresh lock: Piotr`. A new unlocked package or removal of a previously covered lock entry fails. Requirements and lock files are unchanged.

Reviewed source: `31e07db926d3b08855aa2b651bb51cd75c75719e`. Delivery adds only the fixed evidence artifacts to that source. Canonical tests are unchanged from tests-only RED `d12f8b0333c2f277710d3119c5a54ccf72a0c053`.

[Human-readable RED/GREEN evidence](docs/evidence/backlog-w1/w1-rel-20/evidence.md), [manifest and retained verification detail](docs/evidence/backlog-w1/w1-rel-20/manifest.json), [RED](docs/evidence/backlog-w1/w1-rel-20/red.log), [GREEN](docs/evidence/backlog-w1/w1-rel-20/green.log), [verification stdout](docs/evidence/backlog-w1/w1-rel-20/verification.log).

Canonical RED: 4 run, 4 assertion failures, 0 errors, 0 skips. Identical GREEN: 4 run, 0 failures, 0 errors, 0 skips. The absent declared CLI in RED returned 2 rather than the asserted checker result; this is the approved absent-helper seam. The first checkout timed out after 30 seconds before tests; the one authorized retry passed. No underlying Git stderr was exposed and no EROFS/quota cause is asserted.

Local project verification: **PASS**, 46 tests, exit 0, no timeout, clean source. The deterministic evidence renderer exposes stdout only: `FAIL hash check: x: sha256 mismatch` comes from an existing expected negative fixture. It does not represent the suite outcome. Full passing stderr and status follow verbatim. These are local software checks; actual final-head draft PR CI is still pending, with no CI/PR URL asserted here.

The separate verifier `verifier:w1-rel-20` passed the source review. The exact source review and independent probe output below are retained snapshots. Final committed-proof review and explicit reviewer `RISK_FLAG` binding remain pending; no risk classification, acceptance or merge eligibility is inferred.

Known baseline gaps: dill, duckdb, einops, entmax, fasttext-wheel, h5py, ml-dtypes, pymetis, pytorch-frame, pytorch-tabnet, rdkit, relbench, rtdl-num-embeddings, rtdl-revisiting-models, skrub, torch-geometric, torcheval. Actual files: 34 direct normalized names, 121 locked names, 17 covered direct names, 17 baseline gaps. Independent negative probes reject a new normalized dependency, a removed numpy lock entry, and an unsupported recursive include. Contradictory versions/inactive markers still pass name presence as intended.

Limits: leading named requirements only; suffix syntax, activated markers, constraint satisfaction, transitive resolution and installed-package compatibility are outside the check. Exact baseline counts deliberately flag future file drift. A previously tolerated baseline name can disappear and reappear without a temporal regression ledger. No installs, experiments, browser validation, scientific execution, provider changes or merge were performed. Environment readiness imports/data hashes are a separate gate.

<details>
<summary>Passing project verification status (exact)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184540Z/status.json`

SHA256: `c2fc1cd67e8bb338b238be0be27f9857b1677a7eb5a22749b475e59203487e06`

```json
{
  "project": "relational",
  "root": "/tmp/lead-wt/w1-rel-20",
  "argv": [
    "/home/avist/Projects/relational/.venv/bin/python",
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-p",
    "test_*.py"
  ],
  "timeout": 600,
  "exit_code": 0,
  "timed_out": false,
  "verdict": "PASS",
  "duration_s": 3,
  "started_utc": "20261023T184540Z",
  "source": {
    "head": "31e07db926d3b08855aa2b651bb51cd75c75719e",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

</details>

<details>
<summary>Full project verification stderr (exact; 46 tests OK)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184540Z/stderr.txt`

SHA256: `23f6c1096ddd3dcfced843b4c03bcec060ea95d5ad4d75c8d2df0e43af005f67`

```text
..............................................
----------------------------------------------------------------------
Ran 46 tests in 1.423s

OK
```

</details>

<details>
<summary>Full project verification stdout (exact expected negative fixture)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184540Z/stdout.txt`

SHA256: `ac82d3fc9976f1d7b1f404b4f05cae44fb9f53a3cf9ed3d94bfb3b57cb749334`

```text
FAIL hash check:
  x: sha256 mismatch
```

</details>

<details>
<summary>Separate source review (exact; final proof review pending)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/verifier-evidence/w1-rel-20/w1-rel-20.md`

SHA256: `707f19939a309fd30c4c4df25ccb4305438ad674c46b118aa6acb86c75f0c690`

```text
# REL-20 independent source review

Verdict: PASS for source 31e07db926d3b08855aa2b651bb51cd75c75719e. No blocking source findings. Delivery remains INCOMPLETE pending committed human proof, managed draft PR, actual final-head green CI, and Lead validation/acceptance. This is a verifier review, not a DONE acceptance report.

Reviewer: verifier:w1-rel-20; pane w1:p14; worktree /tmp/lead-wt/v-w1-rel-20. Separate from builder:w1-rel-20. No implementation edits, new agents, installs, publication, redeclaration, receipt reuse on another tree, or acceptance performed.

## What changed and affected consumers

Against 00f3dfd721dc66bbab3f9672705fd0f60fe8a416 the only source changes are tests/env/{__init__.py,check_requirements.py,test_requirements.py}. Requirements, historical lock, teaching assets and workflows are unchanged. tests/env/test_requirements.py:15 invokes the real CLI in a subprocess. Existing .github/workflows/pr-checks.yml:44 discovers the suite; __init__.py makes the new subdirectory discoverable. Symbol search found no production consumers. No hosted CI execution is inferred from local discovery.

The checker normalizes names with case folding and collapsed punctuation, strips extras/constraints/markers from the name, compares direct names with lock names, and permits only an explicit fixed baseline (tests/env/check_requirements.py:10-28,31-48,62-74). It is stdlib-only. Baseline ownership is explicitly `refresh lock: Piotr`.

## Safety fact and executed proof

Safety depends on the gap allowance being fixed rather than derived from current files, and newly missing names causing nonzero CLI exit independently of installed distributions. Proven at blast-radius level 4 (real code executed), not running-app/browser level.

Raw independent proof: reviews/w1-rel-20-evidence/probes.txt. Each call uses /home/avist/Projects/relational/.venv/bin/python -B -S against the actual checker; exact argv, stdout/stderr and status are retained. Results:

- Actual baseline: exit 0; requirements=34 locked=121 covered=17 baseline_gaps=17. Independently extracted set difference exactly matches all 17 reported gap names.
- Disposable requirements copy plus Verifier_New.Dependency[extra]>=999: exit 1; UNEXPECTED GAP: verifier-new-dependency.
- Disposable lock with numpy removed: exit 1; UNEXPECTED GAP: numpy.
- Demo__Package vs demo.package with contradictory versions and an inactive marker: exit 0, demonstrating name-only scope.
- Recursive include -r hidden.txt: exit 2 with explicit parser error, so unsupported includes do not silently evade coverage.

Exact baseline: dill, duckdb, einops, entmax, fasttext-wheel, h5py, ml-dtypes, pymetis, pytorch-frame, pytorch-tabnet, rdkit, relbench, rtdl-num-embeddings, rtdl-revisiting-models, skrub, torch-geometric, torcheval.

## Verification and environment

Actual installed /verify-project run:
/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/v-w1-rel-20

Read all three files at /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184218Z. status.json: verdict PASS, exit_code 0, timed_out false, source.head 31e07db926d3b08855aa2b651bb51cd75c75719e, dirty false, root this verifier tree. stderr.txt: Ran 46 tests; OK. stdout.txt contains the existing negative-fixture `FAIL hash check: x: sha256 mismatch`; quoting stdout alone would misrepresent the passing suite. This is local software verification only.

Launch-time environment check command: /home/avist/agent-stack/lead/bin/lead-env relational check --worktree /tmp/lead-wt/v-w1-rel-20. PASS at /home/avist/agent-stack/lead/.lead/evidence/env/relational/1792780929315158286; requested source head, dirty false. Readiness imports/data hashes are separate from checker guarantees. Five-minute expiry applies; acceptance needs its own supported fresh check on the builder tree. Review files were created after these clean-source checks; current dirtiness is evidence only.

## Independent audit of builder proof

Read source-ready.md and canonical receipt/logs under /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20. Verified every receipt artifact SHA256 and its bound draft-source-report.json SHA256. Receipt source head matches requested head; original builder worktree/declaration binding is preserved. This audit does not create a new canonical receipt or cover a future final-report hash.

Tests-only RED d12f8b0333c2f277710d3119c5a54ccf72a0c053 precedes implementation 886115d8 and dependency-removal refinement 31e07db9. git diff d12f8b03 HEAD -- tests/env/test_requirements.py is empty. RED log has four assertion failures, zero errors/skips: subprocess exit 2 for the absent declared checker versus asserted expected exit 0/1. This is the expressly allowed absent-helper seam, not an import error. GREEN has four passing tests, zero failures/errors/skips. The first canonical attempt timed out at Git RED worktree-add before tests; retained canonical-attempt-1.txt discloses 30-second timeout and unavailable underlying Git stderr. Retry stdout says TDD replay RED/GREEN verified; exit file is 0. No unsupported filesystem cause is inferred.

## Risks and limits

No confirmed source defect in the requested contract. Residual boundaries:

- High impact if presented as installation compatibility, but outside contract: parser checks leading names, not suffix validity, marker activation, constraints, transitive resolution or installed versions (check_requirements.py:32,42-48). Executed contradictory-version probe confirms this boundary.
- Low current likelihood, moderate future maintenance cost: exact baseline counts in test_requirements.py:24 will fail on legitimate requirement/lock changes. This is deliberate drift visibility and requires reviewed baseline maintenance, not automatic allowance.
- The fixed 17-name allowance remains tolerated if a previously known gap is later covered then reappears (check_requirements.py:63-64). This is a historical-name baseline, not a temporal regression ledger; no such lock update exists in this diff.
- Human-proof files were absent at the reviewed head. Builder explicitly discloses this; do not claim committed evidence, a PR link, final CI or acceptance from the source handoff. No PR/CI links were supplied and none are verified here. UI/browser screenshots are not applicable (FEATURE_KIND non-ui).

## Before delivery

Lead retains this review and raw probes, records/retains skill outputs, prepares committed human proof with correct stdout limitation disclosure, preserves reviewed source/test identity through evidence-only commits, obtains actual draft PR CI, and validates the final report through supported gates. Cheapest regression check: /home/avist/Projects/relational/.venv/bin/python -B -m unittest discover -s tests/env -p 'test_*.py'; the passing project suite already exercised these four cases. No additional reviewer requested or launched.
```

</details>

<details>
<summary>Independent negative and scope probes (exact)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/verifier-evidence/w1-rel-20/probes.txt`

SHA256: `a4349344f9bb2f8a00b002cc4a3eee1fc1283e97a306f67f22aae558b9350aa8`

```text
Real baseline without site packages 
ARGV: ['/home/avist/Projects/relational/.venv/bin/python', '-B', '-S', '/tmp/lead-wt/v-w1-rel-20/tests/env/check_requirements.py', '/tmp/lead-wt/v-w1-rel-20/requirements-labs.txt', '/tmp/lead-wt/v-w1-rel-20/requirements-labs.lock.txt'] 
EXIT: 0 
STDOUT:
requirements=34 locked=121 covered=17 baseline_gaps=17
Known gaps only; refresh lock: Piotr
KNOWN GAP: dill
KNOWN GAP: duckdb
KNOWN GAP: einops
KNOWN GAP: entmax
KNOWN GAP: fasttext-wheel
KNOWN GAP: h5py
KNOWN GAP: ml-dtypes
KNOWN GAP: pymetis
KNOWN GAP: pytorch-frame
KNOWN GAP: pytorch-tabnet
KNOWN GAP: rdkit
KNOWN GAP: relbench
KNOWN GAP: rtdl-num-embeddings
KNOWN GAP: rtdl-revisiting-models
KNOWN GAP: skrub
KNOWN GAP: torch-geometric
KNOWN GAP: torcheval
PASS: normalized package presence (versions not checked)
STDERR:

INDEPENDENT actual missing names: dill, duckdb, einops, entmax, fasttext-wheel, h5py, ml-dtypes, pymetis, pytorch-frame, pytorch-tabnet, rdkit, relbench, rtdl-num-embeddings, rtdl-revisiting-models, skrub, torch-geometric, torcheval
Disposable newly unlocked package 
ARGV: ['/home/avist/Projects/relational/.venv/bin/python', '-B', '-S', '/tmp/lead-wt/v-w1-rel-20/tests/env/check_requirements.py', '/tmp/rel20-review-qkpes9us/requirements.txt', '/tmp/lead-wt/v-w1-rel-20/requirements-labs.lock.txt'] 
EXIT: 1 
STDOUT:
requirements=35 locked=121 covered=17 baseline_gaps=17
Known gaps only; refresh lock: Piotr
KNOWN GAP: dill
KNOWN GAP: duckdb
KNOWN GAP: einops
KNOWN GAP: entmax
KNOWN GAP: fasttext-wheel
KNOWN GAP: h5py
KNOWN GAP: ml-dtypes
KNOWN GAP: pymetis
KNOWN GAP: pytorch-frame
KNOWN GAP: pytorch-tabnet
KNOWN GAP: rdkit
KNOWN GAP: relbench
KNOWN GAP: rtdl-num-embeddings
KNOWN GAP: rtdl-revisiting-models
KNOWN GAP: skrub
KNOWN GAP: torch-geometric
KNOWN GAP: torcheval
UNEXPECTED GAP: verifier-new-dependency
FAIL: 1 new unlocked package(s)
STDERR:

Disposable removal of previously covered numpy 
ARGV: ['/home/avist/Projects/relational/.venv/bin/python', '-B', '-S', '/tmp/lead-wt/v-w1-rel-20/tests/env/check_requirements.py', '/tmp/lead-wt/v-w1-rel-20/requirements-labs.txt', '/tmp/rel20-review-qkpes9us/lock.txt'] 
EXIT: 1 
STDOUT:
requirements=34 locked=120 covered=16 baseline_gaps=17
Known gaps only; refresh lock: Piotr
KNOWN GAP: dill
KNOWN GAP: duckdb
KNOWN GAP: einops
KNOWN GAP: entmax
KNOWN GAP: fasttext-wheel
KNOWN GAP: h5py
KNOWN GAP: ml-dtypes
KNOWN GAP: pymetis
KNOWN GAP: pytorch-frame
KNOWN GAP: pytorch-tabnet
KNOWN GAP: rdkit
KNOWN GAP: relbench
KNOWN GAP: rtdl-num-embeddings
KNOWN GAP: rtdl-revisiting-models
KNOWN GAP: skrub
KNOWN GAP: torch-geometric
KNOWN GAP: torcheval
UNEXPECTED GAP: numpy
FAIL: 1 new unlocked package(s)
STDERR:

Normalization only; versions and markers not evaluated 
ARGV: ['/home/avist/Projects/relational/.venv/bin/python', '-B', '-S', '/tmp/lead-wt/v-w1-rel-20/tests/env/check_requirements.py', '/tmp/rel20-review-qkpes9us/requirements.txt', '/tmp/rel20-review-qkpes9us/lock.txt'] 
EXIT: 0 
STDOUT:
requirements=1 locked=1 covered=1 baseline_gaps=0
Known gaps only; refresh lock: Piotr
PASS: normalized package presence (versions not checked)
STDERR:

Unsupported recursive requirement fails closed 
ARGV: ['/home/avist/Projects/relational/.venv/bin/python', '-B', '-S', '/tmp/lead-wt/v-w1-rel-20/tests/env/check_requirements.py', '/tmp/rel20-review-qkpes9us/requirements.txt', '/tmp/rel20-review-qkpes9us/lock.txt'] 
EXIT: 2 
STDOUT:
STDERR:
ERROR: /tmp/rel20-review-qkpes9us/requirements.txt:1: expected a named requirement

Canonical receipt artifact and draft-report hashes MATCH; binding remains original builder draft/tree.
ALL INDEPENDENT PROBES PASS
```

</details>

<details>
<summary>Historical checkout timeout diagnostic (exact)</summary>

Retained path: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20/canonical-attempt-1.txt`

SHA256: `97ffe7e85b4618841a2e20d4296365270d1c9cf2ad48e461db8d01edb45df4be`

```text
Command: /home/avist/agent-stack/lead/bin/lead-guard tdd validate --report /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20/draft-source-report.json --receipt /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20/canonical-tdd.json --worktree /tmp/lead-wt/w1-rel-20
Exit code: 1
Exact captured output:
Command '['git', '-C', '/tmp/lead-wt/w1-rel-20', 'worktree', 'add', '--detach', '/tmp/lead-tdd-wxap_ipy/tree', 'd12f8b0333c2f277710d3119c5a54ccf72a0c053']' timed out after 30 seconds

Classification: actual 30-second subprocess timeout during RED checkout; no assertion replay executed. Tool returned this diagnostic without underlying Git stderr. No observed EROFS or quota diagnostic; neither ruled out by missing captured stderr.
```

</details>

Historical blocked audit remains intact at `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20/original-blocked-source-ready.md`, SHA256 `431a701588f8d6f437e0425da0b0c013343917ea46ff91e427023a2801491a6b`. This is historical pre-repair evidence, not the current outcome. The successful retry receipt and original raw logs remain at `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/builder-evidence/w1-rel-20/canonical-tdd.json`; acceptance must bind the final report through its supported replay gate.
