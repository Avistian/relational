# REL-16 builder evidence

Authority: the answered grill was delegated to Herder by Piotr's standing order.
The grill file's ANSWER_BY field does not establish a personal Piotr self-answer.
Scope is the exact current Pages publication set; omitted references are reports,
never additions. No workflow or course-content changes are included.

## Interface and design

Run the existing environment's Python:

```
.venv/bin/python scripts/pages_publish_list.py --compare .github/workflows/pages.yml
```

JSON contains sorted destination `files` relative to `public/`, their `sources`,
and three separate diagnostics: `missing_sources` (copy operands, including
unmatched globs), `missing_references` (manifest slug HTML or labPath absent on
disk), and `omitted_references` (manifest references present but not copied).
The last two lists cover the manifest's direct lesson/lab references, not a crawl
of every HTML, notebook, CSS or JavaScript link. An empty list does not establish
site-wide link integrity. All manifest entries are considered, regardless of
published status, and diagnostics never modify the generated set.

The default plan freezes the copy operations from base 503704f3. This deliberately
retains the legacy policy for this compatibility version; `--compare` detects
file-set drift in the current workflow. `--workflow PATH` overrides the frozen
plan for synthetic fixtures or explicit exploration. It is not necessary for the
normal comparison. Exit 0 means equality (or successful generation), exit 1
means a comparison difference, and exit 2 means invalid input or unsupported
syntax. Missing inputs and manifest gaps are diagnostic, not a deployment PASS.

The parser executes no shell and copies no files. It supports the current
workflow's literal copies, recursive directory/dot-content copies, globs,
optional-copy suffix, file conditionals, exact L058-L070 loop, and removal.
It ignores the exact two manifest-version content commands for file-set
purposes. Destination renames and arbitrary shell are unsupported. This is a
bounded interpreter for the current workflow, not a general Bash implementation.
Only destination membership is compared; byte identity, empty directories,
permissions and symlink target validity are outside this list's contract.

The approved spec/grill supplied the design decision and prioritized behaviors.
Installed TDD and codebase-design guidance were applied at the public CLI seam;
no additional plan-approval gate or out-of-scope design document was introduced.

## Checks and provenance

- Tests-only RED: 9e09a39b. Seven CLI tests ran; six assertion failures and no
  runner errors. The escape-rejection test already returned the expected status
  before the script existed; the six positive assertions supply genuine RED.
- Implementation GREEN: 1ba8e647d4ef15eb573b8caa13033cdad5e3b9ff. The original
  test files are byte-identical. Seven tests pass. Test support consists of
  standard unittest setUp/addCleanup fixture construction; no test-local
  implementation helper or implementation declaration was needed.
- Independent outside check imports no implementation. Bash interprets the
  actual workflow's copy/condition/loop/removal commands; GNU cp creates a
  disposable publication tree. It omits only the two HTML content-version
  commands. Its actual file inventory equals all 18,497 generated destinations.
  Missing sources, missing manifest references and omitted manifest references
  are separately empty. Scratch is removed by TemporaryDirectory.
- The first outside-check attempt failed because hardlinks crossed filesystem
  devices. Its raw stderr and command remain under `attempt-1-*`; the completed
  check uses ordinary GNU cp with optional reflinks. No permission change.
- `/verify-project`: configured `lead-verify relational --worktree ...` passed
  74 tests with exit 0 at implementation GREEN. Its status records dirty:true
  because this evidence directory was untracked. It is software verification,
  not independent verifier acceptance or a scientific experiment result.

Raw outputs and machine-readable statuses are under `evidence/builder/`.
The historical committed package contains this README as its narrative report;
it does not contain a builder `report.md` or `READY.json`. The original builder
wrote READY.json after commit, leaving it untracked in the original worktree.
The separate verifier preserved that external receipt at
`/home/avist/agent-stack/lead/.lead/tasks/rel-16/evidence/verifier/original-builder-READY.json`.
Its canonical replay binding is historical evidence, not a replay of R1.
Separate verification/review remains owned and launched by the parent.
No subagents, installs, downloads, training, spend, push, merge or profile/gate
changes were performed.

## R1 correction

R1 rejects quoted wildcard syntax before quote removal, wildcard `rm -rf`
operands, and removal of the normalized `public` root with exit 2 and no JSON
success report. Literal copies, unquoted copy globs, quoted ordinary filenames
and literal subtree removal keep their existing behavior. These are bounded
fail-closed checks; no general shell interpreter is introduced.

New CLI regressions are in `tests/pages/test_publish_list_shell_regressions.py`.
Each behavior was committed as genuine tests-only RED before its minimal fix.
The original frozen test files and all historical raw evidence are unchanged.
One additional root fixture initially expected GNU rm to accept `public/.`;
GNU rm refuses that spelling. The incorrect fixture commit and raw failure
remain preserved; a separate correction commit established clean, genuine RED
using successful root-removal spellings before the production fix.

New R1 raw logs, exact source bindings, provenance, and `report.md` live outside
Git under `/home/avist/agent-stack/lead/.lead/tasks/rel-16/evidence/r1/`.
Its `READY.json` is written last after the final commit and checks; it is an
external task receipt, not a committed repository artifact. R1 runs focused
Pages fixtures and the Bash/GNU cp destination inventory. Full `/verify-project`
is NOT_RUN because the configured suite calls model.fit; no training or config
change is authorized. No new canonical replay or independent review is claimed.
