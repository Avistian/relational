# REL-16: bounded parser repair and retained review evidence

Local PR-body draft only; no remote PR, push or merge.

Reviewed source SHA: `86fe6e7284421f128686da00240aeddb5595cbab`.
Delivery payload SHA: `e1e1e8807b534136252df8c442cb352f24a471f9`.
The subsequent metadata commit adds this draft and delivery.json only; its exact
final package SHA is recorded in the external piotr-review/READY.json handoff,
avoiding a self-referential commit hash.

Quoted wildcard copy syntax and wildcard/normalized-root removals could produce
false file-set equality. The R1 source rejects these forms with exit 2 and no
success JSON, preserving ordinary quoted filenames and literal subtree removal.
This delivery adds evidence only under docs/evidence/rel-16/rel-16.

Separate read-only verifier `w1:p3X` (`/tmp/lead-wt/night-vrel16-r1`) retained
source **PASS**, **RISK_FLAG YES**, at the exact clean source SHA. Ten focused
Pages methods PASS; independent Bash/GNU cp parity has 18,497 actual and 18,497
generated destinations with zero membership differences. This builder performed
packaging integrity checks only and did not repeat those executions.

Review [evidence.md](evidence.md), [manifest.json](manifest.json),
[proof index](proof-index.md), and the copied
[verifier report](proof/evidence/r1/verifier/report.md).
Raw statuses/logs and original paths have SHA-256 hashes. Original READY was
untracked; its new proof copy does not change that historical provenance.
The invalid GNU rm public/. expectation and its separate correction before
production root repair remain preserved. Intermediate GREEN logs retain dirty
preceding-RED bindings. Original seven-tests-at-once cadence remains a limitation.

The parser supports bounded current-workflow syntax, not general Bash; literal
quoted wildcard filenames and destination renames remain unsupported. Parity
covers membership only, excludes two HTML byte mutations, and proves neither
published bytes nor empty directories, symlink validity, transitive link integrity
or deployment correctness. Manifest diagnostics cover direct references only.

Remaining gates: `/verify-project` **NOT_RUN** for R1 because configured discovery
includes prohibited model fitting; older 74 PASS is not applicable. Genuine local
TDD cycles are retained; R1 canonical replay/admission **NOT_RUN**, with no replay
or gate changes. Piotr risky-package review is **PENDING**. Acceptance is
**NOT_DONE**, lead/Piotr owned. Hosted CI, publication, push and merge **NOT_RUN**.
No source fixes, workers, installs, spend, training, downloads, linker copies,
or permission/profile changes were made for packaging.
