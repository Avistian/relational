# REL-16: reject unsupported publish-list syntax

Quoted wildcard copies and wildcard or normalized-root removals could report false file-set equality. The bounded inventory parser rejects these forms with exit 2 and no success JSON. Ordinary quoted filenames and literal subtree removal remain supported.

## Explicit scoped human acceptance

Piotr explicitly authorized manual accepted-DONE and human-evidence receipts on 2026-10-25 under `piotr-scoped-receipt-waiver-20261025.json` (recorded 07:52:29 UTC), using the separate morning ACCEPT recommendation and chat approval. **RISK_FLAG: YES remains unchanged.** This is scoped human acceptance, not engine acceptance or a global policy change. Humans retain merge authority.

Reviewed source: `86fe6e7284421f128686da00240aeddb5595cbab`.
Original reviewed delivery: `0c993e8cb66e3ebcd4bde430a654eebddeb85253`.
Delivery payload: `e1e1e8807b534136252df8c442cb352f24a471f9`.
This publication descendant adds only this lowercase PR body; it has no separate verifier review. The morning review covers the original delivery, not this later file.

Separate morning verifier `reviewer:rel-16`, pane `w1:p4Z`, recommended ACCEPT at 2026-10-25 07:11:38 UTC. Morning REPORT.md SHA256: `4e96c6a74a1b5d648aaf2cc6802b7a2deefd3e4f9ee9907689c3ae7a2cdb6419`. Its raw evidence records 10 focused Pages methods PASS and 18,497-file membership parity. These are retained reviewer results, not executions by this publication builder.

Historical configured verification at 2026-10-24 22:16:14 UTC binds clean original delivery `0c993e8c...`, exit 0, 77 tests OK. It does not verify this new descendant. Historical TDD and original canonical records retain their original identities and limitations. No fresh /tdd, /verify-project, canonical replay, browser capture, test execution, environment freshness or skill invocation history is claimed.

The scoped waiver covers the missing process/invocation and current-source canonical acceptance records, nonstandard evidence schema and descendant allowlist, absent sealed environment/human-evidence engine receipts, and risk YES human disposition. Receipt engine hashes identify compatibility only; they do not claim validator execution. Full acceptance validators and canonical/process retry are NOT_RUN. The managed publisher may check retained receipt hashes; that is not canonical or test execution.

## Evidence and residual risk

Read [human-readable evidence](docs/evidence/rel-16/rel-16/evidence.md), [manifest](docs/evidence/rel-16/rel-16/manifest.json), [proof index](docs/evidence/rel-16/rel-16/proof-index.md), and [original verifier report](docs/evidence/rel-16/rel-16/proof/evidence/r1/verifier/report.md). The original uppercase PR-body.md and all committed historical reports are preserved, including their then-current NOT_RUN/PENDING statements.

The parser supports current bounded workflow syntax, not general Bash. Quoted literal wildcard filenames and destination renames remain unsupported. Membership parity excludes two HTML byte mutations and proves neither published bytes, empty directories, symlink validity, transitive links nor deployment correctness. The invalid historical GNU rm public/. assertion, correction, dirty intermediate GREEN bindings and original seven-tests-at-once cadence remain disclosed in the preserved packet. Hosted CI must be queried separately; no empty checks or historical runs establish current CI success. No source or harness changes, installs, spend or merge are part of this publication.
