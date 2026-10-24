# REL21 grouped-row evidence package for human review

Reviewed source SHA: `1a3752fbd2dbf6d25a2d75ac92c326cfd8626d7a`.
Evidence payload delivery SHA: `810b70a14b452530ba8f1d44e92cfccdacd6afd4`.
Final documentation-binding commit: the commit adding this file; its exact SHA is recorded in the external task `evidence/piotr-review/READY.json`. This avoids a self-referential commit hash. The payload commit contains all proof and hashes; the later commit adds only this local body.

Start with [evidence.md](evidence.md). Original source/course files and earlier evidence remain byte-identical. This is a local PR body; no remote PR exists for this delivery.

Retained separate verifier: `verifier:rel-21-grouped`, pane `w1:p30`, clean exact source, code/data PASS, `RISK_FLAG: NO`. All 228 manifest pages are covered; no observed row gaps, planned disagreements or missing/unlisted HTML. Focused fixtures: 8 PASS. Independent proof: 12 parser boundary checks and 6 CLI probes. These are retained results, not new executions by the packaging builder.

Retained TDD: RED `06e82129ac40322461819cde82725cc0247c495c`, eight missing-implementation assertion failures, exit 1; GREEN `beda5136f63ea13c78634c03756b556ab4360413`, same immutable eight tests PASS, exit 0. No RED or canonical replay was run. The verifier's original proof-reader error is preserved separately and is not a TDD RED.

Package validation: 29 byte-identical raw proof copies, 39 retained proof-map entries verified, payload SHA-256 map verified, only new files in the owned evidence directory, and supported non-UI structural preflight PASS. Preflight checks workflow structure only; no hosted CI or configured-suite execution is implied. See the external handoff for the packaging-only byte-check operational note.

Acceptance gaps: configured verify-project suite NOT_RUN (exit_code null; configured discovery invokes prohibited model fitting); canonical replay NOT_RUN (prohibited); lead/human acceptance PENDING; hosted CI NOT_RUN. Separate source review does not certify delivery documentation. No further review loop, push, merge, downloads, installs, spend, agents, training, permission/profile/gate changes or shared-ledger rewrite.

Requested human action: review the retained grouped-row coverage proof and resolve acceptance gaps through the lead within explicit authority. Package status is PACKAGED_FOR_HUMAN_REVIEW, not accepted or DONE.
