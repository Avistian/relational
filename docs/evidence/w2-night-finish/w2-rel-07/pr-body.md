# REL-07: retain evidence for Piotr risk review

The published filesystem checker passes its focused contract; existing navigation has 111 broken occurrences (110 missing-file, one missing-fragment). No content was repaired.

[Readable evidence](docs/evidence/w2-night-finish/w2-rel-07/evidence.md), [manifest](docs/evidence/w2-night-finish/w2-rel-07/manifest.json), [full independent review](docs/evidence/w2-night-finish/w2-rel-07/raw/verifier/REVIEW.md).

Implementation SHA: `8b961503896e9fb9ed4cc78235c60d2c11eeee85`.
Reviewed evidence SHA: `758820f0fc221065a5c5be0ff4337b609d4c0d4c`; actor `verifier:w2-rel-07`.
Package payload SHA: `86a1c39435b0273a020eaeb2051e3776d688a76f`.
Final local delivery SHA: resolved by `/home/avist/agent-stack/lead/.lead/tasks/w2-night-finish/evidence/w2-rel-07/piotr-review/READY.json` after this binding commit. It cannot be embedded in its own commit. This body is local only.

Source verdict PASS_SCOPED_READ_ONLY_REVIEW; code FOCUSED_PASS_CONFIGURED_NOT_RUN; data FAIL_EXISTING_LINKS; acceptance NOT_ACCEPTED.
RISK_FLAG: DATA_GAPS_AND_CONFIGURED_NOT_RUN. Piotr review required.

Retained local RED exit 1 (one assertion failure) and GREEN exit 0 (one test) were independently reproduced before packaging. No canonical replay.
Historical configured PASS: 68 tests, exit 0, clean implementation SHA above; it included synthetic model fitting and does not cover this delivery SHA.
Current configured /verify-project NOT_RUN at the model.fit boundary. No fresh configured PASS, model/protocol run, scientific result, hosted CI or package reviewer approval is claimed.

Supported nonexecuting structural preflight: BLOCKED_STRUCTURAL, exit 1; `HUMAN EVIDENCE: code changed since reviewed source; only declared evidence folder may follow source SHA`. It was run once; gates remain unchanged. Full provenance is in delivery-binding.json.

Remaining gates: Piotr risk review; authorized disposition of all 111 preexisting link findings; configured verification execution boundary; canonical execution NOT_RUN by prohibition; package delivery separate review and supported human-evidence structural gate remain unfulfilled. No acceptance or merge.

## 2026-10-25 scoped human acceptance addendum

Piotr explicitly authorized manual accepted-DONE and human-evidence receipts from the separate morning ACCEPT review and chat approval, via `piotr-scoped-receipt-waiver-20261025.json` dated 2026-10-25T07:52:29.421091+00:00. This is scoped manual human acceptance, not engine-run validation. RISK_FLAG: YES; historical DATA_GAPS_AND_CONFIGURED_NOT_RUN remains unchanged. The 111 pre-existing broken occurrences remain unresolved. Earlier statements above record historical package state.

Morning reviewer `reviewer:w2-rel-07` recommended ACCEPT for delivery `f696a2426cb6945f3cf5d33496a42c7cc1e1f236`; morning report SHA256 `aa4619d43604141543e905e3cb78ac8b87d45607d208f7e1dd7ea3e2e5912616`. Implementation remains `8b961503896e9fb9ed4cc78235c60d2c11eeee85`. The publication descendant changes this PR body only and has no separate descendant review; main must assess that limitation before merge.

Historical exact-delivery verify-project receipt: 2026-10-24T22:16:01Z, PASS, exit 0, clean f696a2426cb6945f3cf5d33496a42c7cc1e1f236, 68 tests. It is not fresh verification or verification of this documentation descendant. No tests, browser, canonical replay, environment freshness check, full human-evidence validator, or skill invocation history was manufactured for publication. Those process requirements are expressly waived for these manual receipts; any retained-hash acceptance check is only that check. Current engine hash identifies compatibility only. Hosted CI will be reported from GitHub, not inferred from an empty check list. Draft publication only; worker does not merge.
