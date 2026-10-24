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
