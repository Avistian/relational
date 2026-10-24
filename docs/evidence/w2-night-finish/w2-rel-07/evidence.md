# REL-07 for Piotr review

Status: PACKAGED_FOR_PIOTR_REVIEW_NOT_ACCEPTED. RISK_FLAG: DATA_GAPS_AND_CONFIGURED_NOT_RUN.
Builder packaging only; independent reviewer remains verifier:w2-rel-07. Piotr review is required.

| Layer | Retained verdict |
| --- | --- |
| Source | PASS_SCOPED_READ_ONLY_REVIEW |
| Code | FOCUSED_PASS_CONFIGURED_NOT_RUN |
| Data | FAIL_EXISTING_LINKS |
| Acceptance | NOT_ACCEPTED |
| Delivery claims | QUALIFIED_SYNTHETIC_MODEL_FIT_IN_RETAINED_FULL_SUITE |
| Source risk | LOW_WITHIN_FILESYSTEM_CONTRACT |

Implementation: `8b961503896e9fb9ed4cc78235c60d2c11eeee85`.
Reviewed evidence delivery: `758820f0fc221065a5c5be0ff4337b609d4c0d4c`.
Package payload and final delivery bindings are in delivery-binding.json and the task handoff READY.json.
The retained verifier reviewed the earlier evidence delivery, not this packaging commit.

## Retained local RED / GREEN

RED `de7887c54a76dccba2e69c905036e2104fee3f5e`: exit 1, one assertion failure because the checker CLI is absent, zero errors.
GREEN `8b961503896e9fb9ed4cc78235c60d2c11eeee85`: exit 0, one integration test passes.
Frozen test SHA256: `c36e71299d71cb0bf1129f18149239f6055ee695056ad73c3122e165b96f9386`.
The separate verifier independently reproduced frozen RED/GREEN and focused PASS plus 23 external probes.
No new TDD execution or canonical replay was performed during packaging. Canonical execution: NOT_RUN.
Exact argv, cwd, head, stdout/stderr and frozen binding are retained in raw/builder and raw/verifier.

## Existing data findings

672 HTML pages; 20,179 checked targets; 6,006 exclusions. CLI exit 1, LEAD_RESULT: FAILED.
All **111 preexisting broken occurrences** remain in [broken-links.json](broken-links.json): 110 missing-file and one missing-fragment.
The fragment is reference/solution-map-atlas.html -> lessons/0058-surveys-meta-benchmarks.html#solution-evidence-funnel.
No content fixes or blanket completion claim. The independent census matched all records and repeated output was identical.
Builder tracked hashes cover 21,391 files; verifier hashes cover 21,395 after the four evidence files were added.

## Verification limits and reporting correction

Current configured /verify-project: NOT_RUN. Unchanged recursive discovery reaches
`tests/relmini/test_relmini.py:69-78` -> `labs/relmini/baseline.py:72-79` -> LogisticRegression.fit.
No suite execution, skip injection or argv/config override during packaging.
Historical builder PASS: exit 0, 68 tests, source.head `8b961503896e9fb9ed4cc78235c60d2c11eeee85`, dirty false,
receipt `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261024T202706Z/status.json`.
That PASS executed synthetic model fitting. The older README and READY no-model wording is qualified here; originals remain unchanged.
It establishes neither a verifier full-suite PASS nor a package delivery-SHA PASS nor a scientific result.

Filesystem href/src/poster scope only. No deployed routing, browser rendering, CSS/JS/srcset or non-HTML fragment coverage.
Stable-tree root/symlink containment was exercised; concurrent filesystem replacement races were not proved safe.

## Raw proof and remaining gates

[Full independent review](raw/verifier/REVIEW.md) and [verifier READY](raw/verifier/READY.json) preserve exact actor and risk findings.
[Manifest](manifest.json) binds stored and original raw SHA256 values. Files ending .gz are lossless copies: decompress before checking raw_sha256.
Original historical manifests and reports are retained byte-for-byte; packaging does not rewrite their provenance.
Supported structural preflight result is recorded in delivery-binding.json and task handoff; it is not verification or acceptance.

Piotr must review the data gaps and reporting qualification and explicitly dispose of the 111 findings through an authorized workflow.
Fresh configured verification remains gated by the model execution boundary. Canonical execution remains NOT_RUN by prohibition.
Package delivery has no new separate verifier review or hosted CI. Acceptance and merge remain unclaimed.
No push, merge, install, spend, model/protocol run, source change, permission/profile/gate change, ledger rewrite or worker launch.
