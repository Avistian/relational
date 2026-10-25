STATUS: PENDING
TICKET: REL-14
TASK: rel-14
SLICE: rel-14
RISK_FLAG: YES

Reviewed delivery: `67bf8c68d34e7cb33684570633915935cce5cb4e` (clean before/after).
Software source: `341875ebcad870ffec05b31b1893b01677ba923e`; delivery adds evidence only.
Worker: `builder:rel-14`, original pane `w1:p5X`.
Independent actual reviewer: `verifier:rel-14-final`, this separate verifier session.
Recommendation for Piotr: PENDING. Source PASS; browser PENDING. No engine acceptance or DONE claim.

Read own spec, issues and grill, committed builder READY and frozen/canonical evidence. Applied blast-radius and verify-project read-only. No implementation, test edits, canonical replay, browser/server retry, workers, installations, publication, or packaging commit.

Verification and blast radius

Fresh configured `bin/lead-verify relational --worktree /tmp/lead-wt/v-rel14-final-20261025`: PASS, exit_code 0, source.head `67bf8c68d34e7cb33684570633915935cce5cb4e`, dirty false. 133 tests passed. Copied status/stdout/stderr in raw/ are authoritative; stdout contains negative fixture diagnostics and notebook warnings and is preserved verbatim. This is software verification only.

Fresh focused search: 22 tests PASS, including independent coverage, headings/glossary, ranking, multiword AND, case/diacritics, legacy JSON export, full external-data execution and actual page input-handler execution under Node. Node supplies no browser or URL-scheme evidence.

The key safety fact was run against real source: decoded external data equals the complete previous embedded data (228 entries); inline CSS and assets/search.js are byte identical to REL13 base `edebddae007db2ab6e956e9fa824bb9c5ce18fd8`. Exact home comparison permits only the external script replacement and data initialization change. Rendering, result limit, empty/no-match handling, escaping via textContent, and navigation remain unchanged. Index is 8262 bytes, below 20000. REL13 Course progress link survives at index.html:150 and its target exists.

index.html:185-194 loads same-directory classic scripts synchronously before the handler; window.LessonSearchIndex is available without fetch, CORS-dependent JSON loading, async ordering or modules. Thus source supports file and HTTP loading. Missing/non-array data reaches the unavailable message at index.html:199. Actual browser/network loading remains unproven.

scripts/search_index.py:83-114 still supports explicit JSON output; existing CI invokes --check on the new default JS asset. Consumer search found no remaining production consumer of embedded lesson-search-data; historical JSON remains untouched. Isolated copies using actual lesson/glossary inputs prove default fresh exit 0, stale exit 1 without write, missing exit 1 without creation, broken script reference exit 1 without modification. The real production check passes without changing the worktree. See raw/audit.py and raw/audit.log for exact assertions, commands, diagnostics and return codes.

No concrete introduced code defect was found in the reviewed delivery. This conclusion is bounded by the absent browser proof, not a claim of visual parity.

Frozen and historical provenance

Final selected test digest `a966fc719494fe9efd22cf82cdc871bf08c2d1645b7b083454a4bf79067279fd` matches RED `575b49b9bd335ce1bb64205ff6db8b5c0b22c4c9`, GREEN software source and final delivery. Entire tests tree unchanged RED to GREEN. Retained canonical report/receipt/log hashes and all builder READY evidence hashes were independently checked. Retained RED ran 1 with 1 assertion failure (169499 >= 20000); retained GREEN ran 1 with zero failures/errors/skips. No new canonical execution took place. Original source, tests, canonical records and historical logs remain untouched.

Historical cheap gate: `SKILLS FAIL: missing invocation sharpen; run skill, then skill-record explicitly`. Existing sharpen.md does not establish invocation. This gap is disclosed and has not been backfilled or waived. No synthetic skill records were created.

Browser and CI boundary

file:// 1280x900: PENDING; file:// 390x900: PENDING.
HTTP 1280x900: PENDING; HTTP 390x900: PENDING.
Zero external images/raw capture logs were available at report generation; zero images inspected. No browser PASS inferred from index size, Node or source review. No previously denied browser/server operation was repeated.

Lead external producer must use evidence/capture-plan.json: term baseline, input #lesson-search, result #lesson-search-results li a; existing Playwright interpreter/cache identified there. Produce four actual viewport images after filling the input and waiting for >=1 visible matching result, plus raw argv/browser version, source/clean status, URLs, viewport dimensions, result text/count, timestamps, errors and image hashes. Actual producer identity must stay distinct from this reviewer. Independent visual inspection of all four and hash binding to capture raw logs are still required before an ACCEPT recommendation. No source change or new canonical replay is needed to resume visual review.

Hosted CI: PENDING / NOT_RUN. No final ACCEPT, engine acceptance, merge or publication is implied. No evidence packaging commit was made because browser evidence was absent. Final reviewed delivery remains the exact clean original SHA.

Evidence manifest

READY.json was written last. Its verification list names relative copied nonempty raw files with byte sizes and SHA256 hashes. The raw configured receipt binds the exact reviewed delivery, while the historical canonical receipt binds the earlier software source; these identities are deliberately retained.
