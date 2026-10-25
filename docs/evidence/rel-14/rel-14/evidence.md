# REL14 fresh evidence review

Reviewed screenshot source: `67bf8c68d34e7cb33684570633915935cce5cb4e`.
Software implementation: `341875ebcad870ffec05b31b1893b01677ba923e`.
Builder: `builder:rel-14`; capture producer: `lead:rel-14-capture`;
independent actual reviewer: `verifier:rel-14-captured`, this fresh separate session.
The producer records lead tip `a3dbf76` and explicit Piotr outside-sandbox capture
authority, using the installed headless shell and Playwright. This reviewer
inspected the resulting files and ran software audits; no browser rerun occurred.

Recommendation: **ACCEPT for Piotr**, with `RISK_FLAG: YES` for the preserved
historical missing sharpen invocation. No concrete introduced source defect found.
This is not engine acceptance or a waiver. Parent handles receipts, scoped waiver
if warranted, PR, hosted CI and merge. Hosted CI: NOT_RUN by this review.

## Actual search captures

All four exact PNGs were opened with the image tool. Each visibly shows the input
`baseline` and first lesson result links. Desktop is 1280x900; mobile is 390x900.
Mobile links wrap readably without visible horizontal clipping. The viewport
shows part of the results; the capture's DOM record, not the pixels alone, proves
19 results in each case. Text, href and ordering match the real search module;
all 19 relative lesson targets exist.

| Scheme | Desktop | Mobile | DOM results |
| --- | --- | --- | --- |
| file:// | [actual search PNG](external-capture/file-desktop-search.png) | [actual search PNG](external-capture/file-mobile-search.png) | 19 each |
| HTTP | [actual search PNG](external-capture/http-desktop-search.png) | [actual search PNG](external-capture/http-mobile-search.png) | 19 each |

[Capture result](external-capture/capture-result.json) retains source, identity,
authority, browser version/path, input, result text/hrefs/count, timestamps,
viewport, PNG SHA256, HTTP logs, pageerrors and failed_requests. The four recorded
interaction arrays are empty and the retained HTTP requests are 200. This does
not assert a globally error-free console/network or test clicking every link.
HTTP shows the existing Continue banner; file views do not. Unchanged
`assets/home.js` loads its independent navigation via fetch with a catch fallback;
search uses local classic scripts. File search success is not full navigation
parity. Raw [file manifest](external-capture/file-lead-evidence/manifest.json),
[file capture log](external-capture/file-lead-evidence/capture.log),
[HTTP manifest](external-capture/http-lead-evidence/manifest.json) and
[HTTP capture log](external-capture/http-lead-evidence/capture.log) are preserved.
Their desktop/mobile PNGs are pre-interaction captures, distinct from the four
search-result PNGs above. Manifest field `reviewer` names the lead producer;
it does not identify this independent reviewer.

## Independent source and blast-radius proof

Fresh [source audit](fresh-review/source-audit.log) passes: index is 8262 bytes;
228 external entries equal the previous embedded data; inline CSS, search module
and rest of the home page match the REL13 base apart from the specified script
and initialization changes. Course progress link and target survive. Entire tests
tree is frozen RED to GREEN; canonical receipt/report/log hashes and builder proof
hashes match. No new canonical replay or TDD invocation took place.

22 search tests pass, including coverage, title/headings/glossary, ranking,
multiword AND, diacritics, legacy export, deterministic output and the actual
page input handler under Node. Freshness passes. Isolated real-input copies
return 1 on stale/missing asset and broken wiring without writing or recreating
files; fresh default returns 0. [Capture audit](fresh-review/capture-audit.log)
checks dimensions and hashes of all search PNGs plus the raw lead-evidence PNGs,
recorded input/results/errors, and equality with the actual search module.
The audit script is evidence code, not an implementation or test-suite edit.

The key safety fact is proven by running real code: externalizing the complete
index preserves search data and behavior, with ordered classic scripts. The
existing PR workflow calls `scripts/search_index.py --check`, now checking the
JS default. Unused historical JSON is retained. The extra local script introduces
one required file to publish together with the home page; missing data is handled
by the existing unavailable message. Source correctness plus the four captures
clear that risk for the reviewed local bytes; hosted deployment remains untested.

Final delivery adds evidence only to screenshot source `67bf8c68...`. The exact
final SHA and fresh configured 133-test clean-delivery status, stdout/stderr,
search --check and clean-tree evidence are retained in the external fresh-verifier
accept directory. They are generated after this packaging commit to avoid a
self-referential commit. The old root READY.json is historical builder metadata,
not the fresh final-delivery verdict.

## Preserved history

[Prior verifier report](prior-verifier/source-pass-browser-pending.md) retains
SOURCE PASS / browser PENDING, unchanged, with its raw proof and prior-ready.json.
[Original builder narrative](builder-evidence-historical.md), canonical RED/GREEN,
receipt and report remain intact. Historical missing `sharpen` invocation was not
backfilled, retried or waived. This review does not manufacture invocation records.
