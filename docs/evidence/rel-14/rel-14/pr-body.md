# REL14: external script-loadable lesson search index

Home loads the complete generated lesson index as an ordered classic script,
shrinking index.html from 169499 to 8262 bytes. Search semantics, styling and the
REL13 Course progress link are preserved. Screenshot source is
`67bf8c68d34e7cb33684570633915935cce5cb4e`; subsequent delivery changes contain
only evidence. Builder is `builder:rel-14`, capture producer is
`lead:rel-14-capture`, independent reviewer is `verifier:rel-14-captured`.

Actual `baseline` interactions record 19 result links each:
[file desktop](external-capture/file-desktop-search.png),
[file mobile](external-capture/file-mobile-search.png),
[HTTP desktop](external-capture/http-desktop-search.png), and
[HTTP mobile](external-capture/http-mobile-search.png).
All four were independently inspected with the image tool; viewport dimensions,
hashes, result text/hrefs/count and error context were validated against the raw
[capture result](external-capture/capture-result.json) and lead-evidence logs.
Screenshots show viewport subsets of the 19 results.

Independent blast-radius/source audit passes; 22 search tests and freshness,
stale/missing/wiring counterexamples pass. Frozen canonical RED/GREEN proof is
preserved without replay. Exact final-delivery configured verification (133 tests)
and clean status are recorded externally after the evidence commit; see
[evidence details](evidence.md) and the fresh-verifier acceptance package.

Recommendation for Piotr: ACCEPT, RISK_FLAG YES for the historical missing
sharpen invocation, retained without backfill or waiver. Prior source-PASS /
browser-PENDING report is preserved. Hosted CI NOT_RUN; no engine acceptance,
publication or merge is claimed. Parent owns receipts, waiver decision and shipping.
