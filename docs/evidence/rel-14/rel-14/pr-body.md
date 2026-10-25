# REL14: external script-loadable lesson search index

Move the full lesson index from home HTML into a generated classic script.
Home shrinks from 169499 to 8262 bytes; search semantics, inline styling and
the REL13 Course progress link are preserved. Ordered local scripts support
the intended file:// and HTTP contract without an inline data fallback.

Validation at software SHA `341875ebcad870ffec05b31b1893b01677ba923e`:
clean configured `/verify-project` PASS (133 tests, exit 0); search suite PASS
(22 tests); generator freshness PASS; stale/missing default-asset counterexamples
correctly exit 1. Canonical frozen standalone RED `575b49b9` -> GREEN `341875eb`
passed. Retained proof: [builder evidence](evidence.md).

Browser proof: PENDING separate verifier, actual known-term search and four
screenshots for file:// and HTTP at 1280 and 390 widths. Independent
`/blast-radius`: PENDING. Hosted CI: NOT_RUN. This local draft does not claim
acceptance, browser success, publication or merge.
