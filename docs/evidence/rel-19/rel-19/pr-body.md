Package the REL-19 cached leakage audit with authentic builder and separate
verifier raw proof, original receipts, origin paths/hashes and current cached checks.
Production and frozen tests are unchanged.

Cached verification: 2 tests passed with zero skips; all 12,679 labels checked.
Frozen test and five cache hashes match. Missing-cache reports NOT_RUN; the existing
aggregate corruption probe detects its expected assertion failure (not TDD RED).

Herder delegated an explicit audit-only RED waiver and approved cached-only
verification. This is manual-human authority, not a new Piotr answer or engine PASS.
Full suite, canonical replay and hosted CI: NOT_RUN. The supported skill gate still
fails on the missing historical grill invocation; no rows were invented.

Historical separate review: source PASS, package BLOCKED, RISK_FLAG YES at
`dbd7df5b905f2f79d4c56532c65bedd987899cc1`; it does not review this delivery.
Raw proof whitespace and cached-snapshot/DOB limitations are preserved.
See REPORT.md and evidence/current-finish-20261025/origins.json in this package.

Separate final package verifier /blast-radius is pending. No push or merge is
requested by this builder. Human acceptance/publication follows that review;
merge requires actual hosted CI for the exact final head.
