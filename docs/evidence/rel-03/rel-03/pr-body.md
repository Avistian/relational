Default unittest discovery now runs all 13 existing static visual contracts, including `check_course_visuals`. A thin `load_tests` wrapper delegates to the four original modules, preserving their test logic and IDs. The scripts tree is unchanged from launch base `2dfc5817872fa2a0a7768c8951d3a8b5487d09aa` (PR31 merge, confirmed by the parent before dispatch).

- Tests-only RED: `55ea896b1cbc5920173a0bab468c89284b215126`; GREEN/source: `32fe8400d83c466ce0e41e4bb3f7740f7227e422`.
- An unchanged separate regression first failed because actual targeted discovery found zero cases, then passed after the wrapper. It compares all original IDs, executes them in a fresh process, rejects skips, and avoids recursive self-discovery.
- Configured `/verify-project`: PASS on clean source, 129 tests. Wrapper verbose discovery: all 13 original cases pass (course 2, lesson visuals 3, details 1, openings 7). Standalone canonical-loader check: one regression passes, no errors/skips.
- [Readable proof](proof.md), [original IDs](evidence/original-case-coverage.json), [RED](evidence/red.log), [GREEN](evidence/green.log), [verbose cases](evidence/wrapper-verbose.log), [source verification](evidence/source-verification-status.json).

Builder `builder:rel-03`, pane `w1:p5R`. Independent read-only `/blast-radius`, parent-supported canonical receipts, PR/hosted CI and merge are pending. No push or merge by builder. No canonical two-commit receipt or hosted success is claimed here. Exact delivery/current verification identity is in the final parent task handoff.
