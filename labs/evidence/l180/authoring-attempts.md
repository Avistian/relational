# Retained authoring and verification observations

- Initial behavior check deliberately failed with NotImplementedError before implementation. Runtime is included in local-budget.json.
- Initial full replay completed in83.587382seconds. Repeated lazy NPZ decompression caused unnecessary CPU work. An attempted stop arrived after completion. Materializing each archive once preserves the exact report and accelerates subsequent checks.
- Concurrent verification was refused by the budget wrapper before execution (active reservation). The replay completed normally; no reservation was reset or bypassed.
- First independent report comparison failed because AST-derived tuples differed from JSON lists. Values now normalize before returning; the saved JSON report is unchanged. The failed check remains metered.
- Figure review corrected the flow arrow into column attention and included UTC hours in the actual temporal witness (2011-03-27 00:00 versus06:00). Subsequent portable execution has exact report parity.
- Generic manifest regeneration would reset the version and change old curated titles. Those unrelated changes were removed; existing staged entries preserved, L180added,version33and gallery metadata matched.
- Original Cargo.lock included in the final source archive/hash ledger. Notebook execution follows final pinning. A documentation write initially used a duplicated relational/ path from inside the repository and failed; the corrected paths are used here.
- All numerical attempts are metered. No cloud job, fresh sampler, checkpoint download, gradient probe or model inference was launched.

- A final negative test first exposed that GPU-only affordability could omit all-in cost. The gate now separately requires a finite all-in upper bound; missing totals remain ALL_IN_COST_UNKNOWN. Hypothetical browser admission assumes USD35total, never measured or authorized spending. Numerical checks and portable execution were repeated after this behavior change.
