# L171 RelBench Corpus and Holdout Audit

Approved 2026-10-02. Complete selected scope: seven original RelBench source definitions plus every table/row/FK of one pinned complete rel-f1 snapshot. Source inventory and observed rows remain distinct. This is a corpus-engineering experiment, not a paper's model-performance reproduction.

## Frozen protocol

- RelBench 1.1.0 installed source bytes: seven database modules, registry, hashes and base loader/table/database semantics. SHA256 per file in `evidence/l171/input-manifest.json`; source URLs and retrieval date in `sources/l171/source-ledger.json`.
- Seven databases: rel-amazon, rel-avito, rel-event, rel-f1, rel-hm, rel-stack, rel-trial. Extract table/PK/FK/time declarations without executing upstream code. The expected original seven source modules declare 50 tables and 62 FK columns. Newer RelBench expansions are outside this fixed scope.
- Complete F1 db.zip SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`, matching installed registry. Every extracted parquet must match its archive member and source-declared schema. No rows sampled or silently filtered.
- Corpus manifest has one curator-declared source family per original database. This is NOT an exhaustive lineage certification. Identical archive hashes or shared family identifiers create an undirected link; exclude the entire connected component of the held-out database, with relatives in quarantine. Execute all seven holdouts. Additional copy/bridge/tail records are explicitly synthetic contamination tests.
- Full PK audit counts nulls and repeated non-null keys beyond the first. FK audit counts nulls separately from non-null orphans. Audit all 13 declared F1 FK columns. Null FK alone is not failure; absent target tables or keys invalidate the contract.
- Time counts are descriptive snapshot windows: `<2005-01-01`, `[2005-01-01,2010-01-01)`, `>=2010-01-01`, nulls. These are not task-table splits or query-time training filters. No availability times are fabricated for the three static tables.
- $0 new cloud/API. Aggregate 1,800 seconds for local audits, independent numerical checks, failed attempts and portable solution execution. `local-budget.json` records every metered attempt, including the preliminary parquet inspection. Authoring/rendering/browser/source retrieval are delivery work. At cap: stop with INCOMPLETE; do not shrink scope. No paid dispatch or model training.

## Reproduce from the repository root

Author environment: Python3.12, pandas3.0.3, pyarrow24.0.0. Notebook packages and full runtime receipt are in `_execution_l171_results.json`. No RelBench import is needed for replay. Use existing requirements-labs.txt for notebook/plot/browser tools.

```bash
.venv/bin/python labs/_budget_l171.py .venv/bin/python labs/_audit_l171.py
.venv/bin/python labs/_budget_l171.py .venv/bin/python labs/_verify_l171.py
.venv/bin/python labs/_figures_l171.py
.venv/bin/python labs/_build_l171.py
.venv/bin/python labs/_budget_l171.py .venv/bin/python labs/_execute_l171.py
.venv/bin/python labs/_delivery_l171.py
```

The delivery command checks desktop/mobile controls, notebook portability, canonical source, copied-site links, manifest galleries and deterministic regeneration. `_checkout_l171.py` checks the actual Pages build from the Git index after intended files are staged. `_prepare_l171.py` is author-side source preparation requiring the exact installed RelBench release and original cached F1 archive; replay uses already pinned files and does not need that cache.

The student and solution notebooks embed every pinned input. Run in a fresh directory with pandas and pyarrow available; there is no network or cloud call. Solution execution must produce a report identical to the author report. The student version leaves three live functions unfinished. The written corpus card remains blank even in the reference solution.

## Result and limits

- Source inventory COMPLETE: 7 databases, 50 declared tables, 62 declared FK columns.
- F1 snapshot audit COMPLETE: 9 tables, 97,606 rows, 13 FK columns and 227,716 non-null references. No PK nulls/duplicates, null FKs or dangling FKs in this archive.
- Seven clean splits: six training candidates each. The synthetic F1 contamination intervention quarantines all three relatives, including the two-hop tail.
- Historical availability NOT_ESTABLISHED; 1,145 rows in three tables have no time column. A later pretraining pipeline must separately constrain feature and label availability.
- Other six row audits NOT_RUN. Undisclosed lineage and historical paper byte identity NOT_ESTABLISHED. Candidate selection is not a rights decision; data rights REVIEW_REQUIRED.
- Fresh pretraining/whole paper NOT_RUN; transfer NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Browser, live Colab and deployment are independent delivery states; only actual receipts establish performed checks.
