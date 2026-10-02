# L171 RelBench Corpus and Holdout Audit

| Database | Declared tables | FK columns | Row evidence |
|---|---:|---:|---|
| rel-amazon | 3 | 2 | Source inventory only |
| rel-avito | 8 | 11 | Source inventory only |
| rel-event | 5 | 7 | Source inventory only |
| rel-f1 | 9 | 13 | Complete F1 snapshot |
| rel-hm | 3 | 2 | Source inventory only |
| rel-stack | 7 | 12 | Source inventory only |
| rel-trial | 15 | 15 | Source inventory only |

| Table | Rows | PK nulls / duplicate excess | Time column |
|---|---:|---:|---|
| circuits | 77 | 0 / 0 | None |
| constructor_results | 12,290 | 0 / 0 | date |
| constructor_standings | 13,051 | 0 / 0 | date |
| constructors | 211 | 0 / 0 | None |
| drivers | 857 | 0 / 0 | None |
| qualifying | 9,815 | 0 / 0 | date |
| races | 1,101 | 0 / 0 | date |
| results | 26,080 | 0 / 0 | date |
| standings | 34,124 | 0 / 0 | date |

Across 227,716 non-null FK references: **0 dangling, 0 null FK values**. This is measured archive integrity; historical availability remains unestablished.

Complete declared audit. Other six row audits and fresh pretraining NOT_RUN; availability/transfer NOT_ESTABLISHED; source-family exclusion covers declared lineage only.
