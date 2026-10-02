# L172 Full F1 Schema-Tokenization Audit

| Table | Rows transformed | Columns | Rows admitted to fit |
|---|---:|---:|---:|
| circuits | 77 | 8 | 0 |
| constructor_results | 12,290 | 5 | 8,481 |
| constructor_standings | 13,051 | 7 | 9,229 |
| constructors | 211 | 4 | 0 |
| drivers | 857 | 7 | 0 |
| qualifying | 9,815 | 7 | 2,228 |
| races | 1,101 | 7 | 731 |
| results | 26,080 | 15 | 18,469 |
| standings | 34,124 | 7 | 26,098 |

**Complete: 97,606 rows, 67 columns, 866,746 cells.** All-table column reordering, all-cell mask erasure and every numerical/category held-out fit intervention pass.

Complete selected course pipeline audit. Whole-paper reproduction/fresh pretraining NOT_RUN; historical availability NOT_ESTABLISHED. See report.json for per-column state/fit counts and output hashes.
