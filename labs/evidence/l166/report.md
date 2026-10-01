# L166 selected reproduction

| Model | Paper AUROC | Fresh mean ± sample SD | Result |
|---|---:|---:|---|
| RDBPFN | 0.7219 | 0.721938 ± 0.019983 | CLOSE |
| RDBPFN_single | 0.6640 | 0.663990 ± 0.034177 | CLOSE |
| TabICLv1.1 | 0.7176 | 0.717568 ± 0.009770 | CLOSE |

30 evaluations; 21,060 predictions; 702 unique test queries. All targets match at four decimals.

Paired RDBPFN minus single-table: 0.05794807983213781 AUROC. Paired RDBPFN minus TabICLv1.1: 0.004369296833064962 AUROC. Ten support draws, one task.

Released checkpoint replay COMPLETE; fresh pretraining / whole paper NOT_RUN; historical identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. See [full contract](../../l166-reproduction.md).
