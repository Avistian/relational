# L167 complete saved-evidence audit

| Model | Paper AUROC | Audited mean ± sample SD | Verdict |
|---|---:|---:|---|
| RDBPFN | 0.7219 | 0.721938 ± 0.019983 | CLOSE |
| RDBPFN_single | 0.6640 | 0.663990 ± 0.034177 | CLOSE |
| TabICLv1.1 | 0.7176 | 0.717568 ± 0.009770 | CLOSE |

30 original runs; 21,060 predictions; 702 unique test queries. 41 file hashes checked against frozen pins and original source/input/run manifests. Support indices independently regenerated.

RDB-PFN minus single-table: +.05794808, positive in 10/10 draws. RDB-PFN minus TabICL: +.00436930, positive in 6/10 draws. One task only.

New L167 inference, pretraining and whole paper NOT_RUN; historical identity NOT_ESTABLISHED. Additional paid compute USD 0.
