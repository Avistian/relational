# L168 selected cross-database evaluation

| Database / evidence | Model | Paper AUROC | Measured mean ± sample SD |
|---|---|---:|---:|
| rel-f1 / REUSED_L166 | RDBPFN | 0.7219 | 0.721938 ± 0.019983 |
| rel-f1 / REUSED_L166 | RDBPFN_single | 0.6640 | 0.663990 ± 0.034177 |
| rel-f1 / REUSED_L166 | TabICLv1.1 | 0.7176 | 0.717568 ± 0.009770 |
| rel-trial / FRESH_L168 | RDBPFN | 0.5986 | 0.598556 ± 0.021570 |
| rel-trial / FRESH_L168 | RDBPFN_single | 0.5961 | 0.596101 ± 0.021897 |
| rel-trial / FRESH_L168 | TabICLv1.1 | 0.5926 | 0.592820 ± 0.030357 |

RDB-PFN minus TabICL averages **+0.00437 AUROC on F1** and **+0.00574 on trial**, positive in 6/10 and 6/10 support draws respectively. The equal-database mean gain is **+0.00505**. All three fresh trial means are within the predeclared .02 descriptive distance from the published values; this is not a statistical equivalence test or proof of historical data identity.

The new evidence is 30 complete trial evaluations and **24,750 fresh probabilities**. The comparison also independently rescores 21,060 reused F1 probabilities. Both selected task means favor RDB-PFN over TabICL, but seed-level reversals and only two databases prevent a broad-superiority claim. The small trial gain over its single-table ablation also cautions against attributing every gain to the relational prior.

Fresh pretraining and whole paper NOT_RUN. Historical identity, checkpoint lineage and exact target-schema exclusion NOT_ESTABLISHED.
