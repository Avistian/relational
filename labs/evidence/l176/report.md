# L176 measured nested-support experiment

| Task / frozen model | 64 | 128 | 256 | 512 | 1,024 |
|---|---:|---:|---:|---:|---:|
| rel-f1/RDBPFN | 0.7066 | 0.7220 | 0.7256 | 0.7192 | 0.7188 |
| rel-f1/RDBPFN_single | 0.6030 | 0.6593 | 0.6579 | 0.6929 | 0.7086 |
| rel-f1/TabICLv1.1 | 0.6999 | 0.7250 | 0.7229 | 0.7202 | 0.7173 |
| rel-trial/RDBPFN | 0.5426 | 0.5677 | 0.5746 | 0.5988 | 0.6159 |
| rel-trial/RDBPFN_single | 0.5127 | 0.5380 | 0.5707 | 0.5843 | 0.5963 |
| rel-trial/TabICLv1.1 | 0.5272 | 0.5610 | 0.5797 | 0.5947 | 0.6087 |

Measured mean AUROC over ten support seeds; every cell contains fresh L176 inference. Sample SD and all paired gains are in the notebook and explorer.

**Observed negative mean doubling gains:** rel-f1/RDBPFN: 256→512 (-0.00634); rel-f1/RDBPFN: 512→1024 (-0.00048); rel-f1/RDBPFN_single: 128→256 (-0.00138); rel-f1/TabICLv1.1: 128→256 (-0.00208); rel-f1/TabICLv1.1: 256→512 (-0.00276); rel-f1/TabICLv1.1: 512→1024 (-0.00284). **21 of 24 adjacent-size contrasts contain both positive and negative seed gains.** These observations describe two fixed tasks and do not establish a universal effect.

**Executed:** all 300 fresh nested evaluations and all 300 saved published-protocol evaluations independently rescored: **458,100 probabilities across the two tracks**. These are repeated predictions on the same two test populations, not 458,100 independent examples. Fresh pretraining and whole-paper reproduction remain `NOT_RUN`; learner status is `PENDING_WRITTEN_DEFENSE`.
