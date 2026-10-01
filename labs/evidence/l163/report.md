# L163 Frozen Row Encoder Comparison

Executed synthetic regression task. MAE in synthetic target units; lower is better.

| Encoder | Input variant | Mean MAE | Split SD | Mean prediction change |
|---|---|---:|---:|---:|
| typed | baseline | 1.713803 | 0.151146 | 0.000000 |
| typed | reordered | 1.713803 | 0.151146 | 0.000000 |
| typed | renamed | 1.713803 | 0.151146 | 0.000000 |
| bart | baseline | 14.265465 | 1.285415 | 0.000000 |
| bart | reordered | 55.050553 | 34.406235 | 50.765346 |
| bart | renamed | 705.121763 | 81.465686 | 705.387827 |

| Split | Encoder | Input | Alpha | Test MAE | Train-mean baseline MAE |
|---:|---|---|---:|---:|---:|
| 0 | typed | baseline | 0.01 | 1.835880 | 95.064468 |
| 0 | typed | reordered | 0.01 | 1.835880 | 95.064468 |
| 0 | typed | renamed | 0.01 | 1.835880 | 95.064468 |
| 0 | bart | baseline | 0.01 | 15.691710 | 95.064468 |
| 0 | bart | reordered | 0.01 | 37.219269 | 95.064468 |
| 0 | bart | renamed | 0.01 | 626.908089 | 95.064468 |
| 1 | typed | baseline | 0.01 | 1.544746 | 92.753782 |
| 1 | typed | reordered | 0.01 | 1.544746 | 92.753782 |
| 1 | typed | renamed | 0.01 | 1.544746 | 92.753782 |
| 1 | bart | baseline | 1.0 | 13.196450 | 92.753782 |
| 1 | bart | reordered | 1.0 | 33.220090 | 92.753782 |
| 1 | bart | renamed | 1.0 | 789.490327 | 92.753782 |
| 2 | typed | baseline | 1.0 | 1.760782 | 88.377262 |
| 2 | typed | reordered | 1.0 | 1.760782 | 88.377262 |
| 2 | typed | renamed | 1.0 | 1.760782 | 88.377262 |
| 2 | bart | baseline | 1.0 | 13.908236 | 88.377262 |
| 2 | bart | reordered | 1.0 | 94.712299 | 88.377262 |
| 2 | bart | renamed | 1.0 | 698.966874 | 88.377262 |

Three overlapping splits are not independent datasets; SD is descriptive, not a confidence interval. The target is explicitly numeric/additive, favoring the typed linear path. No LM fine-tuning, natural text semantics, missing-data comparison, GNN or database transfer is tested.

All 864 keyed predictions are in report.json. Six heads were selected on validation before test evaluation. Reordered and renamed inputs use unchanged baseline heads. Historical reproduction NOT_RUN; fidelity and general superiority NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.
