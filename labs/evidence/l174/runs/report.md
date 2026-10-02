# L174 measured temporal adaptation
**Measured: all twelve fits complete.** Lower macro task loss is better; every entry uses all 106,757 test targets.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| freeze | 1.232717 | 1.251662 | 1.260463 | 1.248281 ± 0.014178 |
| full | 1.236253 | 1.238285 | 1.268198 | 1.247579 ± 0.017886 |
| adapter | 1.222419 | 1.244116 | 1.254079 | 1.240205 ± 0.016188 |
| scratch | 1.160042 | 1.163677 | 1.150535 | 1.158085 ± 0.006786 |
| Unchanged source | 1.416096 | 1.405385 | 1.420837 | 1.414106 ± 0.007916 |

2006 training-only constant baseline: **1.175455**. Parameter counts: freeze 9,834; full 28,426; adapter 10,386 (including 552 new); scratch 28,426.
**Measured conclusion:** scratch has lower test macro loss than every pretrained adaptation arm for each of the three seeds. All pretrained adaptation arms improve on their unchanged source checkpoints, yet remain worse than the constant baseline. These runs do not establish a benefit from pretraining. Adapter minus freeze averages -0.008076, with paired differences -0.010298, -0.007546, -0.006384. Scratch minus full averages -0.089494. This is a descriptive result for this exposed F1 population and fixed schedule, not a universal ranking of adaptation methods.

## Every task, reported separately
Mean across three seeds. Raw MAE and accuracy have separate units; they are not pooled.

| Task | Metric | Freeze | Full | Adapter | Scratch | Constant |
|---|---|---:|---:|---:|---:|---:|
| constructor_results.points | mae_raw | 8.611796 | 8.616959 | 8.612052 | 8.608458 | 8.587734 |
| constructor_standings.points | mae_raw | 67.235057 | 63.569940 | 66.607196 | 80.112076 | 81.259583 |
| constructor_standings.position | mae_raw | 1.537240 | 1.349267 | 1.510189 | 2.034099 | 2.677758 |
| constructor_standings.wins | mae_raw | 2.601377 | 2.577220 | 2.550915 | 1.721579 | 1.398972 |
| qualifying.number | accuracy | 0.063396 | 0.063551 | 0.062309 | 0.047870 | 0.025462 |
| qualifying.position | mae_raw | 5.087656 | 5.240291 | 5.092996 | 5.111499 | 5.333876 |
| races.round | mae_raw | 5.020837 | 5.065826 | 5.023071 | 5.215719 | 5.041140 |
| races.year | mae_raw | 15.141475 | 9.049058 | 12.459120 | 9.270119 | 9.756330 |
| results.fastestLap | mae_raw | 12.494605 | 12.465494 | 12.491244 | 13.914970 | 15.036110 |
| results.grid | mae_raw | 4.578870 | 4.514946 | 4.576036 | 4.910453 | 5.390178 |
| results.laps | mae_raw | 10.921629 | 9.689124 | 10.890347 | 11.698908 | 13.532845 |
| results.milliseconds | mae_raw | 747186.729167 | 726688.104167 | 752188.958333 | 720408.750000 | 709182.250000 |
| results.number | accuracy | 0.044238 | 0.047435 | 0.044599 | 0.036453 | 0.025522 |
| results.points | mae_raw | 3.749411 | 3.651033 | 3.724510 | 4.075399 | 4.669570 |
| results.position | mae_raw | 3.299785 | 3.271086 | 3.278323 | 3.391782 | 4.593948 |
| results.positionOrder | mae_raw | 3.210472 | 3.188566 | 3.201896 | 3.934567 | 5.351818 |
| results.rank | mae_raw | 4.466011 | 4.481924 | 4.479092 | 4.611528 | 5.348620 |
| results.statusId | accuracy | 0.749214 | 0.750193 | 0.749368 | 0.681310 | 0.508121 |
| standings.points | mae_raw | 30.536767 | 29.175506 | 30.429408 | 38.682390 | 40.593746 |
| standings.position | mae_raw | 3.542594 | 2.679947 | 3.570794 | 3.635358 | 5.651200 |
| standings.wins | mae_raw | 1.538560 | 1.486762 | 1.528185 | 0.749066 | 0.734807 |

## Per-task losses
| Task | Freeze | Full | Adapter | Scratch | Constant |
|---|---:|---:|---:|---:|---:|
| constructor_results.points | 2.070600 | 2.075592 | 2.070784 | 2.063810 | 2.032302 |
| constructor_standings.points | 2.276229 | 2.169273 | 2.254245 | 2.761086 | 2.779398 |
| constructor_standings.position | 0.081745 | 0.065620 | 0.079833 | 0.182698 | 0.220271 |
| constructor_standings.wins | 1.466771 | 1.457275 | 1.433619 | 0.836836 | 0.529796 |
| qualifying.number | 5.072314 | 5.590886 | 5.110741 | 3.698611 | 3.902007 |
| qualifying.position | 0.399471 | 0.419701 | 0.400248 | 0.384531 | 0.408108 |
| races.round | 0.705367 | 0.715562 | 0.705890 | 0.749712 | 0.709262 |
| races.year | 0.602912 | 0.235394 | 0.449082 | 0.240496 | 0.257495 |
| results.fastestLap | 0.340447 | 0.339264 | 0.339982 | 0.394908 | 0.445506 |
| results.grid | 0.260220 | 0.254331 | 0.259912 | 0.281320 | 0.327659 |
| results.laps | 0.087759 | 0.070101 | 0.086964 | 0.126607 | 0.148470 |
| results.milliseconds | 0.156298 | 0.151245 | 0.157816 | 0.151173 | 0.148603 |
| results.number | 5.329380 | 5.503766 | 5.374130 | 4.735896 | 4.100043 |
| results.points | 1.341415 | 1.303684 | 1.329438 | 1.449648 | 1.611252 |
| results.position | 0.397441 | 0.394646 | 0.393545 | 0.418278 | 0.661057 |
| results.positionOrder | 0.115942 | 0.116206 | 0.116115 | 0.163247 | 0.285799 |
| results.rank | 0.417789 | 0.427385 | 0.422932 | 0.427711 | 0.540141 |
| results.statusId | 1.292504 | 1.252102 | 1.273557 | 1.823679 | 2.137475 |
| standings.points | 2.136575 | 2.046574 | 2.129665 | 2.736846 | 2.864705 |
| standings.position | 0.035031 | 0.020887 | 0.035042 | 0.050349 | 0.069611 |
| standings.wins | 1.627686 | 1.589658 | 1.620763 | 0.642336 | 0.505590 |

## Measured fit runtime
Wall seconds include checkpoint writes, validation and selected test inference. These are operational observations, not deterministic metrics.
| Arm | Seed0 | Seed1 | Seed2 |
|---|---:|---:|---:|
| freeze | 2.309 | 1.455 | 1.517 |
| full | 1.547 | 1.728 | 1.692 |
| adapter | 1.548 | 1.522 | 1.451 |
| scratch | 1.588 | 1.664 | 1.547 |

Raw per-seed losses, metrics, trainability, selection and histories are in report.json. Every epoch is saved under its arm/seed directory.
Whole-paper reproduction NOT_RUN; test exposure RETROSPECTIVE_L173; learner PENDING_WRITTEN_DEFENSE.
