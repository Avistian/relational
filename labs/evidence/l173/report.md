# L173 measured course pretraining
**Complete:** 370,024 targets across 21 tasks. Train / validation / test: 244,410 / 6,354 / 119,260.

| Loss weighting | Seed 0 test | Seed 1 test | Seed 2 test | Mean ± sample SD |
|---|---:|---:|---:|---:|
| cell | 1.338310 | 1.327036 | 1.341849 | 1.335732 ± 0.007736 |
| task | 1.288805 | 1.304340 | 1.348481 | 1.313875 ± 0.030960 |

All entries are **test macro task loss** (lower is better), even for the cell-weighted training arm. Training-only constant baseline: **1.276529**. Each run scores all 119,260 test targets.
**Measured interpretation:** equal-task weighting changes test macro loss by -0.021857 on average (task minus cell). The three paired changes are -0.049505, -0.022696, +0.006632. One seed reverses the direction. **Every trained run has higher test macro loss than the constant baseline.** This short pretraining schedule has not established an aggregate predictive benefit, even against that simple comparator. This supports a description of these six runs, not a general claim that equal-task weighting is better. Test losses also exceed validation losses substantially; do not present the validation curve as future-period performance.

## Per-task results
Baseline is training mean or Laplace-smoothed training class frequencies. Loss is the task-specific criterion. MAE and accuracy are kept separate.

| Task | Train / valid / test | Baseline loss | Cell mean loss | Task mean loss | Metric | Cell / task |
|---|---:|---:|---:|---:|---|---:|
| constructor_results.points | 8481 / 188 / 3621 | 1.940260 | 1.991326 | 1.970660 | mae_raw | 8.162813 / 8.135114 |
| constructor_standings.points | 9229 / 190 / 3632 | 2.726901 | 2.233171 | 2.235344 | mae_raw | 65.842018 / 66.109945 |
| constructor_standings.position | 9229 / 190 / 3632 | 0.299819 | 0.171727 | 0.144594 | mae_raw | 2.276622 / 2.041773 |
| constructor_standings.wins | 9229 / 190 / 3632 | 0.492420 | 2.441439 | 2.568914 | mae_raw | 4.131764 / 4.351093 |
| qualifying.number | 2228 / 376 / 7211 | 3.988189 | 4.460228 | 4.327162 | accuracy | 0.062266 / 0.055748 |
| qualifying.position | 2228 / 376 / 7211 | 0.412378 | 0.389464 | 0.413637 | mae_raw | 5.005698 / 5.189544 |
| races.round | 731 / 19 / 351 | 0.771486 | 0.698964 | 0.734255 | mae_raw | 4.986097 / 5.143593 |
| races.year | 731 / 19 / 351 | 1.771860 | 1.517428 | 1.534869 | mae_raw | 30.278468 / 30.540539 |
| results.fastestLap | 347 / 354 / 6914 | 0.601205 | 0.457381 | 0.423036 | mae_raw | 15.214572 / 14.447728 |
| results.grid | 18469 / 376 / 7235 | 0.327401 | 0.313062 | 0.295263 | mae_raw | 4.987181 / 4.834212 |
| results.laps | 18469 / 376 / 7235 | 0.196520 | 0.114466 | 0.141262 | mae_raw | 12.240486 / 13.642812 |
| results.milliseconds | 3519 / 150 / 3581 | 0.191820 | 0.197369 | 0.145382 | mae_raw | 913164.625000 / 722624.458333 |
| results.number | 18463 / 376 / 7235 | 4.010845 | 4.590885 | 4.359237 | accuracy | 0.045059 / 0.042064 |
| results.points | 18469 / 376 / 7235 | 1.508147 | 1.298362 | 1.240203 | mae_raw | 3.651986 / 3.575987 |
| results.position | 9009 / 279 / 5919 | 0.703005 | 0.474200 | 0.468412 | mae_raw | 3.662590 / 3.613334 |
| results.positionOrder | 18469 / 376 / 7235 | 0.321173 | 0.177266 | 0.220257 | mae_raw | 3.973219 / 4.449274 |
| results.rank | 347 / 354 / 7130 | 0.536988 | 0.399751 | 0.397918 | mae_raw | 4.364815 / 4.338568 |
| results.statusId | 18469 / 376 / 7235 | 2.495315 | 1.431474 | 1.392287 | accuracy | 0.725132 / 0.729786 |
| standings.points | 26098 / 471 / 7555 | 2.798424 | 2.055749 | 2.144819 | mae_raw | 29.323666 / 30.348418 |
| standings.position | 26098 / 471 / 7555 | 0.251743 | 0.067693 | 0.060084 | mae_raw | 5.014585 / 4.915358 |
| standings.wins | 26098 / 471 / 7555 | 0.461206 | 2.568964 | 2.373787 | mae_raw | 2.297859 / 2.161573 |

[All 21 per-task validation curves](../../figures/l173/per-task-curves.svg) — mean and sample SD across seeds; separate loss scales.

## Coverage and source gaps
370,024 targets; 1,082,390 row context references; 696,384 dated parent references. Full exclusion ledger: coverage.json.

Unknown category targets: qualifying.number=950, results.number=94, results.statusId=138.

Whole-paper reproduction NOT_RUN. Cross-database transfer and historical availability NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.
