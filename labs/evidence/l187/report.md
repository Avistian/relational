# L187 measured course experiment

**Author evidence:** complete `L187-F1-ENTITY-PRIVACY` course experiment. All 857 drivers, 2,571 histogram neighbors and 270 simulated releases checked. Published private-GNN reproduction **NOT_RUN**; production DP and complete erasure **NOT_ESTABLISHED**. Learner **PENDING_WRITTEN_DEFENSE**.

The full nine-table snapshot contains **70,876 declared owned records** and **175,933 edges incident to those owned sets**. Driver ID 3 has the largest contribution: **370 results + 353 qualifying + 372 standings + 1 driver = 1,096 records**. These IDs are snapshot identities, not names inferred by an attack.

| Cap C | ε | Kept / 26,080 | Clipping L1 | Noise MAE ± SD | Raw MAE ± SD |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 857 | 25,223 | 1.997 ± 0.124 | 119.977 ± 0.150 |
| 1 | 1 | 857 | 25,223 | 1.006 ± 0.058 | 119.720 ± 0.099 |
| 1 | 2 | 857 | 25,223 | 0.498 ± 0.044 | 119.587 ± 0.039 |
| 5 | 0.5 | 3,135 | 22,945 | 10.209 ± 0.662 | 113.344 ± 0.951 |
| 5 | 1 | 3,135 | 22,945 | 4.973 ± 0.337 | 110.636 ± 0.378 |
| 5 | 2 | 3,135 | 22,945 | 2.397 ± 0.164 | 109.623 ± 0.192 |
| 20 | 0.5 | 8,260 | 17,820 | 39.891 ± 2.555 | 110.678 ± 2.745 |
| 20 | 1 | 8,260 | 17,820 | 20.253 ± 1.415 | 96.531 ± 1.481 |
| 20 | 2 | 8,260 | 17,820 | 9.897 ± 0.651 | 89.770 ± 0.797 |
