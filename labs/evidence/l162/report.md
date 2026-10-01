# L162 Relational Context and Scale Audit

Complete synthetic computation; no model training or measured transfer.

| Graph intervention | Reachable rows | Tables |
|---|---|---|
| h0-cut0-restricted0 | m1 | moons |
| h0-cut0-restricted1 | m1 | moons |
| h0-cut1-restricted0 | m1 | moons |
| h0-cut1-restricted1 | m1 | moons |
| h1-cut0-restricted0 | m1, p1 | moons, planets |
| h1-cut0-restricted1 | m1 | moons |
| h1-cut1-restricted0 | m1, p1 | moons, planets |
| h1-cut1-restricted1 | m1 | moons |
| h2-cut0-restricted0 | m1, m2, p1, s1 | moons, planets, stars |
| h2-cut0-restricted1 | m1 | moons |
| h2-cut1-restricted0 | m1, m2, p1 | moons, planets |
| h2-cut1-restricted1 | m1 | moons |
| h3-cut0-restricted0 | m1, m2, p1, s1 | moons, planets, stars |
| h3-cut0-restricted1 | m1 | moons |
| h3-cut1-restricted0 | m1, m2, p1 | moons, planets |
| h3-cut1-restricted1 | m1 | moons |

| Token case | Total | Dropped | Whole-table pairs | Row pairs | Truncated row pairs |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 0 | 64 | 34 | 34 |
| 1 | 8 | 0 | 64 | 34 | 34 |
| 2 | 800 | 0 | 640000 | 6400 | 6400 |
| 3 | 800 | 0 | 640000 | 6400 | 6400 |
| 4 | 1140 | 1116 | 1299600 | 1210808 | 192 |
| 5 | 1140 | 76 | 1299600 | 1210808 | 1049384 |
| 6 | 1 | 0 | 1 | 1 | 1 |
| 7 | 1 | 0 | 1 | 1 | 1 |

Evidence grid: {'RECONSTRUCTION_ONLY': 32, 'MULTITABLE_ONLY': 31, 'TRANSFER_REVIEW_ELIGIBLE': 1}.

All 64 evidence declarations still return transfer NOT_ESTABLISHED. Review eligibility never authenticates evidence.

Historical reproduction NOT_RUN; fidelity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.
