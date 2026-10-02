# L184 evidence

Selected GelGT reproduction **INCOMPLETE_SOURCE_TEMPORAL_GATE**. Fresh training **NOT_RUN**. Local source/mechanism audit and all **8,712 labels PASS**. Cloud/API **$0**. Learner **PENDING_WRITTEN_DEFENSE**.

| Split | Complete queries | Entity-only entries | Lost distinct entries |
|---|---:|---:|---:|
| train | 7,453 | 771 | 6,682 |
| val | 499 | 47 | 452 |
| test | 760 | 56 | 704 |

All labels reconstruct with maximum error **0**. Original attention: maximum forward error **4.44×10⁻¹⁶** and input-gradient error **2.15×10⁻¹⁵** in a float64, dropout-zero controlled test. Synthetic source probe: **2 queries → 1 cache entry**, admitting one future node to the early query.
