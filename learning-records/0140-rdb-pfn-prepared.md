# RDB-PFN lesson prepared · 2026-10-01

Author preparation, not learner mastery. Bridge L165's context adaptation to synthetic pretraining: schema → keys/latent states → cells → DFS → alternating feature/support-row attention → query probabilities. Do not assume L165b has been studied.

Complete approved v5 Table 9 driver-dnf 512-context experiment: RDBPFN, single-table checkpoint, TabICLv1.1 (32 estimators), seeds 0–9. Means .721938/.663990/.717568 match all paper targets at four decimals. All21,060 predictions cover702unique test queries; paired gains do not establish general superiority. Released labels complement current raw SQL DNF; retain this semantic boundary. Historical preprocessing identity NOT_ESTABLISHED; fresh pretraining and whole paper NOT_RUN.

Visible model passes both actual checkpoint output/gradient and query-isolation checks. Original prior draw:14 tables, 13 relationships, 14 tasks; all 16,547 FK cells resolve. LayerDAG schema pool reused. Portable course mechanisms are separate simplified examples. Three learner functions feed live checks and inference; written defense PENDING_WRITTEN_DEFENSE.

Retrieval next session: why identical DFS inputs matter; which attention axis crosses rows; why query label padding is legal but query labels are not; why a rounded score match does not reproduce the full training lineage. Year4 exit requirements remain unchanged.
