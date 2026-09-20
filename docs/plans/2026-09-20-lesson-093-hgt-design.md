# Approved design: Lesson 093 HGT

User approved the combined lesson, lab and reproduction package on 2026-09-20.

Teach a single skill: trace and implement a type-conditioned attention message from a timestamped source to a target. Bridge R-GCN relation transforms and HAN fixed routes to learned multi-hop composition. Include cold retrieval, a numerical attention intervention, temporal encoding, HGSampling, and a written defense.

Deliver canonical markdown/HTML, portable student/solution notebooks, visible full model and trainer, reference sheet, figures, manifest navigation and an evidence ledger. Three live tasks: receiver softmax, relation attention/message, and relative temporal basis. A same-graph HGT/R-GCN comparison is teaching evidence only.

Named published target: Hu et al. WWW2020 Table2, CS Paper-Field L2: NDCG .403±.041 and MRR .439±.078, five model initializations. Preserve the distinction between paper settings (width256, three layers, validation loss) and release defaults (width400, four layers, validation NDCG). Pin release85eaccd482bc1d1af56c2de297b6e3a88b96d5cd. Full benchmark reproduction is NOT_ESTABLISHED until data, preprocessing, protocol and execution are aligned; a passing operator oracle is insufficient.

Implementation sequence: (1) audit/download source/data; (2) behavioral tests before operators; (3) visible model, loaders, sampler, training and replay entrypoints; (4) run checks and affordable real-data comparison; (5) author explanation/figures/notebooks; (6) execute solution, check desktop/mobile and copied Pages build. Record every unresolved blocker without substituting scores.

Full OAG work is contingent on obtaining bytes and sufficient runtime. Source archive and a complete executable recipe must remain reviewable even if the named run cannot execute. No learner completion or deployment is inferred.
