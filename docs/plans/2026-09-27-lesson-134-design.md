# Lesson 134 — Training at scale

Approved by user 2026-09-27: complete five-seed ten-epoch F1 Table7 reproduction plus bounded full-graph rel-stack training/sampling measurements, aggregate USD10 maximum.

Learning outcome: explain and measure how temporal query ownership, relation-specific fanouts and seed batch size determine actual sampled computation. L133 defines the layer; L134 constructs and budgets its input.

Deliver HTML, reference, portable student/solution notebooks, three live tasks (relation-aware expansion bound, query-cutoff audit, aggregate throughput), a mini-batch computation diagram, measured plots, interactive fanout calculation, exact operators and source/protocol ledgers. Source pin and preserved published target inherit L133; all L134 training is fresh.

F1: unchanged pinned model/trainer, full released graph/task, seeds0–4, ten epochs, first validation minimum, independent scores and original-model parity, audit every sampled batch. Descriptive tolerance0.2 MAE, not equivalence or historical identity.

Scale: full released rel-stack structure and real user-engagement queries; deliberately reduced features and width32 predictor. Fixed query prefix, bounded batches, frozen configurations, synchronised sample/transfer/step timing, warmups reported separately, peak allocated/reserved CUDA memory, independent graph census and time/query checks. No full rel-stack paper result claim.

Budget: USD2 F1 and validation, USD5 scale, USD3 preprocessing/retries/overhead. Reserve worst-case resources before each worker; use no automatic retries and refuse cumulative commitments above USD10. T4 .000164/s, CPU .0000131/core/s, memory .00000222/GiB/s (Modal pricing checked2026-09-27). Pilot gate precedes five-seed dispatch. Scale worker2CPU/32GiB roughlyUSD.940464/hour. No paid task on local WSL.

Implementation sequence: (1) contract tests and visible primitives; (2) source/budget guard and F1 pilot; (3) scale operator and bounded dispatch; (4) collect all artifacts, independent checks; (5) narrative/figures/notebooks; (6) standalone execution, desktop/mobile/keyboard/noJS/print/copied-Pages delivery. Preserve unrelated workspace edits and learner mastery.

The brainstorming skill's referenced writing-plans skill is unavailable in the installed skill directories; this document supplies the implementation plan directly. No delegation requested.
