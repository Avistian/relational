# Approved Lesson 129 design

User approved 2026-09-27: manual feature engineering lesson plus complete selected F1 driver-position user-study replay. Teach exploration, feature hypotheses, temporal SQL, tuned LightGBM, independent scoring, and honest human-effort accounting. Deliver standalone HTML, portable student/solution notebooks, reference, effort log, provenance/protocol ledger, executed experiment and delivery checks. Preserve unrelated work; no publication requested.

Target is released manual-feature user study, paper section 6/Figure 3, NOT the raw-entity LightGBM baseline in Table 7. Pin user-study commit 445bb7a3b1230f49f8e5890ae81754d3e365680f. Recover complete SQL, stypes, preprocessing, 10-trial tuning, and original task keys. Exact historical runtime/seed/score identity must be established separately. Replaying automation cannot recreate human work or establish learner mastery.

Budget: USD10 aggregate; provisional 8 worker-hours at 4 physical cores and16GiB: 8*3600*(4*.0000131+16*.00000222)=USD2.532096, with USD7.467904 reserve. Pilot before full execution, stop if required scope exceeds cap; no automatic retries. Prefer isolated local CPU execution when feasible, no paid compute then. Rates verified https://modal.com/pricing on2026-09-27.

Implementation plan: (1) pin/review sources and task/data lineage; (2) write failing semantic checks for cutoff, keyed predictions, validation-only selection; (3) implement visible functions, full SQL and trainer; (4) independently audit SQL and source parity, execute bounded pilot then complete tuning/runs; (5) collect every artifact and independently score, compare compatible L127 evidence; (6) author causal lesson, computation diagrams and intervention; (7) embed full sources/data/evidence in notebooks, execute solution without repo; (8) verify source identity, exercises, browser/mobile/no-JS/print, deterministic builds and copied Pages.

Writing-plans skill was searched for in available workspace/global skill roots and is absent. This document supplies its implementation-plan function.
