# Lesson 009 complete — feature engineering

User completed Lesson 009 lab ("lab done"). Warm-up retrieval before reading was **3/3** (target encoding inside Pipeline per-fold A; PR baseline = prevalence A; PIT aggregate = child rows before cutoff A) — strong cross-lesson retention (L005 pipeline, L008 metric, L002 point-in-time all retrieved cold).

**Skill secured:** engineer the features a model cannot synthesize itself (ratio of differences, cyclical datetime) and keep fit-bearing features (target encoding) inside the pipeline so they never leak. Distinguishes stateless (within-row) from fit-bearing (across-row) transforms — the boundary that decides what must go in the Pipeline.

**Evidence:** warm-up 3/3 in chat; lab self-reported "lab done." (Lab verified-runnable end-to-end: ratio R² 0.644→0.998, cyclical 0.840→0.979, target-encoding leak 0.76 vs 0.50 honest.)

**Implications:** Q1 single-table-mechanics arc (001–009) complete. Lesson 010 is the **Q1 checkpoint** — assemble the whole leakage spine into one reproducible baseline. After a long 3/3 streak (L007/L008/L009), the checkpoint is the real test: can the user wire all nine concepts together unaided, not just recall them.
