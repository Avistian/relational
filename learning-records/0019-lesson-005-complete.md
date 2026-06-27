# Lesson 005 complete — pipelines & preprocessing

User completed Lesson 005 ("Lab done"). Warm-up retrieval before reading was **3/3**: `best_score_` = inner-CV validation mean (A), scaler-fit-on-all-X leaks test statistics (B), PIT aggregates are preprocessing RDL aims to learn end-to-end (C). The prior `best_score_`-is-training misconception from L004 appears **closed** — it did not resurface.

**Evidence:** warm-up answers in chat (A/B/C); lab self-reported "Lab done."

**Skill secured:** wrap every `fit`-bearing transform inside a `Pipeline`/`ColumnTransformer` so it refits per fold on train rows only; tune via `step__param`; this composes with nested CV.

**Process change this session (see NOTES.md preferences):** user requested that, going forward, every lab ships as a Jupyter **notebook with blank fill-in cells**. Established a reusable pattern in `labs/` (`LAB-TEMPLATE.ipynb` + `labs/README.md`). L005's lab was already done as freeform; the notebook convention starts at L006.

**Implications:** Lesson 006 (missingness taxonomy MCAR/MAR/MNAR) unlocked and prepared. Carry the L005 pipeline discipline into L006 — imputers/indicators must live inside the pipeline.
