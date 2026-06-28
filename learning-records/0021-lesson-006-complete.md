# Lesson 006 complete — missingness taxonomy (MCAR / MAR / MNAR)

User completed Lesson 006 lab (EXIT TICKET submitted). Numbers match expected pattern on sklearn 1.9 venv.

**Evidence (EXIT TICKET):**
- true mean x1: −0.036
- MCAR observed: −0.034 (bias +0.002) — ~unbiased ✓
- MAR observed: −0.415 (bias −0.380) — biased ✓
- MNAR observed: −0.588 (bias −0.552) — most biased ✓
- CV plain impute: 0.721; impute+indicator: 0.737 (+0.016) ✓

**Takeaway submitted:** "MNAR biased mean, indicator helped to mitigate problem" — directionally correct; sharpen to include MAR bias and that the indicator preserves *was-missing* signal for prediction (does not restore the true mean).

**Skill secured:** classify mechanism by what missingness depends on; complete-case / naive mean fails under MAR/MNAR; `add_indicator=True` inside pipeline lets model use informative missingness under MNAR.

**Implications:** Lesson 007 unlocked. Reinforce: MAR vs MNAR is a domain judgement (untestable from data alone); relational gaps are usually structural/MNAR — fair GBDT baseline should not drop rows or impute away signal trees could split on.
