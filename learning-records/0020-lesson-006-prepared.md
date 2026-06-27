# Lesson 006 prepared — missingness taxonomy (MCAR / MAR / MNAR)

Published `lessons/0006-missingness-taxonomy.html`, reference `reference/missingness-taxonomy.html`, reusable `assets/missingness-viz.js` (column-drop viz: toggle MCAR/MAR/MNAR; MAR rings the observed driver column, MNAR ghosts the would-be-high hidden values), and the first notebook lab `labs/0006-missingness.ipynb` (blank fill-in cells).

**Single skill:** given a missingness situation, classify it MCAR/MAR/MNAR by *what the missingness probability depends on* (nothing → MCAR; observed data → MAR; the unobserved value itself → MNAR), then choose handling — and know that complete-case deletion is unbiased only under MCAR, mean imputation distorts variance/relationships, and a **missing-indicator** preserves informative missingness.

**Grounding (high-trust, verified URLs):** van Buuren *FIMD* §1.2 + §2.2.4 (free online; the weighing-scale soft/hard-surface = MAR, scale-wears-out-over-time = MNAR examples come straight from the book); Rubin (1976) Biometrika origin of the taxonomy; sklearn §6.4 Imputation (`SimpleImputer(add_indicator=True)`, `MissingIndicator`, `IterativeImputer` behind `enable_iterative_imputer`). **Not run live** — no sklearn in this session; the notebook lab is the verification (user runs it).

**Thesis bridge:** relational data is saturated with *structural* missingness (a customer with zero orders, a null foreign key) which is usually MNAR-informative; GBDTs handle NaN natively via learned default split directions, while RDL encodes "missing" as an *absent edge / absent neighbor* — so a fair baseline must treat missingness deliberately (impute-inside-pipeline + indicator), never drop rows. Connects back to L002 structural-missingness note and L005 pipeline discipline.

**Exit:** notebook prints the three observed-vs-true means (MCAR≈unbiased, MAR/MNAR biased) + whether `add_indicator` recovered signal, plus one-sentence cause; or "lab done." Lesson 007 (target encoding & leakage, or scaling/encoding deep-dive) unlocks after.
