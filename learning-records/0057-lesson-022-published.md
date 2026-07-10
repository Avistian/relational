# Lesson 022 published — Label leakage patterns (Q3)

**Lesson 022** (curriculum lec 022, Year 1 Q3 · "Evaluation rigor & benchmark literacy"). The
synthesizing leakage lesson: it names the whole family of leaks the learner has been fighting since
L002 and adds the ones they had not met.

**Single skill:** classify a feature-engineering choice into Kapoor & Narayanan's leak taxonomy
("spot leakage in FE"), and recognize that an **illegitimate feature** can manufacture a false
"complex model beats the simple one" result — the **reproducibility collapse**.

**Primary reading.** Kapoor & Narayanan 2022, *Leakage and the Reproducibility Crisis in ML-based
Science* ([arXiv:2207.07048](https://arxiv.org/abs/2207.07048)) — abstract + §2 (taxonomy of 8 leak
types in 3 families), §5 (civil-war reproduction), §6 (model info sheet). **arxiv MCP unavailable this
session** (only `cursor-cloud` MCP present) → verified the abstract by fetching arxiv.org directly.
Key facts used: leakage found across **17 fields / 329 papers**; taxonomy of **8 types**; civil-war
study where every "complex ML beats logistic regression" claim failed to reproduce once leakage was
removed; the **model info sheet** as the prescribed defence.

**Taxonomy taught (the map).** Family 1 (no clean train/test separation): L1.1 no test set, L1.2
preprocessing on all data, L1.3 feature selection on all data, L1.4 duplicates. Family 2: L2
illegitimate feature. Family 3 (test ≠ distribution of interest): L3.1 temporal, L3.2
non-independence, L3.3 sampling bias. Most of Q1–Q2 discipline is Family 1; L021 was L3.1; the new
sharp edge is **L2**.

**Verified live (`labs/_verify_l022.py` + executed solution, sklearn 1.9.0, seed 0):**
- **Reproducibility collapse (L2):** weak honest signal (RF/LR ~0.72 tie). Add one *non-monotone*
  illegitimate column (a tree carves it with 2 splits; a linear model sees class means ~0.5, no
  usable direction) → **RF 0.935 vs LR 0.719 (gap +0.217)**; remove it → **RF 0.712 vs LR 0.721
  (gap −0.009)**. Faithful miniature of the civil-war finding.
- **Near-duplicate leak (L1.4 / L3.2):** naïve random 5-fold CV **0.948** vs `GroupKFold` on record id
  **0.876** (leak **+0.071**). The L004 grouped-split fix, now recognized as a named leak type.
- **Design note:** first tried an ID-encoded and an XOR leak; both failed (StandardScaler made the ID
  monotone → helped LR; greedy trees don't reliably discover XOR among many features). The
  non-monotone banded proxy is the clean mechanism that trees exploit and linear models cannot.

**Two new reusable viz (standard #9 — one viz per distinct mechanism):**
- `assets/leakage-taxonomy-viz.js` — interactive map of the 8 types in 3 families; clicking a chip
  shows definition + *where the learner already met it* (spacing/interleaving) + the fix. The
  load-bearing structural visual (turns 8 scattered leaks into one ontology).
- `assets/repro-collapse-viz.js` — leak ON/OFF bar toggle (RF vs LR): 0.935/0.719 with the leak,
  0.712/0.721 without — the collapse made visible.
Both headless-verified `labs/_viz_check_l022.js` 12/12. **Browser MCP still unavailable** → headless
only, per the standing gap.

**Pedagogy (all Q2-retrospective standards applied):** spaced-retrieval warm-up (`upTo: 22`);
prediction-before-reveal on the collapse (does the complex model still win after the leak is removed?);
teach-back on *why* a leaked feature inflates the complex-vs-simple gap; a `Checklist` mount rendering
the **model info sheet** (7 leak-type questions); 3 quizzes. Fed the artifacts:
- retrieval-pool +2 (`l022-illegit` [misconception], `l022-collapse`).
- paper-deck +1 (`kapoor2022`).
- misconceptions **M19** (Q3): "a very predictive feature / big complex-vs-simple win is good news"
  → leak hypothesis first; audit provenance.
- thesis-dossier +1 (BAR, C3) + a new skeptic objection ("maybe RDL wins are leakage too") with our
  structural answer (RelBench point-in-time cutoffs + mandatory leakage audit).
- `reference/glossary.html` +5 Q3 terms (data leakage, illegitimate feature L2, non-independence /
  duplicates, reproducibility collapse, model info sheet).
`node labs/_check_pedagogy.js` clean.

**Lab:** `labs/0022-label-leakage-patterns.ipynb` — Tier-C, self-contained (no relkit; concept lab,
same call as L019/L021). Built via `labs/_build_l022.py` (emits blank student + filled solution).
3 TODO (crucial fragment each) + stretch: T1 = build `Xleak` + measure the collapse; T2 = the
`GroupKFold` grouped split vs naïve `KFold`; T3 = classify 6 FE snippets into taxonomy codes (the
core skill), auto-checked against a key. Student blank (15 `____`, 0 outputs); solution executed clean
(all CHECK + EXIT) and gitignored. Manifest → 22 entries (quarter 3); all labs re-rendered to
`labs/html/`.

**Env note:** no `python3-venv`/uv preinstalled again this session; installed `uv` via curl and built
`.venv` (sklearn 1.9.0, numpy 2.5.1, pandas 3.0.3). Lean install (numpy/pandas/scikit-learn/ipykernel/
nbconvert only — this lab needs no boosters). Env-setup agent should preinstall the lab venv + deps so
future sessions skip this.

**Thesis bridge:** the sharpest BAR yet — a big relational-vs-GBDT margin is a *leak hypothesis first*,
not evidence. It also arms the skeptic ("maybe RDL wins are leakage"), and the answer is structural:
RelBench enforces point-in-time cutoffs by construction (Fey 2024) and every gain must ship a leakage
audit (model info sheet). The thesis is won with a clean win over a fair baseline, never with a big
number.

Next: Lesson 023 (statistical comparison — Demšar 2006; paired tests on CV folds), continuing Q3.
