# Lesson 023 published — Statistical comparison of classifiers (Q3)

**Lesson 023** (curriculum lec 023, Year 1 Q3 · "Evaluation rigor & benchmark literacy"). The third Q3
pillar and the last clause of the fair-comparison contract: **report the gap honestly** now means *prove
it is not noise*. Core paper: **Demšar 2006**.

**Single skill:** given per-fold scores for two models, run a **paired** significance test to decide whether
the gap is real — and recognize the failure mode: a naive paired t-test on overlapping CV folds is
**anticonservative** (fabricates significance). Fix on one dataset = the **corrected resampled t-test**
(Nadeau & Bengio 2003); across many datasets = **Friedman + Nemenyi CD** (Demšar 2006).

**Primary reading.** Demšar 2006, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR
7:1–30 ([jmlr.org/papers/v7/demsar06a](https://jmlr.org/papers/v7/demsar06a.html)) — abstract, §3.2
(Wilcoxon), §3.5 (Friedman + Nemenyi), CD diagrams. Single-dataset support: Nadeau & Bengio 2003 (corrected
resampled t-test); Dietterich 1998 (5×2cv, McNemar). **arxiv MCP unavailable this session** (only
`user-arxiv` listed, cloud env); verified Demšar's recommendations + Nadeau–Bengio's corrected-t formula
via web fetch of the JMLR/Springer/NeurIPS PDFs. Key facts used: Wilcoxon for two classifiers, Friedman +
Nemenyi post-hoc for many, `CD = q_α·√(k(k+1)/6N)`; Nadeau–Bengio variance multiplier `(1/n + n_test/n_train)`.

**Verified live (`labs/_verify_l023.py` + executed solution, sklearn 1.9.0 / scipy 1.18.0, seed 0):**
- **Failure mode (core):** LogReg 0.8246 vs GaussianNB 0.8148 over `RepeatedStratifiedKFold(10×10)` = 100
  paired folds; mean gap **+0.0098** (63% of folds favour LR). **Naive paired t: t=+4.617, p=1.17e−5
  (SIGNIFICANT)**; **corrected resampled t: t=+1.327, p=0.188 (NOT sig)** — the "win" is noise.
- **The Wilcoxon trap:** Wilcoxon on the same 100 correlated folds gives **p=5.0e−5 (also over-rejects)** —
  Wilcoxon/Friedman are for *independent datasets*, not one dataset's folds. Explicitly taught as a trap.
- **Contrast (real gap):** LogReg 0.8246 vs DecisionTree(d=3) 0.7140, gap +0.1106 → naive p=3e−46,
  corrected p=2.6e−11, Wilcoxon p=3.8e−18 → **all agree significant** (a real gap survives the correction).
- **Demšar CD:** 4 models × N=12 synthetic datasets. Avg ranks LogReg **1.083** · GaussianNB **2.083** ·
  RandomForest **3.417** · HistGBDT **3.417**. Friedman **chi²=27.8, p=4.0e−6**; Nemenyi **CD=1.354**
  (α=0.05, k=4). LogReg–NB gap 1.00 < CD (not sig different); LogReg vs RF/HistGBDT gap 2.33 > CD
  (significant). Near-linear data, so LR deservedly tops — the *method* is the point, not "trees lose".

**Two new reusable viz (standard #9 — one viz per distinct mechanism):**
- `assets/paired-diff-viz.js` — 100 per-fold differences as dots on a zero-centred axis + the 95% CI on the
  mean, with a **naive/corrected toggle** (same mechanism under a variance knob): naive band narrow, 0
  outside → "SIGNIFICANT p=1.2e−5"; corrected band wide, straddles 0 → "not sig p=0.19". The core skill +
  failure mode in one coherent widget.
- `assets/cd-diagram-viz.js` — Demšar's critical-difference diagram: average-rank axis, a CD ruler, and
  join-bars for non-significantly-different cliques; click a model to highlight its group. The paper's
  headline tool (iconic, reusable for any leaderboard read).
Both headless-verified `labs/_viz_check_l023.js` 12/12. **Browser MCP still unavailable** (empty tools
folder) → headless only, per the standing gap.

**Pedagogy (all Q2-retrospective standards applied):** spaced-retrieval warm-up (`upTo: 23`);
prediction-before-reveal on the corrected verdict (still significant? → no, noise); teach-back on *why* the
naive t-test over-rejects; 3 quizzes; a "which test when" decision table. Fed the artifacts:
retrieval-pool +2 (`l023-corrected` [misconception], `l023-cd-diagram`); paper-deck +1 (`demsar2006`);
misconceptions **M20** (Q3: "a mean gap / paired test over folds proves A is better"); thesis-dossier +1
(BAR, C3) + honest-bar item #5 ("prove the gap is not noise"); `reference/glossary.html` +5 Q3 terms
(paired significance test, anticonservative, corrected resampled t-test, Friedman test, Nemenyi CD).
`node labs/_check_pedagogy.js` clean.

**Lab:** `labs/0023-statistical-comparison.ipynb` — Tier-C concept lab (no relkit; concept lab, same call as
L019/L021/L022), built via `labs/_build_l023.py`. 3 TODO (crucial fragment each) + stretch: T1 = build the
100 paired fold diffs + naive `ttest_rel`; T2 = **implement the corrected resampled t-test** (the crucial
fragment — fill the `1/n + 1/(k−1)` correction) + see the Wilcoxon trap; T3 = rank 4 models over 12
datasets, `friedmanchisquare`, and the Nemenyi `CD` formula. Student blank (9 `____`, 0 outputs); solution
executed clean (all CHECK + EXIT) and gitignored. Manifest → 23 entries (quarter 3); all labs re-rendered
to `labs/html/`.

**Env note:** `.venv` was already present and functional this session (sklearn 1.9.0, numpy 2.5.0, scipy
1.18.0, Node v20) — no uv/venv rebuild needed, unlike L021/L022. The prior env-setup preinstall took.

**Thesis bridge:** C3 (the RDL gain is real and fair) is not won by a bigger mean — it is won by a gap a
skeptic's *significance test* cannot dissolve. When a relational model edges a tuned GBDT by tenths of a
point on a RelBench task, the correct reaction is a corrected resampled test (single task) or a Friedman/CD
verdict (across tasks) + an effect size, exactly how TabReD/TabArena report. Added as honest-bar clause #5.

Next: Lesson 024 (The Grinsztajn benchmark — Grinsztajn et al. 2022 §1–4; run one dataset from the repo),
opening the four-lesson Grinsztajn arc (024–027) that pays off the L019 preview.
