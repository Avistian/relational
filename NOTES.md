# Teaching Notes

## Session 0 — 2025-06-24

- User invoked `/teach` with a contrarian thesis: relational deep learning and foundational relational models are undervalued and may outperform current approaches.
- Mission established: multi-year arc, ~1 hour/day, from basic tabular through cutting-edge RDL research.
- Goal includes demonstrating value to the world (benchmarks, publication, or shipped result).
- Prior signal from workspace: user has done serious tabular pipeline work (grouped CV, structural missingness, sklearn baselines in `homework/report.md`) — do not re-teach median imputation from scratch; do validate leakage and evaluation discipline early.

## Session 1 — 2025-06-24

- User requested **detailed expert curriculum**: lectures + readings year-by-year, then novel research framing.
- Created `CURRICULUM.md` (240 lectures, 6 years, agent-facing) and `reference/curriculum.html` (student-facing).
- Extended roadmap from 5 to 6 phases (Year 6 = novel research & thesis validation).
- RESOURCES.md reorganized by curriculum year.

## Session 2 — 2026-06-24

- User asked to critically revise `CURRICULUM.md` for completeness on tabular/relational mastery; ensure TabPFN + newer approaches included. Used arXiv MCP (`user-arxiv-mcp-server`) to verify papers/IDs.
- **Year 2 fully restructured** into 4 coherent quarters: (Q1) classic neural architectures, (Q2) modern DL + honest baselines, (Q3) tabular foundation models / ICL, (Q4) SSL + encoders + bridge.
- **Added (verified arXiv IDs):** TabR `2307.14338`, RealMLP `2407.04491`, TabM `2410.24210` (best DL baseline), TabReD `2406.19380`, TabArena `2506.16791`, TabPFN v2 (Nature 2025), TabICL `2502.05564`, LoCalPFN `2406.05207`, Drift-Resilient TabPFN `2411.10634`, open-env critique `2505.16226`, ExcelFormer `2301.02819`, Trompt `2305.18446`.
- **Relational frontier added:** ContextGNN `2411.19513`, RelGNN `2502.06784`, RelGT `2505.10960`, Griffin `2505.05568`, RDB-PFN `2603.03805` (open synthetic-prior relational FM), GelGT `2605.15575`, RelGT-AC `2606.03040`.
- Added a **Verified paper index** appendix to CURRICULUM.md and a **currency rule** (re-search arXiv each quarter; only add SOTA / failure-mode / baseline papers).
- Critical framing added up top: GBDTs + RealMLP/TabM still win single-table under temporal splits; thesis lives at the *relational* frontier.

## Session 3 — 2026-06-24

- User started Lesson 001; completed comprehension check-in.
- Strong on temporal leakage; baseline-fairness intuition present.
- Retrieval gap closed: design matrix + label vector (structural forcing mechanism).
- **Lesson 001 complete** (2026-06-24).
- Lesson 002 published: design matrix & leakage join sketch.

## Session 9 — 2026-06-27

- User started **Lesson 005** (pipelines & preprocessing).
- Warm-up 3/3: `best_score_` (A), scaler leak (B), aggregates as preprocessing (C). Prior `best_score_` misconception appears closed.
- **Lesson 005 complete** ("Lab done").
- **New preference:** labs now ship as fill-in **notebooks** (`labs/NNNN-<slug>.ipynb`, pattern in `labs/LAB-TEMPLATE.ipynb` + `labs/README.md`). Convention recorded in Preferences below.
- **Lesson 006 published** — missingness taxonomy MCAR/MAR/MNAR; new reusable `assets/missingness-viz.js`; first notebook lab `labs/0006-missingness.ipynb`. Primary reading: van Buuren FIMD §1.2/§2.2.4 (free) + sklearn §6.4. **Not run live** (no sklearn this session) — the notebook is the verification.
- **Lesson 006 complete** — EXIT TICKET: MCAR bias +0.002, MAR −0.380, MNAR −0.552; indicator +0.016 CV acc. Takeaway correct on MNAR/indicator; nudge to name MAR and what indicator actually preserves.

## Session 12 — 2026-06-30

- User started **Lesson 008** (metrics: ROC vs PR & calibration).
- Warm-up 3/3: PR baseline = prevalence (A), rank vs calibrate split (A), calibrator on held-out data (A).
- **Lesson 008 complete** ("lab done"). Lab surfaced two real `CalibratedClassifierCV` snags the user worked through: `cal.fit()` returns the estimator not probabilities (→ `predict_proba(Xte)[:,1]`), and the calibrator needs a base estimator. Note the `fit()`-returns-self confusion for callback.
- **Lesson 009 published** — feature engineering (curriculum lec 009). New reusable `assets/feature-viz.js` (raw column vs engineered ratio scatter). **Verified live (sklearn 1.9):** ratio of differences Ridge R² 0.644→0.998 (HistGBDT 0.968→0.980); cyclical datetime 0.840→0.979; target-encoding leak 0.76 (full-data) vs 0.50 (per-fold pipeline) on a pure-noise category. Lab `labs/0009-feature-engineering.ipynb` runs end-to-end verified. Primary: Kanter & Veeramachaneni 2015 (DFS) + Heaton 2016.
- Thesis bridge made explicit: manual relational FE → DFS/Featuretools → RDL (learns aggregations end-to-end).
- User started **Lesson 009** (feature engineering). Warm-up **3/3** (target encoding in pipeline A; PR baseline = prevalence A; PIT aggregate A — L005/L008/L002 retrieved cold). **Lesson 009 complete** ("lab done").
- **Lesson 010 published — Q1 CHECKPOINT** (reproducible sklearn baseline; curriculum lec 010). Capstone synthesizing 001–009, not a new concept. New reusable `assets/checklist.js` (interactive leakage-spine rubric, reusable for Q2/Q3/Q4 checkpoints). **Verified live (sklearn 1.9, n=8000, 15.4% pos, 18% MAR-missing, mixed-type, irregular signal):** Dummy PR-AUC 0.154 (=prevalence), Logistic CV-PR 0.381 / ROC 0.733 / Brier 0.114, HistGBDT CV-PR 0.470 / ROC 0.789 / Brier 0.108 (wins all); calibration 0.1078→0.1055. Capstone lab `labs/0010-baseline-checkpoint.ipynb` runs end-to-end verified. Primary: Grinsztajn 2022. No new deps (HistGBDT native; XGBoost waits for Q2).
- Reproduction-lab incremental rule (see Preferences) first applies at **Q2 (lec 011+)**: tuned XGBoost/LightGBM/CatBoost build on this Q1 baseline harness. Consider seeding `labs/relkit/` then.
- Q1 (001–010) **complete** after L010 lab. Next: Lesson 011 (Q2, gradient boosting — XGBoost).
- **Curriculum currency (2026-07-01):** Q2 pass added TabPFN-2.5/3, TabICLv2, TabH2O, Relational Transformer, OpenRFM to verified index + RESOURCES.
- Exit: `labs/0010-baseline-checkpoint.ipynb` EXIT TICKET or "lab done."
- **Lesson 011 published (2026-07-01)** — decision trees as partitions (Q2 start). New `assets/tree-viz.js`, `assets/leakage-viz.js` (L002 retrofit), `labs/relkit/` harness, Tier-A OpenML credit lab. Home page now reads `lessons/manifest.json`. Q1 feedback standards in NOTES Preferences + three project skills.

## Session 14 — 2026-07-02

- **Lesson 011 complete** — "lab 11 done." Lab scored ≈9/10 (rubric in [[learning-records/0035-lesson-011-complete.md]]). Live fixes that stuck: Gini = `1−p²−(1−p)²` (not `p`), ΔG = parent − weighted children. Callback for L013/L014 gain.
- **New standard #8 (lab intros):** user flagged labs were too thin on explanation → labs now require in-notebook concept recap + per-task goal/why (`lab-authoring` § Introductory content). `LAB-TEMPLATE.ipynb` updated; Lab 011 retrofitted.
- **Lesson 012 published** — bagging & random forest (curriculum lec 012). New reusable `assets/ensemble-viz.js` (slider: variance collapses ~1/B as trees are averaged). **Verified live (sklearn 1.9, credit_g, prevalence 0.700):** single deep tree CV PR-AUC 0.757; RF(300) 0.901, OOB acc 0.754; per-row pred std 0.270 (single) → 0.068 (10-tree ensemble). Lab `labs/0012-bagging-random-forest.ipynb` imports `relkit` (incremental rule). Primary: Breiman 2001.
- **Web notebooks added:** `scripts/render_notebooks.sh` (nbconvert → `labs/html/`, all 7 labs rendered), `notebooks.html` gallery + `assets/notebooks.js` (View HTML / source / Run on Binder / Colab), `binder/requirements.txt` for the runnable path. Home page links the Notebooks page. Binder is the zero-setup full-run path (has network + relkit via CWD=labs/); Colab noted as needing a clone.
- **Lesson 012 complete** — "lab and lesson 12 done."
- **Lesson 013 published** — boosting intuition (curriculum lec 013). New reusable `assets/residual-viz.js` (stagewise residual boosting: flat mean → fits residual stumps → MSE collapses; slider = rounds M). Lab `labs/0013-boosting-intuition.ipynb` — **crucial fragment** is the by-hand residual loop (that *is* gradient boosting), then lr trade-off, then tree/RF/GBDT on `credit_g`. Solved copy executed end-to-end (all CHECK + EXIT clean). Friedman 2001 added to RESOURCES. Record: [[learning-records/0037-lesson-013-published.md]].
- **Verified live (sklearn 1.9):** toy residual loop MSE 0.359→0.004 (60 rounds); lr sweep (40 rounds) 0.05→0.084 / 0.3→0.008 / 1.0→0.007; **credit_g** tree 0.757 / RF 0.901 / GBDT 0.879 (RF edges untuned boosting on tiny noisy data); **adult** RF 0.785 / GBDT 0.824 (boosting wins with real signal). Honest bagging-vs-boosting contrast baked into the "On our data" table.
- Browser MCP unavailable this session → `residual-viz.js` verified headlessly in Node (no runtime errors; MSE readout collapses M=0→M=40).
- Next: Lesson 014 (XGBoost — Chen & Guestrin 2016; tune XGB, regularized split gain; ΔG-sign callback from L011).

## Session 11 — 2026-06-29

- User started **Lesson 007** (class imbalance).
- Warm-up 3/3: accuracy paradox (A), SMOTE leak (A), indicator signal (A). L006 indicator concept retained.
- **Lesson 007 complete** ("lab done").
- **Lesson 008 published** — metrics: ROC vs PR curves & calibration (curriculum lec 008). New reusable `assets/reliability-viz.js`. **Verified live (sklearn 1.9, 17.7% positive, RF):** ROC-AUC 0.863 vs PR-AUC 0.720 (PR baseline 0.177); Brier 0.0898 raw → 0.0856 sigmoid → 0.0853 isotonic; reliability shows RF under-confident mid-range (pred 0.55 → actual 0.72). Primary: Saito & Rehmsmeier 2015 + Niculescu-Mizil & Caruana 2005 + sklearn §1.16.
- Next: Lesson 009 (feature engineering, curriculum lec 009); lec 010 = Q1 reproducible-baseline checkpoint.
- Exit: `labs/0007-class-imbalance.ipynb` EXIT TICKET or "lab done."

## Session 10 — 2026-06-28

- Created lab **environment**: `.venv/` (sklearn 1.9.0, pandas 3.0.3, numpy 2.5.0; later +imbalanced-learn 0.14.2), `requirements-labs.txt`, `labs/setup-env.sh`, kernel "Relational Labs (.venv)", `.vscode/settings.json`.
- **Lesson 007 published** — class imbalance (curriculum lec 007, He & Garcia 2009). New reusable `assets/imbalance-viz.js` (threshold→confusion-matrix slider; accuracy paradox). New dep imbalanced-learn. **Verified live:** accuracy paradox 0.945 acc / 0.0 recall; SMOTE leak F1 0.887 vs 0.479 honest; class_weight recall 0.55→0.83 (PR-AUC flat 0.756→0.734); ROC 0.914 vs PR 0.747.
- Next: Lesson 008 (metrics: ROC vs PR, calibration) — previewed in L007, not yet published.

## Session 8 — 2026-06-26

- User started **Lesson 004** (grouped & nested CV); warm-up: grouped leak correct (A); nested-CV bias chose D (best_score_=train) → corrected to B (selection bias; best_score_ is a *validation* score chosen by same data).
- **Lesson 004 complete** ("lab done").
- **Lesson 005 published** — pipelines & preprocessing; new reusable `pipeline-viz.js` (leak vs per-fold fit). Primary reading: sklearn §6.1 Pipelines + mixed-types ColumnTransformer example.
- **Verified live on sklearn 1.9:** SelectKBest-before-CV = 0.78 vs 0.44 honest (pure noise); OHE pandas output needs `sparse_output=False`; `handle_unknown="ignore"` required for test-only categories.
- Next: Lesson 006 (missingness taxonomy MCAR/MAR/MNAR) — not yet published. Watch the `best_score_`-is-training misconception doesn't resurface.

## Session 7 — 2026-06-26

- User started **Lesson 003** (train / valid / test & CV).
- Warm-up retrieval solid: country=safe; design matrix forcing mechanism correct.
- **Lesson 003 complete** — lab done; stratified split + CV on train.
- **Lesson 004 published** — grouped & nested CV; new `group-viz.js`; Cawley & Talbot 2010 + sklearn §3.1.2.4 primary reading; verified lab claims on sklearn 1.9 (corrected metadata-routing example to require `set_config`).
- Next: Lesson 005 (pipelines & preprocessing) — not yet published.

## Session 6 — 2026-06-25

- **Lesson 002 complete** — join sketch 5/6; country/static-dimension distinction to reinforce.
- **Lesson 003 published** — train/valid/test, stratified CV, split-viz widget, sklearn lab.

## Session 5 — 2026-06-25

- User started **Lesson 002** (design matrix & leakage).
- Retrieval from L001 assumed solid; session focuses on PIT aggregates and join-sketch lab.

## Session 4 — 2026-06-24

- User asked to add **optional papers** ("follow when time allows") and update the curriculum; scan arXiv + Christoph Molnar's blog for candidates.
- arXiv MCP (`user-arxiv-mcp-server`) was **not available** in this (cloud) environment — verified all arXiv IDs via web search/fetch instead.
- Introduced a **★ core / ◆ optional** paper tier in `CURRICULUM.md` (framing note + per-year "Optional / time-permitting" blocks + an "Optional / extension reading" index with verified IDs).
- **Added optional papers (verified IDs):** Y1 SHAP `1705.07874`, Molnar *Interpretable ML* 3rd ed.; Y2 CARTE `2402.16785`, Interpretable-ML-for-TabPFN `2403.10923`, TabLLM `2210.10723` (as the LLM boundary the thesis rejects), Molnar *Tabular Foundation Models* book + Mindful Modeler; Y3 over-smoothing `1801.07606`, DropEdge `1907.10903`, PNA `2004.05718`, over-squashing `2006.05205`, curvature `2111.14522`, Graphormer `2106.05234`, GraphGPS `2205.12454`, plus HAN `1903.07293` / Cluster-GCN `1905.07953` / TGAT `2002.07962` / RE-Net `1904.05530` indexed for lessons that already name them; Y4 4DBInfer `2404.18209`; Y5 deferred to the currency rule.
- Synced `RESOURCES.md` (optional groups per year + Mindful Modeler community) and `reference/curriculum.html` (hero note, TOC link, new Optional Reading section).
- **Christoph Molnar** (note spelling: Christoph, not Christopher) is now the interpretability spine; he's currently writing a *Tabular Foundation Models* book — directly on-thesis.

## Paper progress

Track with ✓ as completed:

- [ ] Y1: Chen 2016 XGBoost
- [ ] Y1: Grinsztajn 2022
- [ ] Y1: Fey 2024 §1
- [x] Y1: Fey 2024 §1 assigned in Lesson 001

## Preferences

- **Lab environment:** `.venv/` at repo root via `bash labs/setup-env.sh` + [`requirements-labs.txt`](requirements-labs.txt); Jupyter kernel **Relational Labs (.venv)**; Cursor picks up `.venv/bin/python` via `.vscode/settings.json`.
- **Labs as notebooks (from L006 on):** every lab ships as a Jupyter notebook in `labs/NNNN-<slug>.ipynb` following [`labs/LAB-TEMPLATE.ipynb`](labs/LAB-TEMPLATE.ipynb). Convention: **PROVIDED** cells = complete boilerplate; **TODO** cells = blanks (`____` / `# TODO`) only on the skill being practised; **CHECK** cells = auto-feedback assertions (don't edit); final **EXIT TICKET** cell prints the deliverable. Keep blanks focused on the one idea so working memory stays free. Pattern documented in `labs/README.md`.
- **Reproduction labs build incrementally (not yet active — note for later):** once we reach paper-reproduction labs (RelBench baselines, GBDT/RealMLP/TabM, RDL), each lab must build on the code already written in earlier labs rather than re-deriving from scratch — a cumulative, reusable codebase (shared data loaders, eval/CV harness, leakage-safe pipelines, metrics) that grows lab over lab. Likely promote shared code from notebooks into a small importable package (e.g. `labs/relkit/` or `src/`) and have reproduction notebooks import it. Goal: by the time results matter for the thesis, the baseline harness is battle-tested across many labs, not a pile of one-offs. Standalone *concept* labs (like L006) can stay self-contained; the incremental rule kicks in for reproduction/experiment labs.
- **Lesson length:** prefer longer, richer lesson HTML (~35–45 min reading + quizzes) over short skims; chunk with worked examples and multiple check-your-understanding blocks
- **Pace:** year-by-year phases; do not skip tabular foundations even though the thesis is relational
- **Time:** ~1 hour/day **baseline** (minimum on typical days); may study longer when energy and schedule allow — curriculum pacing assumes 360 h/year at baseline
- **Extra time:** prefer deeper labs, paper appendices, checkpoint work, or spaced retrieval — not skipping year exit exams
- **Technical:** lessons use plain `<script>` tags, not ES modules (works on `file://` and GitHub Pages)
- **GitHub Pages:** site at `https://avistian.github.io/relational/` once pushed and Actions enabled (Settings → Pages → Source: GitHub Actions)
- **Mobile:** bookmark the home page; lesson list grows as new lessons are published
- **Out of scope:** see [[MISSION.md]]

### Q1 retrospective (2026-07-01) — standards for all future units

Recorded after Q1 checkpoint. Full rationale: [[learning-records/0032-q1-retrospective-feedback.md]].

1. **Lesson visuals:** Basic HTML is fine for simple ideas; anything spatial, temporal, or mechanistic needs a reusable viz in `assets/`. Always verify viz matches prose in the browser before publishing (`lesson-visuals` skill).
2. **Quiz fairness:** `assets/quiz.js` shuffles options on mount — never rely on answer position. Author may keep `correct: "a"` in lesson source; display order is randomized.
3. **Lab difficulty (mid zone):** TODO cells have blanks only — never prefilled solution code or code hints in markdown. Hints describe *what*, not *how*. **Lab intros (from L011):** concept recap + per-task goal/why markdown before coding — see `lab-authoring` § Introductory content. Paper-repro labs use the 4-block structure (paper step → crucial fragment → harness → reproduction target). When the user says *lab done*, score with the rubric in `lab-authoring` skill (0–2 × 5 axes, max 10).
4. **Home page:** Do not hand-list lessons in `index.html`. Update `lessons/manifest.json` when publishing; `assets/home.js` renders year/quarter navigation.
5. **Research currency:** Before each quarter, run `curriculum-currency` skill (arxiv MCP + TabArena/RelBench checks). Sync `CURRICULUM.md`, `RESOURCES.md`, `reference/curriculum.html`.
6. **Project skills:** Use `lesson-visuals`, `lab-authoring`, `curriculum-currency` from `.agents/skills/` — do not re-derive these rules from chat.
7. **Datasets:** Tier A (real OpenML/UCI) default for Q2+ training labs; Tier C (synthetic) only for mechanism isolation. See `labs/data/README.md`.
8. **Lab intros (from 2026-07-02):** labs must **introduce the topic in-notebook** — concept recap (terms, formulas, one toy worked example), plus goal/why before each task. Student should not need to re-open the lesson HTML for core definitions. Explanations in markdown; implementation stays in TODO blanks. See `lab-authoring` skill § Introductory content. Lab 011 retrofitted as reference.
