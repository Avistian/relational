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

## Session 15 — 2026-07-03

- **Lesson 013 complete** — "lesson/lab 13 done." Lab scored ≈9/10 ([[learning-records/0038-lesson-013-complete.md]]); only gap was the EXIT `takeaway:` sentence left blank. Residual-loop-as-gradient and shrinkage retained.
- **New dependency:** `xgboost` 3.3.0 installed into `.venv` and added to `requirements-labs.txt` (first Q2 lab to add a package beyond sklearn/imblearn).
- **Lesson 014 published** — XGBoost (curriculum lec 014, Chen & Guestrin 2016 §2). Regularized objective Ω=γT+½λ‖w‖²; closed-form leaf weight `w*=−G/(H+λ)`; structure-score split gain (= L011 children−parent, now regularized, −γ toll). Systems callbacks: sparsity-aware missing (L006), column/row subsample (L012), shrinkage (L013).
- **New reusable asset:** `assets/gain-viz.js` (λ shrinks leaf weights/scores; γ lifts a toll line and flips KEEP→PRUNE). Verified headlessly in Node (mount + math; browser MCP unavailable again — only user-arxiv authed).
- **Verified live (sklearn 1.9 + xgboost 3.3.0, `relkit`), from executed `solutions/0014-xgboost.ipynb`:** by-hand leaf weight **exactly** matches XGBoost's leaf output (λ=3 → −0.480/+3.095); λ shrinks w* (0→−0.591 … 10→−0.334); γ=1.5×raw prunes; **credit_g** GBDT 0.879 · XGB default 0.883 · XGB tuned (40-iter search) **0.896** (RF from L012 still 0.901 — honest near-tie); **adult** GBDT 0.824 · XGB default 0.829.
- **Honest-baseline discipline reinforced:** untuned XGB ≈ sklearn; the +0.017 comes from tuning. Lesson states plainly the relational thesis must beat a *tuned* booster, not a default.
- **Lab (incremental rule active):** `labs/0014-xgboost.ipynb` — paper-repro 4-block; crucial fragment = implement `node_score`/`reg_leaf_weight`/`split_gain` and match XGBoost's leaf outputs. Built via `labs/_build_l014.py`; student blank (4 TODO cells, no outputs), solution executed clean & gitignored.
- Manifest regenerated (14 entries), all labs re-rendered to `labs/html/` (incl. `0014-xgboost.html`).
- **Callback closed:** L011 ΔG-sign (children − parent) re-derived in regularized form in L014.
- Next: Lesson 015 (LightGBM — Ke et al. 2017; histogram + leaf-wise growth, speed vs XGB).

## Session 16 — 2026-07-03

- User requested **deep-research merge** (tabular + relational DL) into curriculum files; exhaustive paper list + mark what's already solved.
- **July 2026 currency pass** merged into `CURRICULUM.md`, `RESOURCES.md`, `reference/curriculum.html`, `curriculum-currency` skill.
- **ID fixes:** PyTorch Frame `2402.05964` → `2404.00776`; TALENT `2407.04057` → `2407.00956`.
- **New ★ core:** BeyondArena `2606.30410`, temporal-shift limits `2502.20260`, Operational TTF `2606.29091`, RDBLearn `2602.13697`, KumoRFM-2 `2604.12596`, OpenRFM promoted to ★, Universal Row Encoder `2606.21434`, Desired graph `2606.08491`, three FM paradigms synthesis (Y5).
- **Already solved (no re-add):** Y1 Q1–Q2 published lessons; Grinsztajn/TabM/TabR/RealMLP/TabReD/TabArena/TabPFN/TabICL chain; Fey/RelBench/RelGNN/RelGT/ContextGNN; Griffin/RDB-PFN/RT; RelBench v2/GelGT/RelGT-AC — all were in June 2026 pass.
- **2026 fair stack** recorded in CURRICULUM: tuned GBDT + RealMLP/TabM + TabICLv2 (n≲50K), time splits, TabArena/BeyondArena protocol.
- Exhaustive registry (70+ papers by category) added to CURRICULUM appendix.

### New ★ papers — when it wins / when it breaks (skim notes; deep read at lecture time)

- **BeyondArena (`2606.30410`):** Wins as unified eval framework across IID/temporal/grouped. Breaks if you cite TabArena-only results for "TFMs dominate" — non-IID/large/high-dim still tree/DL territory.
- **RDBLearn (`2602.13697`):** Wins when DFS+TabICL beats training an RDB FM from scratch; open/reproducible. Breaks when graph-native signal needs end-to-end message passing (dense many-to-many, autocomplete with text).
- **KumoRFM-2 (`2604.12596`):** Wins RelBench v1+v2 SOTA; sets commercial ceiling. Breaks for thesis reproduction — proprietary; compare against OpenRFM/RDBLearn/RelGNN instead.
- **OpenRFM (`2606.04320`):** Wins as best **open** relational FM; dual-stage ICL fixes RT label scarcity. Breaks vs KumoRFM-2 on hardest tasks; still beats RT ~30%.
- **Desired graph (`2606.08491`):** Wins when schema-derived REG is overloaded/fragmented; filtering+injection beats raw graph. Breaks if you skip understanding Fey's default REG first — this is an optimizer on top, not a replacement for RDL vocabulary.
- **Operational TTF (`2606.29091`):** Wins as falsifiable barrier for values-only TFMs in rule-governed DBs. Breaks if misread as "TFMs useless" — it's about **operational grounding**, not IID accuracy.

## Session 17 — 2026-07-03

- **Lesson 014 complete** — "lesson 14 done." Lab ≈9/10 ([[learning-records/0040-lesson-014-complete.md]]). Pre-"done" Q&A showed strong retrieval: g/h are loss derivatives w.r.t. current score F (squared: g=F−y, h=1; logistic: g=p−y, h=p(1−p)), recomputed each round and discarded at inference; thresholds from a hessian-weighted histogram; gain≤0 after γ ⇒ prune; η only in the outer update. L011 ΔG callback fully closed.
- **New dependency:** `lightgbm` 4.6.0 installed into `.venv` + `requirements-labs.txt`.
- **Lesson 015 published** — LightGBM (curriculum lec 015, Ke et al. 2017 NeurIPS). Histograms + subtraction, leaf-wise (best-first) growth (⇒ `num_leaves` is the primary knob, not `max_depth`), GOSS (§3), EFB (§4). New reusable `assets/growth-viz.js` (leaf-wise vs level-wise toggle + split slider; readout shows both losses so leaf-wise ≤ level-wise). Record: [[learning-records/0041-lesson-015-published.md]].
- **Verified live (`_verify_l015.py` + executed solution):** GOSS unbiased (~30% rows, 8× amplify, mean err ≈+0.06/200 seeds); credit_g GBDT 0.879 · XGB 0.883 · **LGBM default 0.889** · LGBM tuned 0.893 (num_leaves=7); num_leaves sweep 7→0.889…127→0.884 (overfits); adult XGB 0.829 ≈ LGBM 0.831; **speed (50k×50):** sklearn GBDT 64.3s vs XGB-hist 0.58s (110×) vs LGBM 0.60s (107×) vs LGBM-goss 0.42s (152×).
- **Honest myth-buster kept:** "20× faster" is vs *conventional pre-histogram* GBDT; modern XGBoost-`hist` is on par (mirrors L014's "untuned XGB ≈ 2016 sklearn"). GOSS is opt-in (`boosting_type='goss'`).
- **GOSS bug caught by CHECK:** first `goss_weights` sampled `b·|rest|` (biased by (1−a)); fixed to `b·n` per Algorithm 2. Good example of the auto-feedback loop working.
- **Currency fix:** CURRICULUM had LightGBM as arXiv `1711.08251` (actually an unrelated hep-ph paper). LightGBM has **no arXiv**; corrected to NeurIPS proceedings in CURRICULUM.md + RESOURCES.md.
- Browser MCP unavailable again → `growth-viz.js` verified headlessly in Node (mount + gain-table logic; leaf-wise ≤ level-wise from split 3 on). Manifest regenerated (15 entries); all labs re-rendered to `labs/html/`.
- Next: Lesson 016 (CatBoost — Prokhorenkova et al. 2018; ordered boosting + categorical handling).

## Session 18 — 2026-07-04

- **Lesson 015 complete** — "lesson/lab 15 done." Lab ≈9/10 ([[learning-records/0042-lesson-015-complete.md]]); GOSS unbiasedness + (1−a)/b amplification + leaf-wise-overfits-small-data all retained.
- **New dependency:** `catboost` 1.2.10 installed into `.venv` + `requirements-labs.txt` (first Q2 lab to add a package beyond sklearn/imblearn/xgboost/lightgbm).
- **Lesson 016 published** — CatBoost (curriculum lec 016, Prokhorenkova et al. 2018, arXiv `1706.09516`). The "leakage, not speed" GBDT: **ordered target statistics** (encode row *i* from its permutation prefix + prior `a·p`; greedy whole-column mean leaks the own label, worst on rare cats), **ordered boosting** (§4; gradient from a model that hasn't seen row *i* → kills prediction shift), **oblivious/symmetric trees** (depth-6 = 64-leaf decision table; regularized + fast). Record: [[learning-records/0043-lesson-016-published.md]].
- **New reusable asset:** `assets/ordered-viz.js` (permutation table; pointer picks the current row; greedy circles the row's own y as the leak, ordered shades only the prefix; reshuffle → ordered value moves with the permutation). Headless Node mount check clean (browser MCP unavailable again; only user-arxiv authed).
- **Verified live (`_verify_l016.py` + executed solution, `relkit` 5-fold PR-AUC):** pure-noise category (K=700, y ⟂ cat) AUC(enc,y): greedy **0.845** (leak) vs ordered **0.493** (clean, 20 perms); **credit_g** XGB 0.883 · LGBM 0.880 · **CatBoost 0.896** (native cats win on categorical-rich small data); **adult** all ≈0.831 (near-tie); ordered-vs-plain boosting 0.889 vs 0.896 (near-wash — edge is from ordered TS, not ordered boosting; honest note kept).
- **CatBoost sklearn gotcha (documented in `_verify_l016.py` + lab):** `cat_features` param does not survive `sklearn.clone`, so `relkit.cv_pr_auc`/`cross_val_score` fail on it. Used a clone-free manual 5-fold helper (build a fresh estimator per fold). Also: CatBoost cat columns must be str/int, not NaN → fillna `"missing"` before `astype(str)`.
- **Lab:** `labs/0016-catboost.ipynb` — crucial fragment = `ordered_ts` by hand (encode-before-update); Task 2 = CatBoost vs XGB vs LGBM on credit_g. Student blank (3 TODO), solution executed clean & gitignored. Manifest regenerated (16 entries); all labs re-rendered to `labs/html/`.
- Next: Lesson 017 (Hyperparameter search — Bergstra & Bengio 2012; random vs grid; RandomizedSearchCV).

## Session 19 — 2026-07-05

- **Lesson 016 complete** — "lesson/lab 16 done."
- **Lesson 017 published** — Hyperparameter search (curriculum lec 017, Bergstra & Bengio 2012, JMLR 13 — **no arXiv**, JMLR PDF). Single skill: why random > grid at equal budget under low effective dimensionality (grid tries √n distinct values/axis, random ≈n), log-uniform sampling, and the nested-CV honesty rule (L004 selection-bias callback). Record: [[learning-records/0044-lesson-017-published.md]].
- **New reusable asset:** `assets/search-viz.js` (the iconic B&B Fig 1 — Grid/Random toggle + budget selector + Resample; dashed projections onto the important axis; readout = distinct-axis-values + best peak found). CSS `.sv-*` in lesson `<style>`. Headless Node mount check clean; **browser MCP unavailable again** (empty tools folder; only user-arxiv authed).
- **Verified live (`_verify_l017.py` + executed solution):** synthetic crossover (honest!) — 2-D budget 9 grid **1.019** vs random 0.771 (lucky grid wins low-dim); 5-D budget 32 grid **0.076** vs random **0.967** (grid collapses to 2 values/axis); credit_g tuned XGB (equal budget 27): default 0.883 · grid 0.891 · **random 0.901** (= RF 0.901, > tuned-XGB 0.896); nested CV 0.895 (honest) vs best_score_ 0.901 (optimistic), gap +0.006.
- **Honest myth-buster kept:** random search is not a universal law — a well-placed grid wins in 1–2 dims; the advantage is about *many knobs, most useless*, and grows with dimensionality.
- **Lab:** `labs/0017-hyperparameter-search.ipynb` — 4 TODO + stretch (`HalvingRandomSearchCV`). Task 1 crucial fragment = `random_search` by hand + crossover; Task 2 = Grid vs Randomized on XGB/credit_g equal budget (student writes `dists` w/ loguniform + `n_iter`); Task 3 = nested CV. numpy-2.5 gotcha: `float()` on a 1-elem array raises → use `float(arr[0])`. Student blank; solution executed clean & gitignored. Manifest → 17 entries; all labs re-rendered.
- Next: Lesson 018 (Ensembling & stacking — Wolpert 1992; simple blend), then L019 (when trees win), L020 = Q2 checkpoint.

## Session 20 — 2026-07-06

- **Lesson 017 complete** — "lesson/lab 17 done" (no EXIT ticket pasted → no rubric score; record [[learning-records/0045-lesson-017-complete.md]]).
- **Lesson 018 published** — Ensembling & stacking (curriculum lec 018, Wolpert 1992 *Stacked Generalization*, Neural Networks 5(2) — **no arXiv**, ScienceDirect). Single skill: build a stack whose meta-learner trains on **out-of-fold** base predictions; diversity is the fuel; in-sample base preds leak (crown the memorizer). Record: [[learning-records/0046-lesson-018-published.md]].
- **New reusable asset:** `assets/stacking-viz.js` (level-0/level-1 SVG; 12 rows × 4 folds + META-FEATURE column; OOF "next fold ▶" fills honest green vs In-sample fills leak red). CSS `.stk-*` in lesson `<style>`. Headless Node check clean (`labs/_viz_check_l018.js`, 6/6). **Browser MCP unavailable again** (empty tools folder; only user-arxiv authed) → headless only.
- **Verified live (`_verify_l018.py` + executed solution, relkit 5-fold PR-AUC, credit_g):** bases — logistic 0.874 · XGB 0.883 · LGBM 0.889 · CatBoost 0.900 · **RF 0.901 (best single)**; diversity — GBDT↔GBDT ≈0.89, logistic↔GBDT ≈0.68; blend (OOF avg 5) **0.899** (naïve blend *trails* best single), 3-GBDT blend 0.895; **stack (logistic meta, cv=5) 0.902** (edges best single +0.001), 3-GBDT stack 0.899; **leak trap (70/30 + 1-NN memorizer):** naïve meta weights 1-NN **+3.00**, train PR-AUC 1.000 (mirage) → test 0.895; OOF meta weights 1-NN **−0.11**, train 0.885 → test 0.930; **held-out gap +0.035**.
- **Honest myth-buster kept:** on a 1000-row single table ensembling barely moves the number (+0.001); the durable lessons are the OOF mechanism, the diversity requirement, and the leak. Ensembling pays off with diverse families + more data (leaderboard regime).
- **Thesis bridge:** the real single-table baseline is a *leak-free stacked ensemble* of tuned models (TabArena/Kaggle), not a single default — the bar the RDL thesis must clear. Sets up Y1 Q3 benchmark-literacy + Y2 Q2 lec-057 (ensembling across model families).
- **Lab:** `labs/0018-ensembling-stacking.ipynb` — 3 TODO + stretch. Task 1 crucial fragment = OOF meta-features by hand (`cross_val_predict`) + blend; Task 2 = leak contrast with the 1-NN memorizer; Task 3 = `StackingClassifier` reproduction (0.902). Student blank (4 `____`, 0 outputs); solution executed clean & gitignored. **Setup-cell fix:** path insert now adds `Path(".")` *and its parent* so `relkit` imports from `labs/` or `labs/solutions/`. Manifest → 18 entries; all labs re-rendered.
- Next: Lesson 019 (When trees win — Grinsztajn 2022 preview), then L020 = Q2 checkpoint.

## Session 21 — 2026-07-07

- **Lesson 018 complete** — "lesson/lab 18 done" (no EXIT ticket pasted → no rubric score; record [[learning-records/0047-lesson-018-complete.md]]).
- **Lesson 019 published** — When trees win (curriculum lec 019), the Q2 conceptual capstone before the L020 checkpoint. **Preview** of Grinsztajn 2022 *Why do tree-based models still outperform deep learning on typical tabular data?* ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815), verified via user-arxiv MCP) — abstract + §1 only; full benchmark + §5 experiments are Y1 Q3 (lec 024–027). Single skill: name/recognize the **three inductive biases** (irregular targets, uninformative features, orientation) and the honest conditions each flips. Record: [[learning-records/0048-lesson-019-published.md]].
- **New reusable asset:** `assets/biases-viz.js` — checkerboard visualizer, two modes. *Irregular*: sharp tree board beside the same board through an SVG `feGaussianBlur` (NN smoothness bias); finer tiles → blur washes to gray. *Rotation*: axis-aligned XOR with green tree splits (fit) vs a red **staircase** after rotation (misfit); NN boundary rotates with the data (invariant). CSS `.bv-*` in lesson `<style>`. Headless Node check clean (`labs/_viz_check_l019.js`, 7/7). **Browser MCP unavailable again** (empty tools folder; only user-arxiv authed) → headless only.
- **Verified live (`_verify_l019.py` + executed solution, synthetic Tier-C, RF-300 vs MLP-(256,256), test acc):** irregular (checkerboard 2→16): tree 1.000·0.999·0.969·0.807 vs NN 0.994·0.975·0.837·**0.503** (gap +0.006→**+0.304**); uninformative (+0→+100 noise): k=0 tree 0.936 vs **NN 0.965 (NN wins clean)**, k=100 tree **0.876** vs NN 0.826 (gap −0.029→+0.049; MLP loses 0.139, tree 0.061); rotation (axis-aligned XOR + Q from QR): tree 0.998→**0.831** (drop 0.167), NN 0.979→0.976 (invariant). Lesson + viz quote **lab-executed** rotation numbers (the standalone verify gives 0.996→0.817 because the lab's Task-2 CHECK re-draws from `rng`).
- **Honest myth-buster kept:** "trees always win" is too strong — the MLP won on clean, all-informative features; the real claim is typical tabular data has all three properties *at once*. The edge is a small/medium-data, single-table phenomenon that large data, perceptual inputs, and Year-2 tabular nets narrow or flip. Kept the "GBDTs are not dead" guardrail explicit.
- **Thesis bridge:** core of the Y1 exit criterion — the RDL bet is "the single-table representation discards relational structure," not "DL is bad at tables." Grinsztajn sets the honest bar (beat a learner whose biases fit flat data). Sets up Q3 lec 024–027 + Y1 exit.
- **Lab:** `labs/0019-when-trees-win.ipynb` — 3 TODO (crucial fragment each) + stretch. Task 1 = checkerboard parity label; Task 2 = `np.hstack` a `rng.normal` noise block; Task 3 = orthogonal `Q` via `np.linalg.qr` + `X @ Q`. **Standalone concept lab** (Tier C, self-contained — no relkit; the incremental rule is for reproduction labs). Student blank (7 `____`, 0 outputs); solution executed clean & gitignored. Manifest → 19 entries; all labs re-rendered.
- Next: **Lesson 020 = Q2 checkpoint** (Chen 2016 + Ke 2017; match/beat a published tree baseline). Do not advance past it without passing.
- **Per-lesson decomposition built (2026-07-07):** created [`plan/`](plan/README.md) — turnkey per-lesson specs (skill · paper §· lab crucial fragment · viz · bridge) for **every lesson 031–240**, so future sessions build lessons without re-scoping from papers. Six per-year files + README schema. Recent Y2/Y4/Y5 frontier papers verified via **user-arxiv MCP** before writing (RelGNN, RelGT, Desired-graph, Universal Row Encoder, ContextGNN, RDL survey, RelBench v1/v2, Zahradník, Griffin, RDB-PFN, RDBLearn, RT, OpenRFM, KumoRFM-2, RelGT-AC, GelGT, TabICLv2, BeyondArena, temporal-shift, Operational TTF, Cvitkovic, 4DBInfer). Collapsed CURRICULUM ranges (Y3 111–114, Y4 151–154, Y5 192–194, Y6 211–218 & 231–234) expanded into individual lessons. Linked from `CURRICULUM.md`. Record: [[learning-records/0049-lesson-plan-decomposition.md]]. **Rule going forward:** read the `plan/` entry before building any lesson 031+; it's a starting contract, update it if a live reproduction reveals a better framing.

## Session 22 — 2026-07-09

- **Lesson 020 complete** — "lesson/lab 20 done" (no EXIT ticket pasted → no rubric score; record [[learning-records/0055-lesson-020-complete.md]]). Q2 (011–020) closed.
- **Lesson 021 published — Q3 opener** — Data splits in the wild (curriculum lec 021). Single skill: spot when a random (i.i.d.) split leaks the future and build a **temporal split** (train past → test future) + `TimeSeriesSplit`. Primary: Huyen (splits + distribution shift) + **TabReD preview** (Rubachev 2024, `2406.19380`; abstract fetched from arxiv.org — arxiv MCP down, OpenML blocked by egress). Record: [[learning-records/0056-lesson-021-published.md]].
- **Two new reusable viz (standard #9):** `assets/temporal-split-viz.js` (random-scatter vs temporal-cut of the same stream, side by side) and `assets/drift-viz.js` (per-bucket corr bars + rotating rule dial). Headless-verified `labs/_viz_check_l021.js` 14/14; **browser MCP still unavailable**.
- **Verified live (`labs/_verify_l021.py` + executed solution, synthetic Tier-C drifting stream, seed 0):** logistic random-CV **0.846** · TimeSeriesSplit **0.819** · temporal-HO **0.758** (gap **+0.088**); hist_gbdt 0.832 · 0.809 · 0.757 (gap +0.076); drift corr(x0,y) +0.72→+0.12, corr(x1,y) +0.10→+0.71, prevalence flat → **pure concept drift**. Ordering random-CV > TimeSeriesSplit > temporal-HO is the lesson in one row.
- **Dataset call:** OpenML unreachable (egress: arxiv allowed, openml blocked) + no cache → Tier-C synthetic (mechanism isolation, like L019). TabReD flags `electricity` as leaky anyway, so synthetic was the right call. Documented in the lab intro.
- **Artifacts synced:** retrieval-pool +2 (`l021-temporal` [misconception], `l021-timeseriessplit`); paper-deck +1 (`rubachev2024`); **M18** in misconceptions (new Q3 section); thesis-dossier +1 (BAR+FOR, C3/C1); `reference/glossary.html` +Q3 section. `node labs/_check_pedagogy.js` clean.
- **Lab:** `labs/0021-data-splits-in-the-wild.ipynb` — Tier-C self-contained (no relkit; concept lab), 3 TODO (temporal cut · `TimeSeriesSplit` · per-bucket corr) + stretch. Student blank (9 `____`, 0 outputs); solution executed clean & gitignored. Manifest → 21 (quarter 3); all labs re-rendered.
- **Env note:** no `python3-venv`/uv preinstalled this session; installed `uv` via curl and built `.venv` (sklearn 1.9.0, pandas 3.0.3, numpy 2.5.1). Env-setup agent should preinstall the lab venv + deps so future sessions skip this.
- Next: Lesson 022 (label leakage patterns — Kapoor & Narayanan 2022, `2207.07048`).

## Session 23 — 2026-07-10

- **Lesson 021 complete** — "lesson 21 done" (no EXIT ticket pasted → no rubric score; treated as
  self-reported complete like L020). Record for L021 is [[learning-records/0056-lesson-021-published.md]].
- **Lesson 022 published** — Label leakage patterns (curriculum lec 022, Kapoor & Narayanan 2022,
  `2207.07048`). The synthesizing leakage lesson: names the whole family of leaks (8 types in 3 families)
  and adds the ones not yet met. Single skill: classify an FE choice into the taxonomy + recognize the
  **reproducibility collapse** an illegitimate feature (L2) causes. Record:
  [[learning-records/0057-lesson-022-published.md]].
- **arxiv MCP unavailable again** (only `cursor-cloud` MCP present) → verified the Kapoor & Narayanan
  abstract by fetching arxiv.org directly (17 fields / 329 papers; 8-type taxonomy; civil-war
  reproduction; model info sheet).
- **Verified live (`labs/_verify_l022.py` + executed solution, sklearn 1.9.0, seed 0):** collapse — weak
  honest tie (RF/LR ~0.72) + one non-monotone leaked column → **RF 0.935 vs LR 0.719 (gap +0.217)**;
  remove it → **0.712 vs 0.721 (gap −0.009)**. Dup leak — naïve random CV **0.948** vs GroupKFold on
  record id **0.876** (+0.071). Design note: ID-encoded and XOR leaks both failed (scaler makes ID
  monotone → helps LR; greedy trees don't reliably find XOR); the non-monotone banded proxy is the clean
  mechanism trees exploit but linear models can't.
- **Two new reusable viz (standard #9):** `assets/leakage-taxonomy-viz.js` (clickable 8-type/3-family
  map with "where you met it" + fix — the load-bearing structural visual) and
  `assets/repro-collapse-viz.js` (leak ON/OFF RF-vs-LR bar toggle). Headless `labs/_viz_check_l022.js`
  12/12; **browser MCP still unavailable**.
- **Pedagogy:** warm-up (`upTo: 22`); predict-before-reveal on the collapse; teach-back on why a leak
  widens the complex-vs-simple gap; `Checklist` mount = the **model info sheet** (7 leak-type questions);
  3 quizzes. Artifacts synced: retrieval-pool +2 (`l022-illegit` [misconception], `l022-collapse`);
  paper-deck +1 (`kapoor2022`); misconceptions **M19**; thesis-dossier +1 (BAR, C3) + new skeptic
  objection ("maybe RDL wins are leakage too"); `reference/glossary.html` +5 Q3 terms.
  `node labs/_check_pedagogy.js` clean.
- **Lab:** `labs/0022-label-leakage-patterns.ipynb` — Tier-C concept lab (no relkit), built via
  `labs/_build_l022.py`. 3 TODO (build `Xleak` + collapse · GroupKFold vs KFold · classify 6 FE snippets
  into taxonomy codes) + stretch. Student blank (15 `____`, 0 outputs); solution executed clean &
  gitignored. Manifest → 22; all labs re-rendered.
- **Env note:** no `python3-venv`/uv preinstalled again; installed `uv` via curl, built lean `.venv`
  (sklearn/numpy/pandas/ipykernel/nbconvert — no boosters needed for this lab). Env-setup agent should
  preinstall the lab venv so future sessions skip this.
- Next: Lesson 023 (statistical comparison — Demšar 2006; paired tests on CV folds), continuing Q3.

## Session 24 — 2026-07-14

- **Lesson 022 complete** — "Lesson 22 done" (no EXIT ticket pasted → no rubric score; treated as
  self-reported complete like L020/L021). Record [[learning-records/0058-lesson-022-complete.md]].
- **Lesson 023 published** — Statistical comparison of classifiers (curriculum lec 023, **Demšar 2006**,
  JMLR 7). The third Q3 pillar = the last clause of the fair-comparison contract: *report the gap honestly*
  now means *prove it is not noise*. Single skill: a **paired** significance test on CV folds + its failure
  mode (naive paired t is anticonservative). Record: [[learning-records/0059-lesson-023-published.md]].
- **arxiv MCP unavailable again** (only `user-arxiv` listed in a cloud env) → verified Demšar's
  recommendations + Nadeau–Bengio's corrected-t formula (`1/n + n_test/n_train`) by fetching the
  JMLR/Springer/NeurIPS PDFs directly.
- **Verified live (`labs/_verify_l023.py` + executed solution, sklearn 1.9.0 / scipy 1.18.0, seed 0):**
  LogReg vs GaussianNB, RepeatedStratifiedKFold(10×10): gap **+0.0098**, **naive t p=1.17e−5 (SIG)** vs
  **corrected resampled t p=0.188 (noise)**; Wilcoxon on the same folds **p=5.0e−5 (also over-rejects — the
  trap)**. Real-gap contrast (LR vs stump) survives both (p=3e−46 / 2.6e−11). Demšar: 4 models × 12
  datasets, avg ranks LogReg 1.08 / NB 2.08 / RF 3.42 / HistGBDT 3.42, Friedman p=4.0e−6, **Nemenyi
  CD=1.354** (LR–NB not sig; LR vs trees sig).
- **Two new reusable viz (standard #9):** `assets/paired-diff-viz.js` (per-fold dots + naive/corrected CI
  toggle → verdict flip) and `assets/cd-diagram-viz.js` (critical-difference diagram; click a model for its
  non-different clique). Headless `labs/_viz_check_l023.js` 12/12; **browser MCP still unavailable**.
- **Artifacts synced:** retrieval-pool +2 (`l023-corrected` [misconception], `l023-cd-diagram`); paper-deck
  +1 (`demsar2006`); misconceptions **M20**; thesis-dossier +1 (BAR, C3) + honest-bar clause #5 ("prove the
  gap is not noise"); `reference/glossary.html` +5 Q3 terms. `node labs/_check_pedagogy.js` clean.
- **Lab:** `labs/0023-statistical-comparison.ipynb` — Tier-C concept lab, built via `labs/_build_l023.py`.
  3 TODO (100 paired diffs + naive `ttest_rel` · **implement the corrected resampled t** + Wilcoxon trap ·
  Friedman + Nemenyi CD over 12 datasets) + stretch (5×2cv, Cohen's d). Student blank (9 `____`, 0 outputs);
  solution executed clean & gitignored. Manifest → 23; all labs re-rendered.
- **Env note:** `.venv` already present + functional this session (sklearn 1.9.0, numpy 2.5.0, scipy 1.18.0,
  Node v20) — no uv/venv rebuild needed. The env-setup preinstall finally took.
- Next: Lesson 024 (The Grinsztajn benchmark — Grinsztajn 2022 §1–4; run one dataset), opening the
  Grinsztajn arc 024–027.

## Session 25 — 2026-07-15

- **Lesson 024 published** — The Grinsztajn benchmark (curriculum lec 024, **Grinsztajn, Oyallon &
  Varoquaux 2022**, arXiv `2207.08815`), **§1–4** — opens the Grinsztajn arc 024–027. Single skill:
  read/reproduce the benchmark **protocol** — a random-search **budget curve** (not one tuned
  number), **per-dataset normalization**, and the **dataset-selection criteria** that make the
  tree-vs-DL verdict fair. §5 (three biases) deferred to L025–027. Record:
  [[learning-records/0060-lesson-024-published.md]]. (Run headlessly by the publish loop; no user
  interaction this session.)
- **arxiv MCP unavailable** (headless publish env) → grounded in the published Grinsztajn arc (L019
  already covered §5 biases via user-arxiv) + the paper's known §3/§4 methodology; cited arXiv abs +
  the LeoGrin/tabular-benchmark repo.
- **Verified live (`labs/_verify_l024.py` + executed solution, real OpenML credit-g, seed 0, 30
  configs × 40 orderings):** GBT (HistGradientBoosting) above MLP at **every** budget — default
  (k=1) **0.7906 vs 0.7286** (gap **+0.0620**), fully tuned (k=30) **0.7850 vs 0.7700** (gap
  **+0.0150**); tuning helps the MLP more but the gap narrows-not-closes. Honest small-data artifact
  kept: GBT curve dips at large budget (validation overfitting on 200 val rows → why "not too small"
  is a criterion; callback L004/L017). Normalization worst→0, best→1, ordering preserved.
- **Two new reusable viz (standard #9):** `assets/benchmark-budget-viz.js` (Fig-1 budget curve,
  log-x, raw↔normalized toggle + budget slider, real credit-g data) and
  `assets/dataset-funnel-viz.js` (§3 selection funnel, click-a-stage → criterion + rationale, counts
  illustrative). Headless `labs/_viz_check_l024.js` 11/11; **browser MCP still unavailable** →
  headless-only, as L021–L023.
- **First Tier-A training lab of Q3:** OpenML **reachable** this session (unlike L021/L022 egress
  blocks). `labs/0024-grinsztajn-benchmark.ipynb` uses `relkit.load_tier_a("credit_g")` (incremental
  rule) + reproduces the protocol. 3 TODO + stretch; crucial fragment = the best-so-far budget curve
  (select by valid, report test). Student blank (7 `____`, 0 outputs); solution executed clean &
  gitignored. Built via `labs/_build_l024.py`.
- **Artifacts synced:** retrieval-pool +2 (`l024-budget-curve` [misconception], `l024-normalization`);
  paper-deck +1 (`grinsztajn2022-benchmark`, distinct from L019's `grinsztajn2022` biases card);
  misconceptions **M21**; thesis-dossier +1 (BAR+FOR, C3/C1 — GBT default raises the bar, but the
  contest lives inside the single-table world the thesis attacks); `reference/glossary.html` +6 Q3
  terms. Manifest → 24; all labs re-rendered. `node labs/_check_pedagogy.js` clean.
- Next: Lesson 025 (Inductive bias: smoothness — Grinsztajn 2022 §5.2), continuing the Grinsztajn arc.

## Session 26 — 2026-07-16

- **Lesson 024 marked complete** on the user's word ("lesson 24 done") — no EXIT text pasted, no rubric
  score, per the L017–L022 precedent. Record: [[learning-records/0061-lesson-024-complete.md]].
- **Lesson 025 published** — Inductive bias: smoothness (curriculum lec 025, **Grinsztajn 2022 §5.2**,
  Finding 1; theory backing Rahaman 2019 spectral bias, arXiv `1806.08734`). First of the three
  mechanism lessons in the Grinsztajn arc. Single skill: explain the **smoothness/spectral bias** (MLPs
  fit low-frequency structure first, over-smooth irregular targets; trees are piecewise-constant and
  follow the jags) and the **target-smoothing experiment** that proves it. Record:
  [[learning-records/0062-lesson-025-published.md]].
- **Verified live (`labs/_verify_l025.py`, sklearn 1.9.0):** 1-D — MLP/tree MSE ratio **5.30× (raw) →
  0.19× (smooth)**; multi-D gap experiment (mean of 5 seeds) — GBT **0.938** vs MLP **0.717** at h=0
  (gap **+0.22**) collapsing to ~0 by h=1.0 as the target's variance drops to 19%. Gap tracks the
  variance removed, not h — the mechanism signature. Lab solution (single seed): gap **+0.332 → −0.033**.
- **Two new reusable viz (standard #9, one per beat):** `assets/smoothness-fit-viz.js` (1-D target +
  tree staircase + MLP fit, target-smoothing slider + MSE readout) and `assets/smoothness-gap-viz.js`
  (the §5.2 gap-vs-smoothing curve + "variance kept" annotation). Both driven by real verify-script
  numbers. Headless `labs/_viz_check_l025.js` 12/12; **browser MCP still unavailable** → headless-only,
  as L021–L024.
- **Lab** `labs/0025-inductive-bias-smoothness.ipynb` — **Tier C** (synthetic; mechanism isolation, since
  no real dataset can toggle its own irregularity). Crucial fragment = implement the Gaussian **target
  smoother**; then sweep smoothing and watch the gap close. 3 TODO + stretch; student blank (8 `____`,
  0 outputs); solution executed clean & gitignored. Built via `labs/_build_l025.py`.
- **Artifacts synced:** retrieval-pool +2 (`l025-smoothness` [misconception], `l025-piecewise`);
  paper-deck +1 (`grinsztajn2022-smoothness`); misconceptions **M22** ("trees are just more powerful" →
  inductive-bias mismatch); thesis-dossier +1 (BAR+FOR, C3/C1); `reference/glossary.html` +3 Q3 terms.
  Manifest → 25; all labs re-rendered. `node labs/_check_pedagogy.js` clean.
- Next: Lesson 026 (Inductive bias: rotation — Grinsztajn 2022 §5.3; the rotation experiment).

## Session 27 — 2026-07-16

- **Lesson 025 complete** — "lab/lesson 25 done" (no EXIT ticket pasted → no rubric score; per the
  L017–L024 precedent). Record: [[learning-records/0063-lesson-025-complete.md]].
- **Lesson 026 published** — Inductive bias: rotation (curriculum lec 026, **Grinsztajn 2022 §5.4**,
  Finding 3; theory backing **Ng 2004**, ICML). Second of the three mechanism lessons. Single skill:
  explain **rotational invariance** (Ng 2004) — a tree is NOT invariant (axis-aligned splits tied to the
  original basis), an MLP/ResNet IS (`W·(Qx)=(WQ)·x`) — why that is a liability where columns carry
  individual meaning, and the experiment that proves it (a lossless random rotation collapses the tree,
  leaves the MLP unmoved, **reverses the ranking**). Record: [[learning-records/0064-lesson-026-published.md]].
- **Citation fix (rotation is §5.4, not §5.3):** prior records (L024-complete, L025) had the two
  remaining Grinsztajn findings' section numbers swapped. Verified against arXiv (ar5iv HTML), HAL,
  NeurIPS proceedings, and OpenReview: **§5.3 = uninformative features (Finding 2)**, **§5.4 = rotation
  (Finding 3)**. L026 cites §5.4; fixed L025's one-line forward reference. Lesson *order* unchanged
  (rotation L026 → uninformative L027), since it's baked into L025's quiz 3 and is a sound order (rotation
  is the mechanism Ng's theorem uses to explain L027's uninformative-feature fragility). arxiv MCP
  unavailable in this cloud env → grounded via direct web fetch of the four paper copies.
- **Verified live (`labs/_verify_l026.py` + executed solution, synthetic axis-aligned task, sklearn
  1.9.0):** rotation experiment (mean of 5 seeds) — Tree **0.987→0.747** (−0.240), GBT **0.997→0.824**,
  RF **0.994→0.812**, **MLP 0.862→0.869 (+0.008, invariant)**; original gap +0.126 → rotated −0.122 (the
  ranking **reverses**). Lab solution (single seed): tree 0.973→0.727, MLP 0.838→0.858 (+0.020), gap
  +0.135 → −0.131; Q orthogonal, distances preserved. All CHECK + EXIT clean.
- **Two new reusable viz (standard #9, one per beat):** `assets/rotation-splits-viz.js` (geometry:
  axis-aligned quadrant with two straight tree splits vs the rotated wedge's red staircase; MLP boundary
  rotates with the data) and `assets/rotation-gap-viz.js` (grouped bar chart Tree/GBT/RF/MLP original vs
  rotated + ranking-reversal connectors). Headless `labs/_viz_check_l026.js` 15/15; **browser MCP still
  unavailable** → headless-only, as L021–L025.
- **Lab** `labs/0026-inductive-bias-rotation.ipynb` — **Tier C**. Crucial fragment = implement the
  random-rotation operator (`Q` from `np.linalg.qr`, same `Q` on train+test) + verify it's a true
  rotation; then fit tree/GBT/MLP on original vs rotated and watch the ranking flip. 3 TODO + stretch
  (break invariance with a KBins embedding; rotation-friendly linear target). Student blank (10 `____`,
  0 outputs); solution executed clean & gitignored. Built via `labs/_build_l026.py`.
- **Artifacts synced:** retrieval-pool +2 (`l026-rotation` [misconception], `l026-invariance`);
  paper-deck +1 (`grinsztajn2022-rotation`); misconceptions **M23**; thesis-dossier +1 (BAR+FOR, C3/C1);
  `reference/glossary.html` +4 Q3 terms (rotational invariance, natural basis, random rotation, Ng's
  bound). Manifest → 26; all labs re-rendered. `node labs/_check_pedagogy.js` clean.
- **Env note:** no `python3-venv`/uv preinstalled again; installed `uv` via curl, built lean `.venv`
  (sklearn/numpy/scipy/jupyter — no boosters; Tier-C lab). Env-setup agent should preinstall the lab
  venv so future sessions skip this (recurring since Sessions 22–23).
- Next: Lesson 027 (Inductive bias: uninformative features — Grinsztajn 2022 **§5.3**, Finding 2), the
  last mechanism lesson of the Grinsztajn arc, linked to L026 by Ng's theorem.

## Session 30 — 2026-07-18

- **Lesson 029 complete** — user said "lesson 29 done" (no EXIT ticket → no rubric score, per the
  L017–L028 precedent). Record [[learning-records/0071-lesson-029-complete.md]].
- **Lesson 030 published — Q3 checkpoint: Write a Benchmark Report** (curriculum lec 030, **Grinsztajn
  2022 full**). A capstone, not a new concept (like L010/L020). Single skill: assemble the whole Q3 arc
  (L021–L029) into one **defensible one-page benchmark report** — deployment-matched split + leakage audit
  + budget curve over {tuned GBDT, honest neural baseline, AutoML bar} + corrected resampled t-test +
  inductive-bias explanation → an honest verdict, **including "no significant winner."** Record
  [[learning-records/0072-lesson-030-published.md]]. **Fourth application of standard #17** (thoroughness):
  the Q2 fair-comparison contract's five items restated + Q3's three additions each defined from scratch.
- **Verified live** (`labs/_verify_l030.py` + executed solution, credit_g Tier A): budget curve GBDT
  0.804→0.809, MLP **0.819→0.805** (tuning *dipped* — small-validation overfit, disclosed as a finding);
  paired **5×5 CV** GBDT **0.780** vs MLP **0.772**, gap **+0.0081**, naive t p=0.218, **corrected
  resampled t p=0.643 → not significant (a tie within noise)**. The honest tie *is* the deliverable — a
  bigger mean is not a win (L023). Lab solution CHECK + EXIT all clean; numbers match the lesson exactly.
- **No new viz** — consistent with L010/L020 checkpoint precedent (a checkpoint introduces no new
  mechanism; standard #9's "one viz per *new* mechanism" is satisfied by tables + reused `checklist.js`;
  Q3 mechanism viz live in L021–L029 and are linked). First checkpoint authored under the full pedagogy
  widget suite (#10–#14): warm-up, predict, teach-back, 3 quizzes, 10-item benchmark-report rubric.
- **Lab** `labs/0030-q3-checkpoint.ipynb` — Tier A (credit_g), `_build_l030.py`. Two crucial fragments are
  Q3 *evaluation* skills, not model internals: Task 1 = L024 budget-curve select-by-validation
  (`best_va, best_te = va, te`); Task 2 = L023 corrected resampled t (`(1/n + 1/(k-1)) * var_d`); Task 3 =
  assemble the report (leakage audit + honest verdict). Student blank (6 `____`, 0 outputs); solution
  executed & gitignored. **CHECK subtlety fixed:** the best-so-far TEST curve is *not* monotone (best-by-
  validation can pick a lower-test config) — the dip is the point, so the CHECK no longer asserts
  monotonicity.
- **Neural baseline = sklearn `MLPClassifier`** (torch-free, portable) standing in for the L028 ResNet —
  same inductive biases; the checkpoint tests the *report*, not the architecture. **torch NOT installed**
  this session (contradicts Session 29's note — the venv did *not* carry torch/xgboost forward reliably;
  xgboost present, torch absent). The Q3 checkpoint deliberately avoids torch so it runs anywhere.
- **Artifacts synced:** manifest regenerated → 30 entries (L030 `checkpoint`); `labs/html/0030-*.html`
  rendered; `retrieval-pool.js` +2 (`l030-tie` misconception, `l030-report`); `misconceptions.md` **M27**
  ("a benchmark must crown a winner" → a correctly-established tie is valid); `thesis-dossier.md` +1 line
  (L030, BAR+FOR, C3/C4) and **verdict updated to "after L030 / Q3"** (full honest instrument assembled;
  tie + L029 ⇒ single-table search/architecture returns near-exhausted, upside is representational);
  `glossary.html` +3 (Benchmark report, Model info sheet, Statistical tie). No new paper card (Grinsztajn
  already carded). `node labs/_check_pedagogy.js` clean.
- Next: Lesson 031 — **Q4 opener** (Embeddings for categoricals; entity embeddings, Guo & Berkhahn 2016;
  target-encoding pitfalls), beginning the Q4 consolidation + bridge to neural tabular. Q3 (L021–L030)
  closes once the user completes the L030 report.

## Session 29 — 2026-07-17

- **Lesson 028 complete** — user said "lesson 28 done" (no EXIT ticket → no rubric score, per the
  L017–L027 precedent). Record [[learning-records/0069-lesson-028-complete.md]]. (Housekeeping held this
  time: wrote both the complete record and the L029 publish record in the same session.)
- **Lesson 029 published** — Manual FE vs AutoML (curriculum lec 029, **Feurer, Klein, Eggensperger,
  Springenberg, Blum & Hutter 2015**, Auto-sklearn, NeurIPS 2015 — §2 CASH + §3 the two extensions).
  Background: Thornton 2013 (Auto-WEKA, the CASH framing) + Caruana 2004 (ensemble selection). Single
  skill: know what AutoML automates — **CASH** (jointly search algorithm + hyperparameters, select by
  validation) via Bayesian opt (SMAC), **meta-learning warm-start**, **automated ensemble construction**
  — and run the fair AutoML-vs-tuned-XGB fight; AutoML **ties** a tuned GBDT (buys automation, not
  accuracy) and never touches the representation. Record
  [[learning-records/0070-lesson-029-published.md]]. **Third application of standard #17** (thoroughness)
  — full vocabulary section (AutoML, HPO, algorithm selection, CASH + argmin, surrogate/SMAC,
  meta-features/warm-start, ensemble selection, manual FE) all from first principles.
- **Three reusable viz** (standard #9, one per beat): `assets/cash-search-viz.js` (REAL 40-iter CASH
  trace on credit_g — dots by algorithm + best-so-far step 0.796→0.817 + ★ winner; filter by algorithm),
  `assets/ensemble-select-viz.js` (Caruana greedy ensemble vs single best; toggle shows the 3-algo blend
  + the +0.007 test gain), `assets/automl-bakeoff-viz.js` (default/tuned XGB/AutoML with ±sd whiskers;
  tuning is the jump +0.031, AutoML ties −0.002). Headless `labs/_viz_check_l029.js` 15/15; **browser MCP
  still unavailable** → headless only.
- **Verified live (`labs/_verify_l029.py`, credit_g Tier A):** CASH search (seed 0, budget 40) visits all
  4 algorithms, best-val 0.796→0.817 (winner HistGB, single-best TEST 0.824); greedy ensemble (10 members
  across 3 algos) TEST **0.831** (+0.007, free). Bake-off (5 seeds) ROC-AUC: default XGB **0.775** → tuned
  XGB **0.806** (**+0.031**, the real payoff is tuning at all) ≈ tiny AutoML **0.803** (**−0.002**, bands
  overlap = a tie). `labs/_dump_l029_trace.py` produced the real per-iteration trace for cash-search-viz.
- **Auto-sklearn NOT installed** (Linux/version-fragile); the demo/lab reproduce its *mechanisms* on
  sklearn + xgboost with **random search standing in for SMAC** (installable; Bergstra & Bengio 2012
  justify it). Mechanism taught = CASH + select-by-validation + Caruana ensemble, not the optimizer.
- **Lab** `labs/0029-manual-fe-vs-automl.ipynb` — Tier A (credit_g). Crucial fragment (Task 1) = the
  greedy ensemble-selection pick (`cand = (ens_sum + val_probs[j])/(n_added+1)`); Task 2 = CASH selection
  (`argmax` over validation) + single-vs-ensemble; Task 3 = bake-off (blank = tuned-XGB keep-best-val).
  Student blank (4 `____`, 0 outputs); solution executed clean & gitignored (default 0.788 < tuned 0.804 ≈
  AutoML 0.810). Manifest → 29; `labs/html/0029-*.html` rendered.
- **Env note:** `.venv` from Session 28 already had torch/xgboost/sklearn 1.9.0 — **no bootstrap needed**
  this session (the recurring "preinstall the lab venv incl. CPU torch" ask from Sessions 22–28 paid off).
- Next: Lesson 030 — **Q3 checkpoint** (Grinsztajn 2022 full; write a 1-page benchmark report),
  consolidating the whole Q3 arc (L021–L029).

## Session 28 — 2026-07-17

- **Lesson 027 complete** — user said "lesson 27 done" (no EXIT ticket → no rubric score, per the
  L017–L026 precedent). **Backfilled the missing L027 records:** the L027 publish session (merged PR
  #11) shipped the lesson/lab but never wrote a learning record or NOTES entry — reconstructed as
  [[learning-records/0066-lesson-027-published.md]] + [[learning-records/0067-lesson-027-complete.md]]
  from the committed lesson. (Housekeeping note for future sessions: write the record in the same
  session you publish.)
- **Lesson 028 published** — MLP & ResNet tabular baselines (curriculum lec 028, **Gorishniy, Rubachev,
  Khrulkov & Babenko 2021**, arXiv `2106.11959`, §3.2 + §5; residual idea He et al. 2015, `1512.03385`).
  Pivots Q3 from *why trees win* into *building the honest neural contestant*. Single skill: build the
  tabular ResNet (embed → pre-activation residual blocks → head), explain why the **skip connection**
  (`x + f(x)`, free identity) fixes the **degradation problem**, and know a *tuned* MLP/ResNet is the
  honest neural baseline a GBDT (and an RDL result) must beat. Record:
  [[learning-records/0068-lesson-028-published.md]]. **Second application of standard #17** (thoroughness)
  — full vocabulary section defining linear layer, ReLU, BatchNorm, dropout, epoch/minibatch/Adam,
  residual connection, degradation problem, all from first principles.
- **FIRST PyTorch lesson/lab.** Installed **torch 2.13.0+cpu** into `.venv` and added `torch>=2.2` to
  `requirements-labs.txt`. This is the neural era — Year 2+ (and the RDL stack) all use PyTorch.
- **Three reusable viz** (standard #9, one per beat): `assets/resnet-block-viz.js` (block anatomy, skip
  ON/OFF toggle → `+` node appears/disappears), `assets/depth-trainability-viz.js` (test-acc vs depth,
  plain degrades / ResNet holds; "show training accuracy" toggle exposes the degradation), and
  `assets/baseline-bakeoff-viz.js` (MLP/ResNet/GBDT on credit_g with ±sd; AUC/acc toggle). Headless
  `labs/_viz_check_l028.js` 18/18; **browser MCP still unavailable** → headless only.
- **Verified live (`labs/_verify_l028.py` + executed solution):** depth trainability (synthetic, same
  arch skip on/off, BatchNorm on both) — plain **test** 0.917→0.866 and, decisively, plain **train**
  1.000→0.927 over depth 1→32 (train falls ⇒ **degradation/optimization, not overfitting**), while the
  ResNet holds ~0.90 test / ~1.00 train. Honest **NOT vanishing gradients** framing (BatchNorm present;
  the skip's free identity is the mechanism — He et al. distinguish the two). Bake-off on **credit_g**
  (Tier A, real OpenML, 5 seeds): ROC-AUC MLP **0.752** ≈ ResNet **0.743** (tie), GBDT **0.793** ahead —
  **no universal winner**; GBDT wins small categorical data (consistent with L024–L027).
- **Honest-baseline discipline reinforced (Gorishniy's real point):** many "SOTA" tabular-DL papers
  failed to beat a *properly-tuned* MLP/ResNet; the single-table bar an RDL win must clear is a tuned
  GBDT **and** a tuned ResNet. ResNet≠FT-Transformer (attention baseline deferred to Y2 Q1, where
  per-feature embeddings start to break the L026/L027 biases).
- **Lab** `labs/0028-mlp-resnet-baselines.ipynb` — Tier A (bake-off, credit_g via `relkit`) + Tier C
  (depth mechanism). Crucial fragment = `ResNetBlock.forward` (`return x + f(x)` vs `f(x)`). Lab uses a
  shorter 40-epoch budget so the degradation is *unmistakable* (plain train collapses 0.998→0.496 by
  depth 16); lesson/viz quote the gentler 60-epoch verify numbers — same mechanism, noted in the record.
  Student blank (7 `____`, 0 outputs); solution executed clean & gitignored. Manifest → 28; all labs
  re-rendered.
- **Env note:** no venv preinstalled again; uv-bootstrapped `.venv` + installed CPU torch (~1–2 min).
  **Env-setup agent should now preinstall the lab venv INCLUDING `torch` (CPU)** — this is a recurring
  cost and torch is heavier than the sklearn stack; every Year 2+ lab will need it.
- Next: Lesson 029 (Manual FE vs AutoML — Feurer et al. 2015 Auto-sklearn, skim; compare tuned XGB),
  then L030 = Q3 checkpoint (1-page benchmark report).

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

- [x] Y1: Chen 2016 XGBoost §2 assigned in Lesson 014 (Eq 5–7 implemented in lab)
- [x] Y1: Ke 2017 LightGBM §3 (GOSS) + §4 (EFB) assigned in Lesson 015 (GOSS implemented in lab)
- [x] Y1: Prokhorenkova 2018 CatBoost §3 (ordered TS) + §4 (ordered boosting) assigned in Lesson 016 (ordered TS implemented in lab)
- [x] Y1: Bergstra & Bengio 2012 §1 (random search / low effective dimensionality) assigned in Lesson 017 (grid vs random + nested CV implemented in lab)
- [x] Y1: Wolpert 1992 §1–3 (stacked generalization / out-of-fold meta-features) assigned in Lesson 018 (OOF blend + leak contrast + StackingClassifier in lab)
- [~] Y1: Grinsztajn 2022 — abstract + §1 (three inductive biases) previewed in Lesson 019; **§3–4 (benchmark construction + random-search budget-curve protocol) assigned in Lesson 024** (single-dataset protocol reproduction on credit-g in lab); §5.2 (smoothness) assigned in Lesson 025, §5.4 (rotation, incl. Ng 2004) assigned in Lesson 026, and §5.3 (uninformative features) assigned in Lesson 027 (add/remove-junk ablation + gate in lab) — Grinsztajn arc complete
- [x] Y1: Gorishniy et al. 2021 (`2106.11959`) — §3.2 (MLP + ResNet baselines) assigned in Lesson 028 (residual block `forward` implemented in the first PyTorch lab; depth-degradation + honest bake-off reproduced); He et al. 2015 (`1512.03385`) as the residual/degradation-problem backing
- [x] Y1: Feurer et al. 2015 (Auto-sklearn) — §2 (CASH) + §3 (meta-learning warm-start + ensemble selection) assigned in Lesson 029 (tiny AutoML built on sklearn+xgboost: CASH selection + Caruana greedy ensemble implemented; AutoML-vs-tuned-XGB bake-off reproduced — AutoML ties a tuned GBDT); Thornton et al. 2013 (Auto-WEKA, CASH framing) + Caruana et al. 2004 (ensemble selection) as backing
- [~] Y1/Y2: Rubachev 2024 (TabReD, `2406.19380`) — abstract + §1 + §5.4 previewed in Lesson 021 (random vs temporal splits; optimism gap synthetic demo in lab); full core read is Y2 lec 055
- [x] Y1: Kapoor & Narayanan 2022 (`2207.07048`) — abstract + §2 (8-type taxonomy) + §5 (civil-war reproduction) + §6 (model info sheet) assigned in Lesson 022 (illegitimate-feature collapse + FE-classification + model info sheet in lab)
- [x] Y1: Demšar 2006 (JMLR 7, no arXiv) — §3.2 (Wilcoxon) + §3.5 (Friedman + Nemenyi CD) assigned in Lesson 023; Nadeau & Bengio 2003 (corrected resampled t-test) + Dietterich 1998 (5×2cv/McNemar) as the single-dataset companions (corrected-t + Friedman/CD implemented in lab)
- [ ] Y1: Fey 2024 §1
- [x] Y1: Fey 2024 §1 assigned in Lesson 001

## Preferences

- **Lab environment:** `.venv/` at repo root via `bash labs/setup-env.sh` + [`requirements-labs.txt`](requirements-labs.txt); Jupyter kernel **Relational Labs (.venv)**; Cursor picks up `.venv/bin/python` via `.vscode/settings.json`.
- **Colab bootstrap (every lab):** Colab opens a lab as a lone `.ipynb` with no repo, so `relkit` + boosters are missing. Every lab's **first code cell** is the `@colab-bootstrap` (canonical source `labs/_colab.py`): on Colab it shallow-clones the repo, `pip install`s `requirements-labs.txt`, and `chdir`s into `labs/`; a no-op locally/Binder. Build scripts prepend `bootstrap_cells()`; `scripts/add_colab_bootstrap.py` injects it into any existing notebook (idempotent via the marker). After building/adding a lab, run the injector if the build script doesn't already prepend it, then re-render.
- **Labs as notebooks (from L006 on):** every lab ships as a Jupyter notebook in `labs/NNNN-<slug>.ipynb` following [`labs/LAB-TEMPLATE.ipynb`](labs/LAB-TEMPLATE.ipynb). Convention: **PROVIDED** cells = complete boilerplate; **TODO** cells = blanks (`____` / `# TODO`) only on the skill being practised; **CHECK** cells = auto-feedback assertions (don't edit); final **EXIT TICKET** cell prints the deliverable. Keep blanks focused on the one idea so working memory stays free. Pattern documented in `labs/README.md`.
- **Reproduction labs build incrementally (not yet active — note for later):** once we reach paper-reproduction labs (RelBench baselines, GBDT/RealMLP/TabM, RDL), each lab must build on the code already written in earlier labs rather than re-deriving from scratch — a cumulative, reusable codebase (shared data loaders, eval/CV harness, leakage-safe pipelines, metrics) that grows lab over lab. Likely promote shared code from notebooks into a small importable package (e.g. `labs/relkit/` or `src/`) and have reproduction notebooks import it. Goal: by the time results matter for the thesis, the baseline harness is battle-tested across many labs, not a pile of one-offs. Standalone *concept* labs (like L006) can stay self-contained; the incremental rule kicks in for reproduction/experiment labs.
- **Lesson length:** prefer longer, richer lesson HTML (~35–45 min reading + quizzes) over short skims; chunk with worked examples and multiple check-your-understanding blocks. **From L027 (2026-07-17): no length ceiling — thoroughness wins over brevity (standard #17 below).** Every term, symbol, formula, and mechanism *introduced* in a lesson must be explained in-lesson, from first principles, before it is used; do not assume recall of a definition just because an earlier lesson used it. Make the lesson as long as that requires.
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

### Q2 retrospective (2026-07-08) — standards from the Q2 checkpoint

Recorded after the Q2 checkpoint (L020). Full rationale: [[learning-records/0052-q2-retrospective-feedback.md]].

9. **Multiple visuals per lesson (from 2026-07-08):** default to **one visual per distinct mechanism / claim / "strength"** a lesson teaches, not one viz per lesson. Every mechanistic beat that passes the `lesson-visuals` decision tree gets its own inline viz next to the prose that explains it. Split a mode-toggle widget into separate adjacent viz when the modes are *different concepts* (keep a toggle only when modes are the same mechanism under a knob). A section with only a static results table where the mechanism is visualizable is an under-served beat — add the viz. Verify **every** viz in the browser, not just the first. See `lesson-visuals` skill § How many visuals per lesson. L019 is the recorded anti-pattern (three biases, ~1.5 visuals).

10. **Spaced-retrieval warm-up opens each lesson (from 2026-07-08):** every lesson opens with a `RetrievalBank.mount(..., { upTo: <thisLesson>, count: 3 })` warm-up (`assets/retrieval-bank.js` + `assets/retrieval-pool.js`). Draws only from *earlier* lessons (spacing), interleaves concepts/quarters, and is Leitner-scheduled in `localStorage` (missed items return sooner, mastered ones later). When a lesson ships a durable testable idea, add a pool item with a **stable `id`** (never renumber). Storage strength > fluency — see `lesson-pedagogy` skill. Reference: L019.

11. **Prediction-before-reveal on non-obvious results (from 2026-07-08):** before a lesson reveals a genuinely non-obvious number/outcome (bake-off, "who wins", surprising result), make the learner commit a prediction first with `Predict.mount` (`assets/predict.js`) — the pretesting effect. Use once or twice per lesson on the surprising results, not on every table. Reference: L019 (the "clean features → MLP wins" result).

12. **Misconceptions ledger kept in sync (from 2026-07-08):** `misconceptions.md` is the living list of wrong beliefs surfaced in labs/warm-ups/check-ins. When one surfaces: add a row (wrong · correct · lesson · status), add a matching `"misconception": true` item to `assets/retrieval-pool.js` so it re-enters the spaced rotation, and mark it `retired` after ≥2 correct spaced sessions (Leitner box ≥ 3). Verify the assets with `node labs/_check_pedagogy.js`.

13. **Thesis dossier updated each lesson (from 2026-07-08):** `thesis-dossier.md` is the skeptic-facing argument for the mission's bet (claims C1–C4, an evidence ledger, counter-evidence, skeptic objections, current verdict). After each lesson add one Evidence-Ledger line tagged FOR / BAR (raises the honest baseline) / AGAINST. **Never delete counter-evidence.** This is the artifact the "publish/ship to convince skeptics" success criterion is ultimately built from. See `lesson-pedagogy` skill § Cross-cutting artifacts.

14. **Teach-back once per lesson (from 2026-07-08):** every lesson has one `Teachback.mount` free-recall prompt (`assets/teachback.js`) on the load-bearing idea — the learner explains in their own words, then reveals a model answer + self-check points. Grade it when the learner pastes it to chat; a term explained cold graduates to `GLOSSARY.md`. Recognition ≠ explanation. Reference: L019 (rotational invariance).

15. **Two glossaries, two jobs (from 2026-07-08):** `reference/glossary.html` is the **authoritative** ubiquitous language — every lesson must be consistent with it; add a term when a lesson introduces it. `GLOSSARY.md` is the learner's **personal mastery log** (own words, only once explained cold). Do not conflate them.

16. **Paper flashcards for core papers (from 2026-07-08):** when a lesson assigns a core (★) paper, add a one-claim card to `assets/paper-deck.js` (stable `id`, one crisp claim). Reviewed on `flashcards.html` via `assets/flashcards.js` (Leitner self-rated recall) — keeps the paper-dense curriculum warm across years.

### Thoroughness standard (2026-07-17, from L027) — applies to L027 and all future lessons

17. **Explain everything introduced — thoroughness over brevity (from 2026-07-17):** the user asked that *everything introduced in a lesson be thoroughly explained, even if it makes the lesson longer*, and that this decision hold for all future lessons. Concretely, when authoring any lesson from L027 on:
    - **No new term/symbol/formula used unexplained.** The first time a lesson uses a term (even one from an earlier lesson — e.g. "spectral bias", "orthogonal matrix", "sample complexity", "MDI importance"), give a one-line plain-language definition inline or in a callout, before relying on it. Assume the reader is meeting it fresh. Do not gate understanding behind "see Lesson 0XX".
    - **Derive, don't assert.** When a formula or result appears (e.g. why an MLP is rotation-invariant, why Ng's bound is linear), walk through *why* in prose or a small worked step — not just the final expression.
    - **Unpack every experiment.** State what is held fixed, what is varied, what is measured, and why that isolates the mechanism, before showing the numbers. Read each results table row aloud in prose.
    - **Length is not a constraint.** Drop the ~35–45 min soft cap when the material needs more; a longer, self-contained lesson beats a shorter one that leans on unstated prior knowledge. Keep the chunking/worked-example/check-block structure so length stays digestible.
    - This *extends*, and does not replace, standards #1–#16 (still one skill per lesson, still visuals per mechanism, still the pedagogy widgets). Reference implementation: **L027**.
