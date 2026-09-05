# Teaching Notes

## Notebook quality standard — 2026-09-05

- User explicitly said L047 notebooks were lower quality than L046 and that lessons **NEED to be HIGH quality**. This is a standing requirement for every future lesson and its notebook, not a one-off visual request. Compare the complete learning experience with the strongest recent lessons before delivery.
- A notebook must teach independently: clear paper-to-code explanations, worked examples, diagrams beside the mechanism they explain, short readable implementation chunks, substantive live TODOs, diagnostic CHECKs, and interpretation prompts. File existence and cell count are not evidence of teaching quality.
- Embed portable diagrams so they are visible before execution. Display measured results as labeled tables/plots with uncertainty and interpretation; separate author-reference snapshots from the student's current outputs. Keep the supervised, pretraining, local-ablation, and paper-results scopes explicit.
- Execute the solution, verify the student implementation remains on the trained path, inspect generated figures, regenerate the prepared HTML, and check attachment/link integrity. Record browser limitations honestly. Persist this checklist in the lab-authoring skill.
- L047 revision verified: 31/31 solution checks; the three-dataset comparison and context probes exactly match the recorded reference (109.4 s comparison). Nine attached figures and four runtime plots inspected; prepared HTML image/link checks pass. The notebook Bank smoke reproduces the recorded 0.742204 pipeline result, remains INCOMPARABLE, and reuses the completed seed when the gate is repeated. Full presets and full pretraining remain NOT_RUN.

## Lesson 047 visual revision — 2026-09-05

- User found the original L047 visuals weaker than preceding lessons. Text cards and a single bar did not sufficiently expose the mechanisms. Match the earlier diagram quality: visible tensor structure, connections, matrices, stage highlighting, and controls coupled to calculations.
- Rebuilt L047 with five responsive SVG figures: attention axes, full architecture, companion-weight intervention, CutMix/mixup, and contrastive/denoising objectives. Keep synthetic illustrations explicitly separate from measured results.
- Rendered 40 desktop/mobile SVG states with installed librsvg and visually inspected representative outputs; glyph bounds, label overlaps, card containment, and numerical checks pass. Browser-level verification remains unavailable; this is a vector-renderer check, not a browser pass.

## Session 46 — 2026-09-05

- User reports lesson 46 done; no EXIT evidence pasted, so completion is self-reported and unscored (LR-0108).
- Lesson 47 follows curriculum SAINT / intersample attention, with supervised from-scratch code, a visible notebook, and Bank paper-results operators.
- User re-emphasized paper reproducibility and high-quality explanation. Apply standards #17/#24/#25 with explicit paper-vs-code audits; no predetermined winning model.
- Verified released stage forward and input gradients exactly on transplanted weights. PDF equations and numeric embeddings differ from released code; record source choices rather than treating them as identical.
- Evaluation context belongs in the ledger: companion membership changes predictions even in eval mode. Local three-table/three-seed ablation is not the paper benchmark; Table 3 pretraining remains untested.
- The reused CD widget incorrectly assumed significance and left long labels outside its SVG; fixed conditional wording and label margins, verified in the L047 headless checks.
- Browser download was not approved, so rendered visual verification remains outstanding. Headless checks must not be described as a browser pass.

## Session 45 — 2026-08-28

- **NEW standard #25 — paper-results scale-up + visible implementation** (user `/teach`): next
  paper-mirror lessons must try *harder* to reproduce the paper's **results**, not only the
  architecture. If local scale cannot train it, ship a **Modal** script *and* a **Colab**-gated
  cell so the student can train and compare. This is the **required next step after the lab**,
  not optional stretch. Record: [[learning-records/0105-paper-results-visible-scale-up.md]].
- **Why:** L043–L045 already mirrored the architecture and were honest about downscaling, but
  the paper's table was left as "out of scope / stretch". A downscaled ranking (TabNet last on
  four tiny tables; NODE last vs CatBoost; TabTransformer 0/3 vs CatBoost) is a *different
  experiment*. Mixing it with the paper's claim is how you learn the **wrong conclusion**.
- **Visible implementation:** the paper's model must appear as **inlined PROVIDED cells** in the
  notebook (`relkit.paper_repro.inline_source`), not only `from relkit.X import TheModel`. relkit
  stays the canonical file for Modal / `_verify`; the notebook is a readable copy that **keeps
  the student's TODO functions** (those names are skipped on inline).
- Applied now to **L043 / L044 / L045** (the paper-mirror units already shipped). Future paper
  lessons inherit the checklist item in `lab-authoring`.

## Session 44 — 2026-08-22

- **L043 retrofitted with architecture diagrams** on user request ("pure equations make it hard to read
  and understand"). Record: [[learning-records/0103-l043-diagrams-and-browser-verification.md]].
  L043 had **one** viz for **five** mechanisms, and it sat *after* every equation — the recorded L019
  anti-pattern (standard #9) resurfacing in Y2. Three new components, each inline with its prose:
  `tabnet-arch-viz` (Fig. 4a encoder, 7-stage stepper, each stage printing the equation its block
  computes, placed **before** the mechanism subsections), `sparsemax-viz` (Algorithm 1 as a water line:
  `τ` solved for, the shaded area always totalling exactly 1, softmax's never-zero contrast, and the mask
  entropy `L_sparse` penalises), `tabnet-fblock-viz` (Fig. 4c: shared vs step-dependent GLU layers, the
  three `√0.5` merges, the split, plus an inset opening one layer).
- **Draw from the code, not from the figure — a diagram is a consistency check on the prose.** Laying the
  boxes out against `labs/relkit/tabnet.py` forced every tensor to be named, and one had no source in the
  lesson: **`a[0]` comes from an extra feature transformer over the *unmasked* features at step 0, whose
  `d` half is discarded.** Fig. 4a shows it, the equations never mention it, and it is what people omit
  when reimplementing from the paper. Now a defbox. Do this on every equation-heavy retrofit.
- **The mandatory browser check IS runnable here — stop writing "Browser MCP unavailable".** There is a
  `/usr/local/bin/google-chrome`, and `npm install puppeteer-core` in **`/tmp`** drives it while the
  workspace keeps its zero-dependency, no-`package.json` posture. It immediately caught three things the
  fake-DOM harness cannot see: an SVG **z-order bug** (an edge label painted behind a box, since arrows
  are drawn before boxes), two **label collisions**, and a **bar scale** that used ~30% of its vertical
  space at the default slider setting. Method + screenshot recipe in the record.
- **New assertions worth copying into every future `_viz_check_lNNN.js`:** geometry checks on hand-laid-out
  diagrams (every box/label inside the viewBox, **no two boxes overlapping**), exact-arithmetic checks on
  the operator being taught, clamping on public setters that feed a fixed pixel scale, and readout↔equation
  coupling per stage. Also gave that file's fake DOM `firstChild`/`removeChild` so redraws really clear.
- Verification: `labs/_viz_check_l043.js` **81 → 222 checks**, all pass; `labs/_check_pedagogy.js` 40/40;
  browser pass clean at 900 px and 375 px for all four widgets. Fixed in passing: LR-0101 cited the
  attentive transformer as Fig. 4b; per arXiv:1908.07442v5 it is **4d** (4b is the decoder).

## Session 43 — 2026-08-08

- **Lesson 043 published with lab — TabNet (sequential attention)**, the first novel architecture put
  through L042's baseline-first bar, and the first full application of standard **#24** (paper-mirror).
  Record: [[learning-records/0101-lesson-043-published.md]].
- **The bar bit, and that is the lesson.** From-scratch TabNet under one shared frame on 4 small tables:
  mean ranks **TabNet 2.50** vs **MLP 1.75** / **ResNet 2.00** (GBDT 3.75), Friedman **p = 0.127**. Say it
  in two separate sentences (now **M55**): p > 0.05 licenses only *"cannot distinguish on this sample"* —
  never "significantly worse", and never "equivalent" — **but** the burden of proof is the *new* model's,
  so it did **not clear the bar**. The paper's own **Appendix A (KDD)** ties/trails XGBoost + CatBoost.
- **Sparsity ≠ sequentiality** (**M54**), the most compressible-but-wrong summary of TabNet. sparsemax buys
  exact zeros (a real *selection*, since softmax's `exp(z) > 0` can never switch a feature off); the
  **prior scale** `P[i] = ∏(γ − M[j])` is the memory that makes the attention *sequential*. At γ = 1 the
  leftover budget is `1 − M`, so a fully-used feature is banned from all later steps.
- **Interpretability was tested, not asserted** (**M53**) — on the paper's *own* generators, because on real
  data no attribution is falsifiable. **Syn2** (global relevance): clean success, top-4 exact, 76.8% of
  `M_agg` mass on the truth. **Syn4** (instance-wise): only **partial** — switch feature found (0.118),
  mass moves the right way, but 15.6% vs 97.9% of rows favour their own group. The paper needed **10M**
  rather than 10k samples for its sharp Fig. 5 masks. Masks are *evidence to validate*, not free
  explanation.
- **An unexplained discrepancy is on the record, not smoothed over.** From-scratch TabNet *outscored*
  `pytorch_tabnet` end-to-end (credit_g 0.748 vs 0.694; diabetes 0.824 vs 0.766, tol 0.04). Two hypotheses
  were tested in committed harnesses: *training length* (a longer leash left credit_g at **0.686** — no
  help — but moved diabetes to **0.785**, |Δ| 0.039, just inside tolerance) and *LR schedule* (the paper's
  StepLR decay on the reference gave credit_g **0.680** — no help). So the **credit_g** gap is unexplained;
  scope the claim to that table rather than saying nothing closed. Direction is the safe one for #22, but it
  is still a gap, and it does *not* license "ours is better". Definitive next test = weight transplant +
  `torch.allclose` (shipped as a lab stretch task).
- Verification: `labs/_check_l043.py` **22/22**, `labs/_viz_check_l043.js` **81/81**,
  `labs/_check_pedagogy.js` **40/40**, solution notebook executed end-to-end (**25/25** CHECKs, 126 s CPU).

## Session 42 — 2026-08-08

- **Lesson 042 complete** — user said "lesson 42 done" (no EXIT numbers pasted → no hostile-reader rubric
  score, per the L017–L041 precedent). Record: [[learning-records/0099-lesson-042-complete.md]].
- **GitHub Pages publish fixed.** The `deploy` job's `wretry` wrapper *was* the bug: a Pages deployment is
  keyed on the commit SHA, `deploy-pages` cancels it at its timeout when the backend stalls in
  `deployment_queued`, and the retry then recreated the cancelled record and failed instantly — 3 attempts
  exhausted, 12m22s red run on the doctrine commit. Removed the wrapper, added `paths-ignore` so
  bookkeeping-only commits don't spend a deployment, bumped the actions off Node 20. Record:
  [[learning-records/0100-pages-deploy-fix.md]].
- **Two publishing facts worth remembering:** the deploy `timeout` input is **clamped to 600000 ms** (a
  longer value is silently ignored), and a stalled deployment is recovered by **pushing a new commit**, not
  by re-running the same SHA.

## Session 41 — 2026-08-06

- **NEW standard #24 — paper-mirror doctrine (user directive this session):** future lessons/labs must
  **always try to mirror paper implementations from scratch** — not just the architecture, but also the
  **datasets / protocol** and **reproducibility discipline** — so we learn as much as possible *from the
  papers themselves*. Consolidates and elevates #18 (implement the paper), #22 (from scratch; libraries
  validate), #23 (multi-dataset rigor), and #20 (honesty/compute). Record:
  [[learning-records/0098-paper-mirror-from-scratch.md]]. Full rule in Preferences #24 + `lab-authoring`.
- **What "mirror" means here:** (1) implement the paper's load-bearing mechanism from scratch; (2) prefer
  the paper's own datasets/splits when affordable, else a documented substitute with an honest gap;
  (3) ship a reproducibility ledger (seeds, versions, protocol deviations, verify harness + results JSON).
- Does **not** replace mid-zone (#3), lab-with-lesson (#21), or compute escalation (#20) — it tells those
  standards *what* to build when a paper is the lesson's core source.
- Next: apply #24 on the next paper-implementation unit (post-L042).

## Session 40 — 2026-07-29

- **Lesson 039 complete** — user said "lesson 39 done" (no essay pasted → no hostile-reader rubric score,
  per L017–L038 precedent). Record: [[learning-records/0091-lesson-039-complete.md]].
- **Lesson 040 published — Year 1 Exit Exam** (curriculum lec 040, checkpoint). Single skill: regenerate
  a disclosed XGBoost baseline under a fair protocol, attempt one honest challenger, classify
  BEAT / TIE / FAIL against ±0.002 ROC-AUC noise, write Grinsztajn's three biases, STAND or REVISE the
  L039 claim. Record: [[learning-records/0092-lesson-040-published.md]].
- **TIE is a full pass.** Modal expected outcome on adult (L020 evidence of record). Soft-selling
  Δ=+0.001 as a beat = M48; demanding a must-win = M49.
- **Two new reusable viz (standard #9, one per beat):** `assets/exit-verdict-viz.js` (BEAT/TIE/EXPLAIN)
  and `assets/exit-gates-viz.js` (six protocol gates). Reused `biases-viz.js`. Headless
  `labs/_viz_check_l040.js` all pass; browser MCP unavailable → node only. No new bake-off.
- **Lab** `labs/0040-year-1-exit-exam.ipynb` — Tier A OpenML `adult` (public regenerable bar). Homework
  kept as stretch/`make verify` because the private submission is outside the repo; NOTES #19 discipline
  (noise bands, baseline parity) still governs the fork classifier.
- **Artifacts synced:** retrieval-pool +3; glossary +4; paper-deck +1 (`grinsztajn2022-exit`);
  misconceptions M48/M49; thesis-dossier +L040; RESOURCES Grinsztajn extended; manifest → **40**;
  L039 nav → L040.
- Next: user completes the exit lab; then Year 2 opens at Lesson 041.

## Session 39 — 2026-07-29

- **Lesson 038 complete** — user said "lesson 38 completed" (no written review pasted → no rubric score,
  per L017–L037 precedent). Record: [[learning-records/0089-lesson-038-complete.md]].
- **Lesson 039 published — Year 1 Synthesis Essay** (curriculum lec 039, **Grinsztajn et al. 2022 full**
  re-read). Penultimate Y1 unit before the exit exam. Single skill: compose a synthesis essay arguing
  **what trees beat and why** — three inductive biases, exhaustion cascade, flip conditions, join frontier,
  credibility coda — ending in a claim a hostile reviewer could grade. Record:
  [[learning-records/0090-lesson-039-published.md]].
- **Genre lesson, not a new bake-off.** No new verified numbers; every figure is evidence of record from
  L010 / L019–L038. Lab is a written essay (like L038), `labPath: null`.
- **Two new reusable viz (standard #9, one per beat):** `assets/y1-arc-viz.js` (Year 1 as four argument
  moves; 12 milestone chips) and `assets/trees-frontier-viz.js` (WIN / EXHAUSTED / FRONTIER zones). Reused
  `biases-viz.js`. Headless `labs/_viz_check_l039.js` all pass; browser MCP unavailable → node only.
- **Misconceptions M46** (synthesis = chronological recap) and **M47** (trees win because more powerful).
- **Artifacts synced:** retrieval-pool +3; glossary +7 Q4 terms; paper-deck +1 (`grinsztajn2022-synthesis`);
  thesis-dossier +L039 (FOR+BAR); RESOURCES Grinsztajn entry extended; manifest → **39**.
- Next: Lesson 040 — Year 1 exit exam (beat XGB on a flat task or explain why not).

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

## Session 35 — 2026-07-24

- **Lesson 034 complete** — user said "lesson 34 done" (no EXIT ticket → no rubric score, per the
  L017–L033 precedent). Record [[learning-records/0081-lesson-034-complete.md]].
- **Lesson 035 published — What Joins Destroy** (curriculum lec 035, **Fey et al. 2024 §1–2**, ★ preview) —
  the **intellectual pivot of Year 1** and the close of the L034–L035 thread. Single skill: enumerate the
  structure a flatten-then-aggregate pipeline discards (cardinality, event identity, temporal order,
  higher-order/multi-hop paths) and recognise an **aggregation collision**. Record
  [[learning-records/0082-lesson-035-published.md]]. **Ninth application of standard #17** (lossy vs lossless
  map, aggregation collision, cardinality, event identity, temporal order, multi-hop path, relational entity
  graph all defined first-principles).
- **Fey is a *position* preview, not an architecture** — standard #18 scope (stated in the lab intro) is to
  **demonstrate issue (4)** ("forcing data into a single table aggregates into lower-granularity features,
  losing fine-grain signal") as a runnable collision, not implement a model. Grounded the lesson in the
  paper's verbatim five-issue enumeration (downloaded 2312.04615 via arxiv MCP; §1–2). Message passing / REG
  architecture / RelBench deferred to Y3–Y4 (forward refs plain text).
- **One new reusable viz (standard #9, the single mechanistic beat):** `assets/flatten-loss-viz.js` — two
  customer subgraphs (Ada rising $10→$30→$50 / 3 products; Bo falling $50→$30→$10 / 1 product) collapse under
  JOIN+AGGREGATE to two **identical** rows `n=3/total=90/avg=30/max=50` (red "⚠ identical" banner); a "Reveal
  the lost structure" toggle shows trend ↗/↘ and distinct-product counts. Headless `labs/_viz_check_l035.js`
  **13/13**; **browser MCP still unavailable** (no chrome-devtools server) → node verification only,
  consistent with L021–L034.
- **Lab 035 = Tier C, pandas + sklearn, mechanism/demonstration lab** (runs in seconds). Deterministic DB,
  cutoff t = 2024-06-01, **all orders pre-t** (PIT was L034's job — this isolates the *loss*, not leakage).
  Crucial fragments: Task 1 flatten + confirm Ada/Bo collide; Task 2 fit `LogisticRegression` and prove
  P(churn)=**0.502 for both** though true labels are **0 vs 1** (identical input ⇒ identical output); Task 3
  recover structure — `spend_trend` (+40/−40, restores order) + `n_distinct_products` (3/1, restores
  identity). Stretch: a third customer Zoe collides again (the treadmill). Solution executed **clean** (3
  CHECK + EXIT + stretch); gitignored; rendered to `labs/html/0035-*.html`. 7 student blanks.
- **Thesis:** L035 turns "lossy by construction" into a *demonstration* — the collision proves the loss can
  be total (different entities → one feature row → same prediction), upstream of any model. Sharpest
  statement yet of C1; names the four discarded dimensions. **Honesty guard:** still a demonstration of
  *cost*, not a *win* — no result yet shows a graph model recovering the structure to beat the honest bar
  (L028–L030); that is the Y1-exit essay + Y3–Y4 burden. Dossier verdict updated to "after L035."
- **Artifacts synced:** retrieval-pool **+2** (`l035-collision` [misconception], `l035-thesis`); paper-deck
  **+1** (`fey2024-fecost`, keyed to L035; broad `fey2024` stays at L001); misconceptions **M36** (a better
  model can recover a lossy flatten → loss is before the model) + **M37** (add more aggregates → unbounded
  per-task treadmill; lossy ≠ leaky); thesis-dossier **+1** (FOR + BAR, C1/C2) + verdict;
  `reference/glossary.html` **+5** Q4 terms (lossy map, aggregation collision, fine-grain signal, structure a
  flatten discards, relational entity graph); `RESOURCES.md` enriched the Fey 2312.04615 entry into the L035
  primary-reading note. Manifest → **35**; `node labs/_check_pedagogy.js` clean; `_viz_check_l035.js` 13/13;
  `_viz_check_l034.js` regression clean.
- Next: Lesson 036 — **Revisit your homework pipeline** (audit the user's own `homework/report.md` against
  the Q1–Q3 leakage-spine checklist; reuse `checklist.js` + `pipeline-viz.js`; no new concept). **Prompt the
  user for their homework artifact at the start of that session** — L036 is personalised to their prior work.

## Session 34 — 2026-07-21

- **Lesson 033 complete** — user said "lesson 33 done" (no EXIT ticket → no rubric score, per the
  L017–L032 precedent). Record [[learning-records/0079-lesson-033-complete.md]].
- **Lesson 034 published — Relational Data Without RDL** (curriculum lec 034, **Kimball & Ross** dimensional
  modeling / star schema & joins). Single skill: read a 3-table schema (fact + dimensions, PK/FK) and write
  the join + point-in-time aggregation that flattens it into one design matrix at a chosen entity grain.
  Record [[learning-records/0080-lesson-034-published.md]]. **Eighth application of standard #17** (every
  term defined first-principles: relational database, entity, PK, FK, fact/dimension table, star schema,
  grain, one-to-many cardinality, join, aggregation, design matrix, PIT correctness).
- **Kimball is a modeling summary, not an architecture** — so standard #18's scope is **operationalise the
  flatten** (stated in the lab intro): turn a relational schema into one leak-free design matrix and prove
  the point-in-time guard is load-bearing. Opens the L034–L035 "relational data without RDL" thread; forward
  refs to L035 are plain text (lesson not yet published).
- **Two new reusable viz (standard #9, one per beat):** `assets/star-schema-viz.js` (the 3-table schema —
  customers/orders/order_items — with PK/FK badges, one-to-many ∞/1 glyphs, fact-vs-dimension labels, per-
  table grain, and per-relationship highlight buttons) and `assets/join-flatten-viz.js` (the flatten: one
  customer's nested orders collapse into a single fixed-width aggregate row on "Flatten"; a customer toggle
  shows every entity yields the same columns; readout names the discarded structure — an L035 hook). Reused
  **`leakage-viz.js`** (L002) for the point-in-time beat. Headless `labs/_viz_check_l034.js` **17/17**;
  **browser MCP still unavailable** (no chrome-devtools server) → node verification only, consistent with
  L021–L033.
- **Lab 034 = Tier C, pandas-only, mechanism lab** (no relkit/OpenML → runs in seconds). Deterministic toy
  DB (cutoff t = 2024-06-01; C1 has a future $999 order as the leakage trap; C4/C5 have zero pre-t orders).
  Crucial fragments: Task 1 PIT filter + `groupby("customer_id")` aggregation; Task 2 the leak-free design
  matrix (`LEFT JOIN` + `fillna(0)`, one row per customer); Task 3 prove the guard matters (leaky all-orders
  aggregate changes C1 from n=3/total=125 to n=4/total=1124). Stretch: a two-hop `customer→orders→items`
  distinct-product count. Solution executed **clean** (all 3 CHECK pass, EXIT + stretch print correct);
  gitignored; lab rendered to `labs/html/0034-*.html`. 8 student blanks.
- **Thesis:** L034 makes the join *literal* — the flat table Q1–Q3 assumed is **manufactured** by a hand-
  written, per-task, lossy pipeline (pick grain → join FKs → aggregate one-to-many → re-impose PIT). That
  turns C1 ("flattening is the lossy step RDL replaces") from slogan into a mechanism the learner can build
  and audit; L035 will quantify the discarded structure. Dossier verdict updated to "after L034."
- **Artifacts synced:** retrieval-pool **+2** (`l034-flatten` [misconception], `l034-thesis`); paper-deck
  **+1** (`kimball2013`); misconceptions **M34** (the flat table is a given → it's a hand-built lossy flatten)
  + **M35** (PIT is only about the label → every aggregate must filter to before t); thesis-dossier **+1**
  (FOR, C1/C2) + verdict; `reference/glossary.html` **+9** Q4 terms (relational DB, PK, FK, fact/dimension,
  star schema, grain, one-to-many, flatten, PIT correctness); `RESOURCES.md` +1 (Kimball). Manifest → **34**;
  `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l034.js` 17/17.
- Next: Lesson 035 — **What joins destroy** (Fey et al. ICML 2024 §2, the feature-engineering cost; ★
  preview) — the intellectual pivot of Year 1: enumerate the structure a flatten discards (cardinality,
  identity, higher-order paths, temporal order within a neighbour set). Planned new viz `flatten-loss-viz.js`
  (two distinct neighbourhoods collapsing to the same feature row — aggregation collision).

## Session 33 — 2026-07-21

- **Lesson 032 complete** — user said "lesson 32 done" (no EXIT ticket → no rubric score, per the
  L017–L031 precedent). Record [[learning-records/0077-lesson-032-complete.md]].
- **Lesson 033 published — When to Stop Feature Engineering** (curriculum lec 033, **Domingos 2012**,
  *A Few Useful Things to Know about ML*, CACM). Single skill: allocate a fixed modeling budget between FE
  and tuning; stop adding hand features when the marginal held-out gain sinks into the CV noise band (not
  when you run out of ideas). Record [[learning-records/0078-lesson-033-published.md]]. **Seventh
  application of standard #17** (every term defined first-principles: FE, marginal/diminishing returns,
  bias–variance/overfitting tax, CV noise band ±σ, budget/opportunity cost, "more data beats a cleverer
  algorithm").
- **Domingos is an essay, not an architecture** — so standard #18's "implement the paper" scope is
  **operationalise the essay's three claims as one controlled experiment** (stated in the lab intro): a
  fixed-order feature-budget loop, a curve that peaks then declines, and a noise-band stopping rule.
- **Verified live** (`labs/_verify_l033.py`, credit_g Tier A, seed 0, 5-fold, model held fixed): adding
  hand features 0→10 to a **HistGB**, CV ROC-AUC peaks at **k=3 (0.7911 vs 0.7865 baseline, +0.0046 —
  inside the ±0.032 band → not significant, L023)** then **declines to 0.7659 by k=8, below baseline** (the
  overfitting tax). A **linear** model drifts only **+0.006** (baseline 0.7913), also within noise. On a
  strong model + small flat table, single-table FE buys nothing provable, and then costs.
- **One new reusable viz (standard #9):** `assets/fe-returns-viz.js` — the diminishing-returns curve
  (two real curves GBDT + linear with ±1σ noise bands, a "stop here" marker at the GBDT peak, a k-slider
  reading out Δ-vs-baseline and within-noise/exceeds-1σ). Reused **`feature-viz.js`** (L009) for the
  "why FE matters for a weak model" beat. Headless `labs/_viz_check_l033.js` **11/11**; **browser MCP still
  unavailable** (no chrome-devtools server) → node verification only, consistent with L021–L032.
- **Lab 033 = Tier A, sklearn, experiment lab.** Crucial fragments: the feature-budget loop (Task 1), the
  stopping rule `first_within_noise` (Task 2), and the GBDT-vs-linear model-dependence rerun (Task 3);
  stretch points at the L023 corrected resampled t-test to prove the peak is not significant. Solution
  executed clean (all CHECK pass, EXIT numbers match `_verify_l033.py` exactly); gitignored; lab rendered.
  **Thread gotcha recurred** (Session 31 note): HistGB oversubscribes OpenMP after the torch install →
  ~5 min/fit *inside the sandbox*; ran the verify + nbconvert **outside the sandbox** with
  `OMP_NUM_THREADS=4` (SETUP cell pins the env vars too).
- **Thesis:** the load-bearing L033 line — this **quantifies the ceiling**. Manual single-table FE returns
  have effectively gone to zero, then negative, against a competent model (sixth Q4 deflation:
  L028–L032 ties + now L033's decline). The features that would still pay are relational aggregates
  *across* tables (DFS by hand, L009) → what RDL learns end-to-end; "the returns moved across the join,"
  the human-effort ratio Y4 tests. Dossier verdict updated to "after L033."
- **Artifacts synced:** retrieval-pool **+2** (`l033-fe-returns` [misconception], `l033-thesis`);
  paper-deck **+1** (`domingos2012`); misconceptions **M32** (FE always helps → diminishing then negative;
  stop at Δ<σ) + **M33** (diminishing returns is model-relative); thesis-dossier **+1** (FOR+BAR, C1/C4/C3)
  + verdict; `reference/glossary.html` **+4** Q4 terms (diminishing returns, marginal return, CV noise band,
  FE budget rule); `RESOURCES.md` +1 (Domingos 2012). Manifest → **33**; `node labs/_check_pedagogy.js`
  clean; `node labs/_viz_check_l033.js` 11/11.
- Next: Lesson 034 — **Relational data without RDL** (Kimball star schema & joins; sketch a 3-table join),
  opening the "relational data without RDL" thread (L034–L035) that sets up the what-joins-destroy bridge.

## Session 32 — 2026-07-19

- **Lesson 031 complete** — user said "lesson 31 done" (no EXIT ticket → no rubric score, per the
  L017–L030 precedent). Record [[learning-records/0075-lesson-031-complete.md]].
- **Lesson 032 published — TabTransformer Preview** (curriculum lec 032, **Huang et al. 2020**, `2012.06678`;
  "read the architecture figure"). Single skill: read/reconstruct the TabTransformer data-flow (categorical
  entity embeddings → Transformer self-attention → **contextual** embeddings; continuous features bypass →
  concat → MLP head) and state its honest verdict. Record [[learning-records/0076-lesson-032-published.md]].
  **Sixth application of standard #17** (every term defined first-principles: Transformer, token,
  self-attention, `softmax(Q·Kᵀ/√d)V`, context-free vs contextual, FFN, LayerNorm). It's a **preview** —
  the deep treatment + training lab are Y2 lec 045.
- **Honest verdict kept:** TabTransformer **matches** tree ensembles on supervised tabular (the +1.0% AUC is
  over *other DL*, not trees); real wins are robustness to noise/missingness + a **+2.1% semi-supervised**
  lift. Fifth flat-table tie in a row (L028/L029/L030/L031/L032) — single-table cleverness is exhausted; the
  thesis hook is that a contextual embedding = a weighted aggregate of related vectors = within-row message
  passing, the exact operation RDL runs across tables via foreign keys.
- **Two new reusable viz (standard #9, one per beat):** `assets/tabtransformer-arch-viz.js` (Fig. 1 as a
  4-stage stepper — one figure walked stage by stage) and `assets/attention-context-viz.js` (context-free vs
  contextual toggle + click-a-query attention blend on a 3-token row; illustrative numbers, captioned).
  Headless `labs/_viz_check_l032.js` **14/14**; **browser MCP still unavailable** (no chrome-devtools server).
- **NEW standard #18 — labs implement the paper (user directive this session):** starting with Lab 032,
  a lab's crucial content is a **faithful (if minimal) implementation of the lesson's core paper**, and labs
  should be *very informative*. Decide scope per paper — **whole model** / **key parts** / **gradual across
  labs** (for a multi-lesson paper) — and state it in the lab intro. Annotate cells with the paper element
  (fig/§/eq), keep PROVIDED minimal, prefer real data + torch, tie EXIT to the paper's claim. Recorded in
  NOTES Preferences #18 + the `lab-authoring` skill (§ "Labs implement the paper"). Reference impl: L032.
- **Lab 032 = Tier A, torch, forward-only** — *rebuilt this session* under #18 (initial numpy toy scrapped).
  The user asked labs to follow paper implementation and flagged that TabTransformer recurs at L045, so the
  paper is **split gradually**: **L032 builds the architecture** (scaled dot-product self-attention [Vaswani
  §3.2] → the Transformer block with residual+FFN+LayerNorm [Huang Fig. 1] → the full TabTransformer forward
  on real credit_g), **L045 trains + pre-trains + benchmarks** it. A real **18,671-param** module, forward-run
  on 1000 rows; contextual proof: `checking_status`'s contextual vector moves **0.160** when only `housing`
  changes, its entity embedding fixed. `_verify_l032.py` + executed solution clean; lab HTML re-rendered.
  User decisions: *full Transformer block, forward-only; torch not numpy.*

## Session 31 — 2026-07-18

- **Lesson 030 complete** — user said "lesson 30 done" (no EXIT ticket → no rubric score, per the
  L017–L029 precedent). Record [[learning-records/0073-lesson-030-complete.md]]. Closes **Y1 Q3**.
- **Lesson 031 published — Embeddings for Categoricals (Q4 opener)** (curriculum lec 031, **Guo &
  Berkhahn 2016**, core ★). Single skill: the four categorical encodings (one-hot / ordinal / target /
  entity embedding), why ordinal fakes an order, why target encoding **leaks** unless out-of-fold, and how
  entity embeddings learn a dense similarity-capturing vector — the first *learned representation* and the
  bridge to neural tabular / RDL. Record [[learning-records/0074-lesson-031-published.md]]. **Fifth
  application of standard #17** (every term/formula defined from first principles: cardinality, one-hot,
  ordinal false order, target encoding, smoothing, OOF, entity embedding, embedding dim, representation).
- **torch INSTALLED this session** (user directive: "pytorch will be needed anyway"). `.venv/bin/pip
  install torch --index-url .../cpu` → **torch 2.13.0+cpu** (aarch64). It was already in
  `requirements-labs.txt` (line 11) but had never actually been in the venv (recurring Session 22–30 gap).
  Now the entity-embedding experiments + lab stretch are real, not sklearn stand-ins.
- **ENV GOTCHA (important, recurring-risk):** after the torch install, sklearn `HistGradientBoosting`
  oversubscribed OpenMP threads → **~21 s per fit** (vs 0.28 s). Fix: set `OMP_NUM_THREADS=1` (+ OPENBLAS/
  MKL/NUMEXPR) **before** importing numpy/sklearn, and `torch.set_num_threads(1)`. Baked into
  `_verify_l031.py` and the lab SETUP cell. If future labs feel hung, suspect thread oversubscription first.
- **Verified live** (`labs/_verify_l031.py`, credit_g Tier A): bake-off (5-fold × 3 seeds) — one-hot 61
  cols linear **0.782**/gbdt 0.778; ordinal 20 cols linear **0.739** (false order hurts linear) / gbdt
  0.774 (tree shrugs); OOF-target 20 cols linear 0.784 / gbdt 0.769. Leak: naive target-encode a
  near-unique id → **0.891 AUC on pure noise**, OOF → **0.504** (chance). Entity-embedding MLP **0.774 ±
  0.033** vs **fair** one-hot MLP **0.798 ± 0.043** (3 splits) = **a tie**; an *undertrained* one-hot MLP
  (0.728) would have faked a +0.07 win — the **L028 weak-baseline trap, live**. Honest, thesis-consistent
  landing: a learned representation buys nothing on a small flat table; the payoff is structural/at-scale.
- **Three new reusable viz (standard #9, one per beat):** `assets/encoding-taxonomy-viz.js` (the same
  column under all 4 encodings — toggle is right here: same mechanism, one knob), `assets/target-leak-viz.js`
  (naive vs OOF on a unique id; real 0.891 vs 0.504), `assets/embedding-space-viz.js` (REAL credit_g
  `purpose` embedding, PCA→2-D, + illustrative Guo-Berkhahn German-states map). Headless
  `labs/_viz_check_l031.js` **14/14**; **browser MCP unavailable** (headless) → node verification only,
  per L021–L030.
- **Lab** `labs/0031-embeddings-for-categoricals.ipynb` — Tier A, `_build_l031.py`. Crucial fragment
  (Task 1) = the **smoothed OOF target encoder** (`(count*mean + m*glob)/(count+m)`); Task 2 = the encoding
  bake-off (fill the target branch); Task 3 = the leak (naive vs OOF on a signal-free id). Student blank
  (4 `____`, 0 outputs); solution executed clean & gitignored; matches lesson numbers. **Runnable torch
  stretch** (commented, ungraded) trains a real entity-embedding net.
- **Artifacts synced:** manifest → **31 entries** (L031 Q4, published); `labs/html/0031-*.html` rendered;
  `retrieval-pool.js` +3 (`l031-ordinal`, `l031-leak` misconception, `l031-embeddings`); `paper-deck.js`
  +1 (`guo2016`, core ★); `misconceptions.md` **M28** (target encoding leaks unless OOF) + **M29**
  (embeddings only tie one-hot on a small flat table — the L028 trap); `thesis-dossier.md` +1 line (L031,
  FOR+BAR, C4/C1) and **verdict updated to "after L031 / Q4 opener"**; `glossary.html` **+ a new Q4
  section** (9 terms). `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l031.js` 14/14.
- Next: Lesson 032 — **TabTransformer preview** (Huang et al. 2020; read the architecture figure),
  continuing the Q4 bridge to neural tabular.

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

### Labs implement the paper (2026-07-19, from L032) — applies to L032 and all future labs

18. **Labs follow the paper's implementation — and are very informative (from 2026-07-19, user directive):** the user asked that, starting with **Lab 032**, a lab's crucial content be a **faithful (if minimal) implementation of the lesson's core paper**, not a generic sklearn/toy exercise; and that labs be *very informative*. Per lab, **decide the implementation scope** explicitly and state it in the lab intro:
    - **Whole model** — build and run the paper's architecture end-to-end in one lab when it is small enough (an MLP, a single attention block, a GBDT-from-scratch stump loop).
    - **Key parts** — implement the paper's *load-bearing* mechanism(s) faithfully and PROVIDE/borrow the rest, when the full model is too large for one sitting (e.g. implement scaled dot-product + the transformer block; use a plain MLP head).
    - **Gradual across labs** — when one paper spans **several lessons**, split its implementation across those labs so each lab lands one coherent, runnable piece **aligned with its lesson**. Reference: **TabTransformer (Huang 2020)** — L032 (preview) builds the *architecture + forward pass* (real data, no training); Y2 lec 045 adds *training, semi-supervised pre-training, and benchmarking*. Note the split in each lab's intro and in the learning record so the arc is legible.
    - **"Very informative" concretely:** (a) annotate each implementation cell with the paper element it realises — figure/section/equation ref (e.g. "Fig. 1 Transformer layer", "Vaswani §3.2 eq. 1"); (b) keep PROVIDED scaffolding minimal so the student writes the *load-bearing* code, not boilerplate (mid-zone, standard #3); (c) prefer **real data + a runnable result** over toys wherever feasible (Tier A; use torch now that it is installed); (d) make the EXIT ticket / reproduction target tie back to the paper's actual claim; (e) it is fine — encouraged — for a paper-implementation lab to be longer than a mechanism lab (length is not the constraint, per standard #17).
    - This *extends* the four-block paper-reproduction structure and the "reproduction labs build incrementally" note already in Preferences; it does not replace mid-zone difficulty (#3), tiers (#7), or intro content (#8). Full rule in the `lab-authoring` skill. Reference implementation: **L032**.

### The learner's own pipeline (2026-07-25, from L036) — the Tier-A artifact for the rest of Q4

19. **The homework submission lives OUTSIDE this workspace, at `/home/avist/Projects/homework` (from 2026-07-25):** it is the ReAction L&D response-prediction submission — `report.md` (Tasks 1–4), `src/features.py` + `src/modeling.py`, `notebooks/01..04`, and `artifacts/` (per-fold CSVs `cv_folds_M{0,1,2a,2b,2c}_*.csv`, OOF `.npz`, `model.joblib`, `feature_pipeline.joblib`, SHAP tables). Shape: 119,498 situations × 99 columns joined from persons/situations/responses, **5,587 labelled (4.68 %)**, 5 classes, **4,851 persons of whom 675 repeat**, encoded to 132 features. There is **no event timestamp** anywhere in the schema — only `responses.observed_at`, recorded 4–8 weeks *after* the event. Reuse this artifact rather than a public dataset whenever a Q4 lesson wants Tier A; **L037 (package the pipeline) and L040 (Y1 exit) should build on the fixed version**, and the audit harness `labs/_audit_l036.py` (+ `_selection_l036.py`) already reproduces its M1 arm bit-exactly (log-loss 1.4248 ± 0.0392, ECE 0.0363).
    - **The four L036 findings, for continuity:** (1) `CalibratedClassifierCV(cv=5)` takes an *ungrouped* nested split inside the correctly-grouped outer fold → degrades the shipped calibrator, leaves the reported metric honest (re-measured 1.4232 / 0.0360); (2) M2a shipped over M1 on 0.0032 nats = 8 % of one fold's σ, losing 2/5 folds and flipping under leave-one-fold-out → a *decision* defect; (3) encoder/KMeans/propensity fit on all 119,498 rows → transductive, but re-measured in-fold the optimism is **undetectable** (−0.0011 nats, 3 % of fold σ); (4) no event timestamp ⇒ no temporal split for a weekly-retraining deployment → a declarable limitation. Also: the report's only "significant" claim (M1−M0, naive p = 0.0146) dies under Nadeau–Bengio (0.0514) and Holm (0.0583) while the effect survives.
    - **Harness note (reproducibility finding):** `src.modeling._lgbm()` uses `n_jobs=-1`, which on this 12-core WSL2 box costs **210 s per multiclass fit vs 9.2 s at `n_jobs=6`** (thread contention on 5.6k rows). Any rerun of that pipeline should cap `n_jobs`; it is the difference between a re-measurement being affordable and being skipped. Clean best-of-3 timings on a quiet box (`labs/_timing_l037_results.json`): **9.81 s at `n_jobs=1`, 6.01 at 2, 3.52 at 6, 13.74 at 12**; `deterministic=True` costs about 30 % on top (4.58 s at 6) and, here, buys nothing.
    - **The L037 reproducibility ledger, for continuity (harnesses `labs/_repro{,2,3}_l037.py`, `_repro_env_l037.py`, `_ece_estimator_l037.py`; results in the matching `_*_results.json`):** nine one-knob perturbations against a hash of the full OOF matrix, reference config = person-grouped 5-fold, 400 trees, `n_jobs=6`, seed 0, float32, lightgbm 4.6.0 → **sha `d2f0e4bf9b4fd761`, mean log-loss 1.632168, fold σ 0.035009**. **Bit-identical (8):** rerun; `n_jobs` ∈ {1, 2, 12}; `deterministic=True` at all four thread counts; `force_row_wise` vs `force_col_wise`; shuffled training-row order; model seed ∈ {1,2,3,4}. **Moved (1):** dropping the `.astype(np.float32)` → sha `a4377f2a443dc970`, **258/5,587 argmax flips (4.6 %)**, max |Δp| 0.326, Δ mean log-loss **+0.00133** (folds 0/1/4 stayed bit-identical, 2/3 diverged). **Crashed (1):** lightgbm 4.5.0 + scikit-learn 1.9.0 → `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'`. **Splitter seed** (the knob nobody calls a knob): mean log-loss **1.6191–1.6357, range 0.0166 nats** over 5 draws = **5× the 0.0032 margin that chose the shipped model**. **ECE estimator:** pooled 0.017760 vs mean-of-folds 0.033200 (**1.87×**) on the same predictions; noise floor from a perfectly-calibrated control = **0.0149** at n=5,587, **0.0335** at n=1,117, **0.1071 ± 0.0297** at n=107 — so the submission's one failed ship-gate (0.094 vs a 0.05 threshold on the 107-row `age = missing` slice) sits *below* its own floor and cannot be passed by any model. Counter-intuitive corollary measured in the lab: on a badly-calibrated model (ECE ≈ 0.093) the two estimators agree within 1 % — **the estimator choice matters most exactly when the number is good.**
    - **`requirements-labs.lock.txt` now exists** (121 packages), written by the L037 lab's stretch task. `requirements-labs.txt` still states only lower bounds and remains the install path; the lock records what actually produced every verified number in this workspace. Regenerate it whenever the env changes deliberately.

### Compute policy (2026-07-25, user sidenote) — what to do when a lesson outgrows the laptop

20. **The box is a 12-core WSL2 CPU with no GPU, ~15 GB RAM. Two cloud paths exist and they are NOT alternatives — they have different operators (from 2026-07-25):**
    - **Modal = the *authoring* runtime (the agent's). ALREADY SET UP AND VERIFIED — see `modal/README.md`.** The token from `curie-llm` is at **`~/.modal.toml`** (workspace `pszar92`) and the CLI is **`~/.local/bin/modal`** (pipx, client 1.4.1, *not* in the project `.venv`), so **no signup step is needed and the agent can escalate today, unattended.** Harness `modal/common.py` (App `relational-labs`, CPU + GPU images, volume `relational-artifacts` at `/artifacts`), copied from the `curie-llm` pattern at the user's direction. Both smoke tests pass (2026-07-25): CPU `{python 3.12.6, 24 cores, sklearn 1.9.0, lightgbm 4.7.0}`, GPU `{torch 2.13.0+cu130, cuda_available True, Tesla T4, matmul ok}`. Starter plan is **$0 with $30/month of credits** (per-second, no card, no rollover) ≈ **50 T4-hours**, ~14 A100-40GB-hours, CPU $0.135/core-hour (batch CPU work is effectively free). **Agent shell calls need `full_network` permission** — the sandbox blocks modal.com. Academic credit grants (**up to $10k** for students/labs/researchers) — apply *before* Y3/Y4 RelBench, not during, since RelBench scale will not fit in $30.
    - **Colab = the *learner's* runtime (the user's).** Already wired end-to-end: every lab's first code cell is the `@colab-bootstrap` (`labs/_colab.py`) and `notebooks.html` renders an "Open in Colab" link per lab. Free tier = **T4 16 GB, ~15–30 GPU-h/week (dynamic, unpublished), 12 h session cap, ~90 min idle disconnect keyed to *browser-tab interaction*, ephemeral disk**; exceeding the quota drops you to CPU-only for hours. Therefore: **never design a lab that needs an unattended long run on free Colab.** Pro (~$10/mo) buys 24 h runtimes + background execution if it ever becomes the bottleneck. **Kaggle Notebooks** are the better *free* unattended option (background execution, ~30 GPU-h/week) if Colab throttles.
    - **Consequence for lesson production:** if the agent needs a number for the lesson HTML → **Modal**. If the student needs a GPU to do the lab → **Colab/Kaggle**, with a CPU fallback config so the CHECK cells still pass on the laptop.

    **Escalation ladder — work down it in order, never skip a rung silently:**
    0. **Suspect thread oversubscription before hardware.** It has been the real cause *every single time* so far: L031 HistGB **21 s → 0.28 s** per fit with `OMP_NUM_THREADS=1`; L036 LightGBM **210 s → 9.2 s** with `n_jobs=6`. Check this first — it is free and usually 10–20×.
    1. **Shrink the measurement, not the claim.** Fewer trees/epochs, subsampled rows, fewer folds — but hold the config **identical across arms** so the comparison stays internally valid, and say so in the intro (L036: 120 trees vs the submission's 400, both arms).
    2. **Change the scope, not the paper** (standard #18): forward-only → key-parts → gradual-across-labs. L032 built the TabTransformer *architecture* and deferred training to L045; that was a compute decision as much as a pedagogical one.
    3. **Tier C synthetic** for mechanism isolation, or a **torch-free stand-in** for portability (L030 used `MLPClassifier` for the ResNet so the checkpoint runs anywhere).
    4. **Escalate to Modal** when the lesson's *verified* number genuinely needs a GPU or more than ~30 min of CPU. Commit the launch script next to the audit harnesses (`labs/_*.py`) so the number is reproducible, and quote the hardware in the learning record.
    5. **Escalate to Colab-GPU for the lab** when the *student* needs the GPU: label the lab "**GPU recommended — open in Colab**", keep total runtime under ~20 min on a T4, checkpoint to Drive, and ship a smaller CPU config that still makes the CHECK cells pass.

    **Modal job convention:** one shared `modal/common.py`, one thin `modal/l0NN_<thing>.py` per lesson that needs it, launched with `modal run --detach`; artifacts written under `/artifacts` + `artifacts.commit()`, pulled with `modal volume get`. Return **only plain stdlib types** from a remote function — the value is unpickled *locally* where torch does not exist (`torch.__version__` is a `TorchVersion`, not a `str`; returning it raw raises `DeserializationError` — found by the smoke test). Pin threads in cloud jobs too: the container reports 24 cores, so the L031/L036 oversubscription trap applies there as well. Full runbook + gotchas in `modal/README.md`.

    **The honesty rule (this is the load-bearing part).** A downscaled run is a *different experiment*. State the config in the lab intro **and** the learning record, never quote a shrunken result as if it were the paper's, and if the affordable version cannot support the claim, **say the claim is unsupported** rather than shipping a toy that looks like evidence. If a lesson is genuinely blocked on compute, raise it in the session instead of quietly substituting something cheaper — that is standard #17/L036 discipline applied to hardware.

    **When this actually bites (don't build infrastructure early).** L037–040 and much of early Y2 are CPU-fine. First real trigger: **L042** (MLP/ResNet *trained*, not forward-only). Hard requirement by **L062–L068** (TabPFN / TabICL) and unavoidable in **Y3–Y4** (GNNs; RelBench `rel-amazon` is millions of nodes — this is also where $30/month stops being enough, hence the academic grant). Run one **20-minute Modal smoke test before** the first lesson that depends on it, never during.

### The lab ships WITH the lesson (2026-08-05, user directive) — applies to all future lessons

21. **A lesson is not "created"/"published" until its lab notebook exists (from 2026-08-05, user directive: "make sure you dont forget about lab anymore"):** the curriculum's `Lab` column is a deliverable, not a suggestion. When the request is "create lesson N", the lab `labs/NNNN-<slug>.ipynb` is part of that unit of work and ships in the same session as the lesson HTML — built via `_build_lNNN.py`, its verified numbers produced by `_verify_lNNN.py` (or a Modal job, standard #20), its `labPath` set in `lessons/manifest.json`, and its Colab link rendered by `notebooks.html`. **`labPath: null` is allowed ONLY for genuine writing lessons with no code deliverable** — the essay/peer-review/synthesis pattern (L038, L039; a checkpoint-essay). Environment-setup or "forward-pass only" is **not** an exemption: if the lesson trains, reads, benchmarks, or implements anything, it gets a notebook.
    - **The miss this fixes:** L041 (rtdl setup) shipped `labPath: null` with an inline lab, defensibly (pure setup); but **L042 was published `labPath: null` when its curriculum lab is "Train ResNet baseline"** — a training lab, exactly the case standard #20 names as "the first real training lab." That was a real omission, retrofitted the same day. Do not repeat it: check the curriculum `Lab` column before calling a lesson done.
    - **Build-before-done checklist (every non-writing lesson):** (a) `_verify_lNNN.py` run, numbers recorded; (b) `_build_lNNN.py` → student `labs/NNNN-*.ipynb` (PROVIDED/TODO/CHECK/EXIT, concept recap, `@colab-bootstrap` first cell) + `labs/solutions/NNNN-*.ipynb`; (c) `manifest.json` `labPath` set; (d) `notebooks.html` renders it; (e) any new lab dep added to `requirements-labs.txt`; (f) lesson HTML's inline "Lab" section points at the notebook. Only then is the lesson done.
    - This *extends* standards #3/#7/#8 (mid-zone, tiers, intro content) and #18 (labs implement the paper); it does not replace them. Full rule in the `lab-authoring` skill (§ "The lab ships with the lesson").

    **BINDER DROPPED (user decision, 2026-07-25).** `binder/requirements.txt` omitted **torch, xgboost, lightgbm, catboost** while `notebooks.html` advertised Binder as "the *real* environment" — so the advertised zero-setup path had been silently broken for every lab from **L014** on (~20 labs) since the gallery shipped at L012. Rather than pay for slow Binder image builds, **Colab is now the single canonical run-anywhere runtime** (it already had the working `@colab-bootstrap` and offers a free T4). Removed: the `binder/` directory, `binderUrl()` + the "Run on Binder" link in `assets/notebooks.js`, the legend bullet in `notebooks.html` ("four options" → three), and the Binder mentions in `index.html` and `labs/README.md`. The `@colab-bootstrap` prose in ~30 already-built notebooks still says "on a local venv or Binder it does nothing" — harmless (it *is* a no-op anywhere but Colab) and not worth rebuilding/re-rendering every lab; `labs/_colab.py` is the source of truth and now says "local venv or Jupyter", so it self-corrects as labs are rebuilt.

### Build from scratch; reference libraries only VALIDATE (2026-08-05, user directive) — applies to all future labs

22. **The load-bearing model/mechanism is implemented FROM SCRATCH; a reference library is a validation point, never the teacher (from 2026-08-05, user directive: "use rtdl as a validation point ... most of the models to learn anything should be built from scratch").** To learn anything, the student must write the load-bearing code — the forward pass, the update rule, the algorithm — not import it as a black box. A canonical implementation (**rtdl** for MLP / ResNet / FT-Transformer; the analogous reference for other topics) is used **only to validate** the hand-built version:
    - **Preferred (crisp):** identical forward pass — build the from-scratch and reference modules with the *same* architecture and copied weights, assert `torch.allclose` on the outputs.
    - **Practical (when weight-matching is fiddly):** train both under the *same* protocol and assert the from-scratch score is within a stated tolerance of the reference (e.g. `|Δ ROC-AUC| < 0.03` over seeds) — "your implementation reproduces the reference." This catches silent bugs and the too-weak-baseline trap (L038) while keeping the *understanding* in the student's hands.
    - Only peripheral boilerplate (data load, plotting, the *other* models in a bake-off) may be imported. Promote reusable from-scratch models into `labs/relkit/` (e.g. `relkit/nets.py`) so later labs build on them (the incremental-codebase preference), rather than swapping in the library.
    - **Exception:** when the lesson's *skill is explicitly using the ecosystem tool* (e.g. L041 "set up rtdl", or a lesson about a library's API), the tool may be used directly — state it in the lab intro.
    - Extends standard #18 (labs implement the paper). Reference implementation: **L042** (from-scratch MLP/ResNet in `relkit/nets.py`, validated against rtdl).

### Never conclude from one dataset — multi-dataset rigor (2026-08-05, user directive) — applies to all future lessons

23. **A comparative conclusion needs MULTIPLE datasets and rank-based statistics — not one dataset (from 2026-08-05, user directive: "comparing and making conclusions on one dataset is [not] enough ... make sure future lessons will be as research rigorous as possible").** Any "A beats/ties B" claim — above all "no universal winner" — is only credible across datasets. From here on, every comparison lesson/lab must:
    - (a) evaluate on **≥3 real datasets** (more is better; use *small* OpenML tables so it stays CPU-affordable — e.g. credit_g/diabetes/blood-transfusion/kc1/phoneme — or subsample larger ones and **state the downscaling**, honesty rule #20);
    - (b) report **per-dataset mean ± std over ≥3 seeds** (confidence intervals, never a bare point estimate);
    - (c) summarise **across** datasets with a **rank-based** method — mean ranks + **Friedman test** and a **Nemenyi critical-difference diagram** (the L023/L030 method; reuse `assets/cd-diagram-viz.js`) — with effect sizes, not a single t-test;
    - (d) when our own compute can't reach the N the strong claim needs, **ground the strong version in the large published benchmark and cite it** (Grinsztajn 2022 ≈45 datasets; Gorishniy 2021; TabArena) — clearly separating "verified here on k datasets" from "established in the literature on N". A single-dataset result is a **demonstration**, never the evidence.
    - Extends standards #17 (thoroughness) and #20 (honesty/compute). Reference implementation: **L042** (credit_g + small-OpenML set: per-dataset CIs, mean-rank/Friedman summary, big-benchmark citation).

### Paper-mirror doctrine — implementation, datasets, reproducibility (2026-08-06, user directive) — applies to all future lessons/labs

24. **Always try to mirror paper implementations from scratch — including datasets and reproducibility (from 2026-08-06, user directive: "make sure that future lessons/labs will always try to mirror paper implementations from scratch, so we can learn as much as possible from papers. Also in terms of datasets and reproducibility point of view").** When a lesson's core source is a paper, the unit is a *paper mirror*, not a loosely inspired exercise. Three axes, all required unless an exception is stated in the lab intro:

    **(A) Implementation (extends #18 + #22).** Implement the paper's load-bearing algorithm/architecture **from scratch**. Annotate cells with paper figure/section/equation refs. Choose scope (whole model / key parts / gradual across labs) and state it. Reference libraries validate only — they do not teach the mechanism.

    **(B) Datasets (extends #7 + #23).** Prefer the **paper's own datasets, splits, and preprocessing** when they are open and affordable under #20. If that is impossible (license, size, egress, compute):
    - pick the closest Tier-A/B substitute (or a documented subsample of the paper's data);
    - **name the paper datasets you are *not* running** and what claim therefore cannot be reproduced here;
    - never present a substitute bake-off as "we reproduced Paper X" without that gap statement;
    - for comparative claims, still obey #23 (≥3 datasets, seeds, ranks/Friedman/CD) — and when the paper itself used a large suite, cite that suite for the strong claim.

    **(C) Reproducibility (extends #20 honesty + the L037 ledger pattern).** Every paper-mirror lab ships a minimal reproducibility contract in the intro + EXIT:
    - fixed seeds (and splitter seeds called out, not just model seeds — L037 lesson);
    - library versions via `requirements-labs.txt` / lockfile when numbers are claimed;
    - protocol held identical across arms; downscales stated as a *different experiment*;
    - `_verify_lNNN.py` (+ optional Modal job) produces committed `_*_results.json`;
    - EXIT target = paper metric within a stated tolerance **or** an honest fail / protocol-deviation note (never a silent toy that looks like the paper's table).

    - **Exception (narrow):** pure writing lessons (`labPath: null`), or a lesson whose skill is explicitly *using* a tool/API (e.g. L041). Still cite the paper; do not pretend the notebook is a reproduction.
    - **Compatibility:** the "reproduction labs build incrementally" preference still holds for *harness* code (`relkit/`) — reuse loaders/CV/metrics; do **not** reuse that as an excuse to import the model from a library instead of writing it.
    - Reference stack: **L032** (paper architecture mirror), **L037** (reproducibility ledger), **L042** (from-scratch + multi-dataset + rtdl validate). Full authoring checklist in the `lab-authoring` skill.

### Paper-results scale-up + visible implementation (2026-08-28, user directive) — applies to all future paper-mirror lessons/labs

25. **After the lab, reproduce the paper's results — and keep the implementation visible in the notebook (from 2026-08-28, user `/teach`: "next lessons will be trying even harder to reproduce results of the implemented architectures, if scale will not allow you to train it, add either modal script so I can train and compare or Google colab, next steps after lab implemented … I want to be sure that I will learn right conclusions." Follow-up: "code is not hidden behind package, so I can see direct implementation in notebook").** Two failure modes this closes: (1) a downscaled bake-off silently becoming "what the paper found"; (2) a from-scratch model that the student never actually *sees* because the bake-off `import`s `relkit`.

    **(A) Visible implementation.** The paper's architecture (forward pass, train loop, the rest of the encoder beyond the TODO fragment) appears as a **PROVIDED cell whose source is inlined** from the canonical `labs/relkit/*.py` file (`relkit.paper_repro.inline_source`, skipping the names the student writes). `from relkit.tabnet import TabNetEncoder` is allowed in `_verify` / Modal / as a *checker*; it is **not** an acceptable way to present the model in the student notebook. Data loaders, CV, and metrics may still be imported from `relkit`.

    **(B) Paper-results track, required after EXIT.** The learning lab may downscale (minutes on CPU). That is a different experiment from the paper's table. Every paper-mirror lab then ships a **NEXT STEP** (not stretch) that trains closer to the paper:
    - same from-scratch code the student just read;
    - paper dataset / HPs / metric when open, else a documented substitute with named gaps;
    - a printed **conclusion ledger** with three buckets — *verified here* / *paper claim* / *scale-up* — and verdicts MATCH / CLOSE / FAIL / INCOMPARABLE / NOT_RUN / DIRECTION_* (`relkit.paper_repro.format_ledger`).
    Until the scale-up has been run, the paper claim stays **cited, not reproduced**.

    **(C) When local scale cannot train it.** Ship **both** operators, don't pick silently:
    - **Modal** (unattended): `modal/l0NN_paper_repro.py`, `modal run --detach … --preset closer`. Template: `modal/_template_paper_repro.py`.
    - **Colab** (student, GPU): the same loop inlined in the notebook, gated by `RUN_PAPER_REPRO = False` until they attach a T4. Never an unattended long run on free Colab (idle timeout is tab interaction).
    Presets: `smoke` (seconds) · `closer` (T4, tens of minutes) · `paper` (hours, paper HPs).

    **(D) Right conclusions.** A DIRECTION_TIE on a subsample of a 0.002 paper edge is a lesson about *power*, not a refutation. An INCOMPARABLE 85.7% on a different split is not a MATCH. Do not average buckets; do not invert a lab ranking "because papers win at scale."

    Extends #18 / #20 / #22 / #24. Reference: L043–L045 retrofits this session. Full checklist in `lab-authoring`.
