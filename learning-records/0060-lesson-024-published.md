# 0060 — Lesson 024 published: The Grinsztajn Benchmark

**Date:** 2026-07-15
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 024 — opens the Grinsztajn arc (024–027)
**Core paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform
deep learning on typical tabular data?* (NeurIPS 2022 D&B, arXiv:2207.08815), §1–4.

## Single skill
Read and reproduce the Grinsztajn benchmark **protocol**: compare model families across a
random-search hyperparameter **budget curve** (not a single tuned number), **normalize** test
scores per dataset so they can be aggregated, and know the **dataset-selection criteria** that make
the tree-vs-DL verdict fair. §5 (the three inductive biases) is deferred to L025–L027.

## Why this was the ZPD
L019 previewed the three biases (abstract + §1). Q3 (021–023) made a *single* comparison honest —
split, features, significance test. L024 is the natural next step: zoom out from one comparison to a
whole benchmark, i.e. the *measuring instrument* (datasets + protocol) that earns the right to the
mechanistic conclusion the next three lessons dissect. Matches CURRICULUM row 024 exactly.

## What shipped
- **Lesson** `lessons/0024-grinsztajn-benchmark.html` (~40 min): why one tuned number lies → dataset
  curation funnel (§3) → budget curve (§4) → per-dataset normalization → what the benchmark
  concludes (+ forward links to L025–L027 biases) → thesis bridge → subtleties → 3 quizzes.
- **Two new reusable viz** (standard #9, one per mechanism):
  - `assets/benchmark-budget-viz.js` — the iconic Fig-1 budget curve (test acc vs # random-search
    iters, log-x), GBT above NN at every budget; **raw ↔ normalized** toggle (mechanism 2) + budget
    slider. Data are the real credit-g curves from the verify script.
  - `assets/dataset-funnel-viz.js` — the §3 selection funnel; click a stage for its criterion +
    rationale (counts illustrative, criteria load-bearing).
  - Headless check `labs/_viz_check_l024.js` — 11/11 pass. **Browser MCP unavailable** again
    (headless env) → node verification only, consistent with L021–L023.
- **Lab** `labs/0024-grinsztajn-benchmark.ipynb` — **Tier A** (real OpenML credit-g via
  `relkit.load_tier_a`), reproduction of the *protocol*. 3 TODO + stretch. Crucial fragment (Task 1)
  = the **best-so-far budget curve** (select by validation, report test, avg over orderings); Task 2
  = affine min-max normalization; Task 3 = the gap curve + verdict. Student blank (7 `____`, 0
  outputs); solution executed clean & gitignored.
- **Verify** `labs/_verify_l024.py` — ran end-to-end (~132 s).

## Verified live (credit-g, seed 0, 30 configs, 40 orderings)
- Budget curve: **GBT above MLP at every budget.** Default (k=1) GBT **0.7906** vs MLP **0.7286**
  (gap **+0.0620**); fully tuned (k=30) GBT **0.7850** vs MLP **0.7700** (gap **+0.0150**).
- Tuning helps the MLP more (0.729→0.770) than the GBT (barely moves) — the gap **narrows but never
  closes**. Grinsztajn's Figure 1 in miniature.
- Honest small-data artifact named in the lesson: the GBT curve *dips* slightly at large budgets
  because picking best-of-many configs overfits the 200-row validation set (callback L004/L017) —
  which is exactly why "not too small" is a selection criterion.
- Normalization: worst model → 0.00, best → 1.00, ordering preserved.
- Solution notebook: all CHECK + EXIT clean; table numbers match the lesson exactly.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l024-budget-curve` [misconception], `l024-normalization`).
- `assets/paper-deck.js` +1 (`grinsztajn2022-benchmark`; distinct from the L019 biases card
  `grinsztajn2022`).
- `misconceptions.md` **M21** (one tuned number ≠ fair benchmark).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1) — GBT default raises the bar; the whole
  contest lives inside the single-table world the thesis attacks.
- `reference/glossary.html` +6 Q3 terms (budget curve, per-dataset normalization, selection
  criteria, heterogeneous columns, default vs ceiling).
- `lessons/manifest.json` → 24 entries; all labs re-rendered to `labs/html/`.
- `node labs/_check_pedagogy.js` clean.

## Env note
`.venv` present & functional (sklearn 1.9.0, numpy, scipy, Node v24). **OpenML reachable this
session** → first Tier-A training lab of Q3 (credit-g), unlike the egress-blocked L021/L022.

## Next
Lesson 025 (Inductive bias: smoothness — Grinsztajn 2022 §5.2; explain the smoothness bias),
continuing the Grinsztajn arc.
