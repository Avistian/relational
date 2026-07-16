# 0062 — Lesson 025 published: Inductive Bias — Smoothness

**Date:** 2026-07-16
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 025 — second lesson of the Grinsztajn arc (024–027), first of the
three inductive-bias mechanism lessons.
**Core paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep
learning on typical tabular data?* (NeurIPS 2022 D&B, arXiv:2207.08815), **§5.2 (Finding 1)**. Theory
backing assigned as skim: Rahaman et al. 2019, *On the Spectral Bias of Neural Networks* (arXiv:1806.08734).

## Single skill
Explain the **smoothness inductive bias** — why an MLP is pulled toward smooth (low-frequency) functions
while a tree fits irregular, jagged targets — and know the experiment that proves it is *the* mechanism:
**Gaussian-smoothing the target function closes the tree-vs-neural-net gap**.

## Why this was the ZPD
L024 built the *measuring instrument* (benchmark + protocol) and explicitly forward-linked L025–L027 as
the three biases that explain the verdict. Smoothness is the first and most fundamental bias, and it
connects cleanly to the L019 preview (`l019-biases`, M17). Matches CURRICULUM row 025 exactly
("Inductive bias: smoothness / Grinsztajn 2022 §5.2 / Explain smoothness bias").

## What shipped
- **Lesson** `lessons/0025-inductive-bias-smoothness.html` (~35 min): inductive bias → the MLP's
  smoothness/spectral bias vs the tree's piecewise-constant disposition → 1-D fit viz → predict → the
  target-smoothing gap experiment → teach-back → thesis bridge → subtleties → 3 quizzes.
- **Two new reusable viz** (standard #9, one per distinct mechanistic beat), both driven by REAL numbers
  from the verify script:
  - `assets/smoothness-fit-viz.js` — 1-D: an irregular target, a tree staircase, an MLP smooth fit, with
    a 3-stop target-smoothing slider + MSE readout. MLP/tree error ratio **5.30×** (raw) → 2.40× (medium)
    → **0.19×** (smooth, MLP wins). Shows the MLP over-smoothing the jags.
  - `assets/smoothness-gap-viz.js` — the §5.2 summary curve: GBT vs MLP test R² across smoothing
    length-scales, with a "variance kept" annotation. Gap **+0.22 → ~0** as the high-frequency variance
    is erased.
  - Headless check `labs/_viz_check_l025.js` — 12/12 pass. **Browser MCP unavailable** (headless env) →
    node verification only, consistent with L021–L024.
- **Lab** `labs/0025-inductive-bias-smoothness.ipynb` — **Tier C** (synthetic, mechanism isolation, per
  `lab-authoring`: real data cannot toggle its own irregularity). 3 TODO + stretch. Crucial fragment
  (Task 1) = **implement the Gaussian target smoother** (fill `d2` + the weight `exp(-d2/2h²)`); Task 2 =
  sweep smoothing and record the gap; Task 3 = show the gap tracks the target's variance. Student blank
  (8 `____`, 0 outputs); solution executed clean & gitignored.
- **Verify** `labs/_verify_l025.py` — 1-D curves + the multi-D gap curve (mean of 5 seeds); ran clean.
  **Built via** `labs/_build_l025.py` (mirrors the L024 build pattern; prepends `_colab.bootstrap_cells`).

## Verified live
- **1-D (seed 0, 300 samples, tree max_leaf_nodes=28, MLP 128×128):** raw tree MSE **0.0132** vs MLP
  **0.0699** (5.30× worse); smooth (h=0.12) tree 0.0064 vs MLP **0.0012** (MLP now wins, 0.19×).
- **Gap experiment (mean of 5 seeds, irregular 8-D regression):** GBT **0.938** vs MLP **0.717** at h=0
  (gap **+0.22**); collapses to −0.02 at h=1.0 (variance kept **19%**) and ~0.00 at h=2.0. The gap tracks
  the variance removed, not h — the signature of a mechanism.
- **Lab solution (single seed, n=2000):** raw gap **+0.332** → **−0.033** once >70% of variance smoothed;
  all CHECK + EXIT clean.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l025-smoothness` [misconception], `l025-piecewise`).
- `assets/paper-deck.js` +1 (`grinsztajn2022-smoothness`; distinct from L019 `grinsztajn2022` biases card
  and L024 `grinsztajn2022-benchmark`).
- `misconceptions.md` **M22** ("trees win because they're simply more powerful" → it's an inductive-bias
  *mismatch*; smoothing the target closes the gap).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1 — the smoothness bias is the honest reason MLP
  heads lose, and a hint that the irregularity is often crushed relational structure).
- `reference/glossary.html` +3 Q3 terms (smoothness/spectral bias, piecewise-constant fit, target
  smoothing).
- `lessons/manifest.json` → 25 entries; all labs re-rendered to `labs/html/`.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l025.js` clean.

## Env note
`.venv` present & functional (sklearn 1.9.0, numpy, scipy, Node v24). All experiments synthetic/local —
no network needed (unlike the Tier-A credit-g fetch in L024).

## Next
Lesson 026 (Inductive bias: rotation — Grinsztajn 2022 §5.3; the rotation experiment), continuing the
Grinsztajn arc. Trees are **not** rotationally invariant, which is a virtue on tables where columns are
individually meaningful (previewed in L019, `l019-rotation`).
