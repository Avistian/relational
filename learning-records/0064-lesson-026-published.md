# 0064 — Lesson 026 published: Inductive Bias — Rotation

**Date:** 2026-07-16
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 026 — third lesson of the Grinsztajn arc (024–027), second of the
three inductive-bias mechanism lessons.
**Core paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep
learning on typical tabular data?* (NeurIPS 2022 D&B, arXiv:2207.08815), **§5.4 (Finding 3: data are
non-invariant by rotation, so learning procedures should not be either)**. Theory backing assigned as
skim: **Andrew Ng 2004**, *Feature selection, L1 vs L2 regularization, and rotational invariance*
(ICML 2004).

## Single skill
Explain **rotational invariance** in the sense of Ng (2004) — why a decision tree is *not* invariant
(axis-aligned splits are tied to the original basis) while an MLP *is* (linear first layer:
`W·(Qx)=(WQ)·x`), why that invariance is a **liability** on tabular data whose columns carry individual
meaning, and recognize the experiment that proves it: **a random rotation leaves the MLP untouched but
collapses the tree, reversing the ranking.**

## Section-number correction (important)
The workspace's prior records (L024-complete, L025) labelled **rotation as §5.3** and **uninformative
features as §5.4**. That is backwards. Verified against four independent copies of the paper (arXiv
ar5iv HTML, HAL, NeurIPS proceedings, OpenReview): **§5.3 = Finding 2 (uninformative features)**,
**§5.4 = Finding 3 (rotation)**. This lesson cites the **correct** section (§5.4 for rotation); L025's
one-line forward reference was fixed to match. The *lesson order* is unchanged (rotation = L026,
uninformative = L027) because it is baked into L025's quiz 3 ("rotational invariance … is the L026
bias") and the arc — and it is defensible pedagogically (rotation is the *mechanism* that Ng's theorem
uses to explain the uninformative-feature fragility of L027).

## Why this was the ZPD
L025 taught the first bias (smoothness) and explicitly forward-linked L026 = rotation. Rotation is the
bias that speaks most directly to the relational thesis (columns carry individual meaning → a database
schema is the maximal version of that), so it is the natural next mechanism and a strong thesis-bridge
beat. Matches the CURRICULUM Grinsztajn-arc plan (L024 built the instrument; L025–L027 dissect the three
biases).

## What shipped
- **Lesson** `lessons/0026-inductive-bias-rotation.html` (~35 min): rotational-invariance definition
  (Ng 2004) → why a tree isn't invariant / an MLP is → geometry viz → predict → the rotation experiment
  (ranking reversal) + table → Ng's sample-complexity link to L027 → teach-back → thesis bridge →
  subtleties → 3 quizzes → primary reading → lab.
- **Two new reusable viz** (standard #9, one per distinct mechanistic beat), both driven by REAL numbers
  from the verify script:
  - `assets/rotation-splits-viz.js` — geometry beat: an axis-aligned class quadrant a tree carves with
    two straight splits, vs the rotated diagonal wedge it can only *staircase* (red polyline), with the
    MLP boundary rotating *with* the data. Toggle: Original / Rotated basis. Readouts carry tree 0.987 →
    0.747 and MLP invariance.
  - `assets/rotation-gap-viz.js` — experiment beat: grouped bar chart of test accuracy (Tree/GBT/RF/MLP,
    original vs rotated) with a "Show ranking reversal" connector toggle. Shows tree-based collapse, MLP
    flat (+0.008), and the reversal.
  - Headless check `labs/_viz_check_l026.js` — 15/15 pass. **Browser MCP unavailable** (headless env) →
    node verification only, consistent with L021–L025.
- **Lab** `labs/0026-inductive-bias-rotation.ipynb` — **Tier C** (synthetic axis-aligned classification,
  mechanism isolation: no real dataset can toggle its own basis while holding information fixed). 3 TODO
  + stretch. Crucial fragment (Task 1) = **implement the random-rotation operator** (`Q` from
  `np.linalg.qr`; apply the same `Q` to train + test); Task 2 = fit tree/GBT/MLP on original vs rotated
  and record the gap; Task 3 = quantify invariance (MLP) vs collapse (tree) + the reversal. Student blank
  (10 `____`, 0 outputs); solution executed clean & gitignored. Built via `labs/_build_l026.py`.
- **Verify** `labs/_verify_l026.py` — the 2-D geometry demo + the multi-seed rotation experiment; ran
  clean.

## Verified live
- **Rotation experiment (mean of 5 seeds, 6 informative + 10 uninformative features, test accuracy):**
  Tree **0.987 → 0.747** (−0.240), GBT **0.997 → 0.824** (−0.173), RF **0.994 → 0.812** (−0.182),
  MLP **0.862 → 0.869** (+0.008). Original basis: tree beats MLP by **+0.126**; rotated: MLP beats tree
  by **−0.122** — the **ranking reverses**. MLP ≈ invariant (+0.008), all tree-based models collapse.
- **Lab solution (single seed, n=4000):** tree 0.973 → 0.727, GBT 0.991 → 0.814, MLP 0.838 → 0.858
  (+0.020); gap +0.135 → −0.131 (reversal); Q orthogonal (`QᵀQ=I`), distances preserved (lossless).
  All CHECK + EXIT clean.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l026-rotation` [misconception], `l026-invariance`).
- `assets/paper-deck.js` +1 (`grinsztajn2022-rotation`; distinct from L019 `grinsztajn2022`, L024
  `grinsztajn2022-benchmark`, L025 `grinsztajn2022-smoothness`).
- `misconceptions.md` **M23** ("rotation invariance is an elegant, desirable property" → on tables it is
  a liability; a lossless rotation reverses the ranking; Ng's ≥-linear sample-complexity bound).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1 — inherited MLP-head invariance is the bar; the
  "columns are meaningful" instinct is the same one behind "don't dissolve the schema").
- `reference/glossary.html` +4 Q3 terms (rotational invariance, natural basis, random rotation, Ng's
  sample-complexity bound).
- `lessons/manifest.json` → 26 entries; all labs re-rendered to `labs/html/` (incl. `0026-*.html`).
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l026.js` 15/15 clean.

## Env note
No `python3-venv` preinstalled this session → installed `uv` via curl and built a lean `.venv`
(scikit-learn 1.9.0, numpy 2.5.1, scipy 1.18.0, + jupyter/nbconvert/ipykernel; Node v22). All
experiments synthetic/local — no network needed. Env-setup agent should preinstall the lab venv so
future sessions skip the uv bootstrap (recurring note since Sessions 22–23).

## Next
Lesson 027 (Inductive bias: uninformative features — Grinsztajn 2022 **§5.3**, Finding 2; the
add/remove-junk-features experiment), the last of the three mechanism lessons — linked to L026 by Ng's
theorem — before the arc closes and Q3 continues.
