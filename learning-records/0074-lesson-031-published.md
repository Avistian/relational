# 0074 — Lesson 031 published: Embeddings for Categoricals (Q4 opener)

**Date:** 2026-07-18
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 031 — the **Q4 opener**. Matches the CURRICULUM row (031:
"Embeddings for categoricals · entity embeddings (Guo & Berkhahn 2016) · Target encoding pitfalls").
Opens Q4 "Consolidation & bridge to neural tabular" (L031–L040). First lesson of Year 1 Q4.
**Core paper (★):** Guo & Berkhahn 2016, *Entity Embeddings of Categorical Variables* (`1604.06737`),
§2–4. Background: Micci-Barreca 2001 (smoothed target encoding), fast.ai tabular (embedding-size rule).

## Single skill
Know the four ways to encode a categorical for a model — one-hot, ordinal, target (mean) encoding, and
learned **entity embeddings** — why one-hot is safe-but-wide, why ordinal invents a false order, why
target encoding is powerful but **leaks** unless done out-of-fold, and how entity embeddings turn each
level into a learned dense vector that captures similarity; and understand that on a small single table
they all roughly **tie**, so the embedding's real payoff (compactness at high cardinality + a reusable
learned geometry) is a **representational** one — the bridge to neural tabular and RDL.

## Why this was the ZPD
Q3 closed with the verdict that single-table **search** (L029 AutoML tie) and **architecture** (L028
neural tie, L030 checkpoint tie) are exhausted, so the open axis is the **representation** (C4). The most
elementary place a representation is chosen is how a categorical becomes numbers — so Q4 opens there. It
also introduces the *entity embedding*, which is literally the atom of RDL (embedding foreign-key
entities), making it the natural first step of the bridge. Target encoding also lets Q3's leakage
discipline (L002/L022) recur on a *feature*, consolidating rather than only advancing.

## What shipped
- **Lesson** `lessons/0031-embeddings-for-categoricals.html` (~55 min, standard #17 thoroughness): full
  vocabulary section (categorical, cardinality, one-hot, ordinal, target encoding, smoothing, out-of-fold,
  entity embedding, representation — each from first principles) → the encoding-taxonomy viz → the real
  bake-off table → target-encoding **leakage** (predict → viz → the OOF+smoothing fix) → **entity
  embeddings** (dim heuristic, Guo-Berkhahn, embedding-space viz) → predict → the honest
  embedding-vs-one-hot **tie** table (with the weak-baseline trap called out) → teach-back → thesis bridge
  → 5 subtleties → 3 quizzes → primary reading → lab.
- **Three reusable viz (standard #9, one per mechanistic beat):**
  - `assets/encoding-taxonomy-viz.js` — the same credit_g `checking_status` column under all four
    encodings (toggle; correct use of a mode toggle = one mechanism/one knob), with real smoothed target
    means and the +54-column one-hot cost.
  - `assets/target-leak-viz.js` — naive vs out-of-fold target encoding of a unique id; naive rows are
    flagged "= its own label", real payoff AUCs **0.891** (leak) vs **0.504** (honest).
  - `assets/embedding-space-viz.js` — the **real** learned credit_g `purpose` embedding (5-D, PCA→2-D,
    coloured by risk, sized by count) with an honest "loose/tie" caption, plus an illustrative
    Guo-Berkhahn German-states map reproducing the paper's geographic-recovery finding.
  - Headless check `labs/_viz_check_l031.js` — **14/14 pass**. **Browser MCP unavailable** (headless env)
    → node verification only, consistent with L021–L030.
- **Lab** `labs/0031-embeddings-for-categoricals.ipynb` — Tier A (real credit_g). Crucial fragment
  (Task 1) = the **smoothed out-of-fold target encoder** (`(count*mean + m*glob)/(count+m)`); Task 2 =
  the encoding bake-off (fill the target branch = `np.column_stack([oof_target_encode(X[c], y) for c in
  CAT])`); Task 3 = the leak (fill the naive `groupby.transform("mean")`). Student blank (4 `____`, 0
  outputs); solution executed clean & gitignored; numbers match the lesson. Runnable-but-commented
  **torch stretch** trains a real entity-embedding net. Built via `_build_l031.py`; rendered to
  `labs/html/0031-*.html`.
- **Verify** `labs/_verify_l031.py` — ran clean; numbers below.

## Verified live (credit_g, real Tier-A, via _verify_l031.py + executed solution)
- **Encoding bake-off (5-fold × 3 seeds, mean test ROC-AUC):** one-hot 61 cols → linear **0.782** / gbdt
  0.778; ordinal 20 cols → linear **0.739** / gbdt 0.774; OOF-target 20 cols → linear 0.784 / gbdt 0.769.
  Read: ordinal's **false order** costs the linear model 0.043 but the tree only 0.004; OOF target **ties
  one-hot at a third of the width**.
- **Target-encoding leak (near-unique id, no real signal, GBDT):** naive train **0.891** / test **0.891**
  (encoding built before the split → the feature *is* the label); OOF train 0.514 / test **0.504**
  (chance, the truth).
- **Entity embeddings (real torch net, 3 splits):** entity-embedding MLP **0.774 ± 0.033** vs a **fair**
  one-hot MLP **0.798 ± 0.043** → a **tie**. A first, undertrained one-hot MLP scored **0.728** — which
  would have faked a +0.07 embedding "win": the **L028 weak-baseline trap, reproduced live**. Fixing the
  baseline erased the win.
- Lab solution CHECK + EXIT all clean; the executed numbers match the lesson.

## Honest framings kept
- **On a small flat table, a learned representation ties.** Entity embeddings land in the same ~0.78 band
  as one-hot / target / tuned GBDT (L030). Not a failure — a statement about the *data*: little similarity
  or cardinality to exploit. The payoff is structural, at scale, and on high-cardinality foreign keys.
- **The weak-baseline trap is real and shown, not just warned about.** The undertrained one-hot MLP
  (0.728) → fair (0.798) flip is the L028/L029 lesson made concrete in this very lesson.
- **Target encoding is an embedding.** The 1-D learned-number case; entity embeddings are the d-dim
  generalisation — so the same out-of-fold leakage discipline rides along.

## Env note
**torch 2.13.0+cpu installed this session** (user directive), finally closing the recurring Session 22–30
gap; it was already in `requirements-labs.txt`. **Thread gotcha:** after the torch install, sklearn
`HistGradientBoosting` oversubscribed OpenMP threads (~21 s/fit vs 0.28 s); fixed by setting
`OMP_NUM_THREADS=1` (+OPENBLAS/MKL/NUMEXPR) before importing numpy/sklearn and `torch.set_num_threads(1)`
— baked into `_verify_l031.py` and the lab SETUP cell. Note this for all future torch+sklearn labs.

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l031-ordinal`, `l031-leak` [misconception], `l031-embeddings`).
- `assets/paper-deck.js` +1 (`guo2016`, core ★ paper).
- `misconceptions.md` **M28** (target encoding is safe → it leaks unless OOF; 0.891→0.504 on a signal-free
  id) and **M29** (embeddings beat one-hot → they *tie* on a small flat table; the L028 trap).
- `thesis-dossier.md` +1 ledger line (L031, FOR + BAR, C4/C1 — entity embeddings are the atom of RDL;
  they tie on a flat table because the value is structural) and **Current verdict updated to "after
  L031 / Q4 opener."**
- `reference/glossary.html` — a **new Q4 section** with 9 terms (categorical encoding, cardinality,
  one-hot, ordinal, OOF target encoding, target-encoding smoothing, entity embedding, embedding dimension,
  learned representation).
- `lessons/manifest.json` regenerated → **31 entries** (L031 Q4, published); `labs/html/0031-*.html`
  rendered.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l031.js` 14/14 clean.

## Next
Lesson 032 — **TabTransformer preview** (Huang et al. 2020; read the architecture figure), continuing the
Q4 bridge to neural tabular — where the per-column embeddings introduced here feed a self-attention
encoder. Q4 (L031–L040) proceeds toward the tabular↔relational seam.
