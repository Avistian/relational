# 0102 — Lesson 044 published: NODE (differentiable trees)

**Date:** 2026-08-21
**Status:** Published **with lab** (awaiting user completion).
**Curriculum:** Year 2 · Q1 · lecture 044 (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md` / `plan/year-2.md`: *NODE (differentiable trees)*; curriculum lab: *train NODE,
compare to CatBoost (both "tree-like") on the same task*.
**Primary reading:** Popov, Morozov & Babenko, ICLR 2020 — *Neural Oblivious Decision Ensembles for Deep
Learning on Tabular Data* ([arXiv:1909.06312](https://arxiv.org/abs/1909.06312) ★), §2 (the architecture)
+ §3 (experiments).
**Companion:** Peters, Niculae & Martins, ACL 2019 — *Sparse Sequence-to-Sequence Models*
([arXiv:1905.05702](https://arxiv.org/abs/1905.05702), the entmax family + entmax15 bisection).

## Single skill
Describe an **oblivious differentiable decision tree** and say **when a differentiable tree ensemble helps
over a GBDT** — then *prove the mechanism from scratch* (entmax15 feature choice, entmoid soft split,
outer-product routing), and hold NODE to **L042's baseline-first rule** against **CatBoost** (its exact
tree-shape twin) plus MLP/ResNet under one shared frame.

## Why this was the ZPD
L043 used the bar on TabNet — the *first* 2019 attempt at giving a net a tree's inductive bias. L044 is the
*other* one, and it is deliberately paired: NODE and CatBoost are the **same tree shape** (oblivious /
symmetric, Y1 **L016**), so the experiment isolates a single variable — *make the tree differentiable* — and
the lesson can ask the sharp question the whole thesis rides on: **what does differentiability actually buy,
if not accuracy on a flat table?** It recruits heavy prior recall: oblivious trees = CatBoost's symmetric
trees (L016), the entmax→sparsemax family (L043's sparsemax is α = 2; entmax15 is α = 1.5), fair-budget
ranks + under-powered Friedman (L023/L030), HP-budget parity (L038), and the baseline-first rule (L042).

## Standard #24 (paper-mirror) — how each axis was met
- **(A) Implementation.** `labs/relkit/node.py` implements NODE's load-bearing pieces **from scratch**,
  annotated with paper/repo refs: `entmax15` (α = 1.5 entmax by **bisection**, Peters 2019), `entmoid15`
  (the two-class entmax in closed form, NODE repo `lib/nn_utils.py`), `ODST` (one layer = an ensemble of
  differentiable oblivious trees: `feature_logits` → entmax feature choice, `thresholds` +
  `log_temperatures` → entmoid split, `bin_codes_1hot` → **outer-product** leaf routing), `DenseNODE`
  (DenseNet-style stacking), plus `train_node` / `node_auc`. The `entmax` package is the **checker only**.
- **(B) Datasets.** Bake-off on four small **Tier-A** OpenML tables (`credit_g`, `diabetes`,
  `blood_transfusion`, `kc1`) via `relkit.data.load_tier_a`. Gap statement shipped in lesson + lab: this is
  a **down-scaled demonstration**, *not* the paper's 40+ dataset benchmark, so the paper's Table-2
  win-rates are **not reproduced** — only the *mechanism* (validated exactly) and the *direction* of the
  flat-table verdict are claimed here.
- **(C) Reproducibility.** Fixed seeds (splitter seeds included), one shared frame across arms,
  `labs/_verify_l044.py` → committed `labs/_verify_l044_results.json` (444 s). Environment of record:
  **torch ≥ 2.2, entmax 1.3, catboost ≥ 1.2, rtdl_revisiting_models 0.0.2** (`entmax>=1.3` added to
  `requirements-labs.txt`, validation only).

## Verified numbers (evidence of record — `labs/_verify_l044.py`, budget 4, seeds 0/1/2, 100 epochs)
**Mechanism validated to machine precision** (from-scratch vs the `entmax` package): entmax15 max |Δ|
**5.6e-16**, entmoid15 max |Δ| **3.3e-16**; entmax15 produces **84.3%** exact zeros where softmax produces
none. (`labs/_check_l044.py` — **15/15 pass**: simplex + exact-zeros + reference agreement, entmoid ==
two-class entmax15, exact 0/1 saturation, ODST routing sums to 1 per (row, tree), the **oblivious**
one-distribution-per-level property, and a learning sanity check — NODE fits a 2-feature interaction to
AUC 1.000.)

**Bake-off (4 small tables, one shared frame, equal budget).**

| dataset | NODE | CatBoost | MLP | ResNet |
|---|---|---|---|---|
| credit_g | 0.776 ± 0.010 | 0.798 ± 0.032 | **0.808 ± 0.026** | 0.785 ± 0.035 |
| diabetes | 0.811 ± 0.021 | 0.815 ± 0.025 | **0.831 ± 0.024** | 0.823 ± 0.016 |
| blood_transfusion | 0.737 ± 0.048 | 0.734 ± 0.013 | 0.742 ± 0.019 | **0.774 ± 0.021** |
| kc1 | 0.791 ± 0.010 | **0.817 ± 0.014** | 0.790 ± 0.014 | 0.804 ± 0.013 |
| **mean rank** | **3.50** | **2.50** | **2.00** | **2.00** |

Friedman **chi2 = 3.6, p = 0.308** (k = 4, N = 4). **NODE beats CatBoost on 1/4.** **Cost:** NODE
**60.2 s** vs CatBoost **0.9 s** on credit_g — **~70× slower** for a *worse* rank.

**The disciplined reading (now M56 / M57).** Two separate statements, kept apart: (1) *statistically*,
p = 0.308 on four datasets licenses only "**cannot distinguish on this sample**" (L023); (2) *evidentially*,
the baseline-first rule (L042) puts the burden on the **new** model, so ranking **last** and costing ~70×
means the expensive model **did not clear the bar** here. The paper's own benchmark-scale win (beats GBDT on
most of 40+ datasets by a small margin, with thousands of trees + heavy tuning) is not contradicted — it is
simply out of scope for four small tables, and the lesson says so.

## The honest takeaway — value is COMPOSITION, not flat-table accuracy (M56 / M57)
The single most important idea in the lesson, and the one it ties to the thesis: NODE and CatBoost are the
**same tree shape**; NODE only changes *how the splits are found* (gradient descent, not greedy search). On
one flat table that trainability buys **nothing** and costs ~70×. Its value is that a GBDT's greedy splits
have **no gradient**, so a GBDT can never (a) co-learn feature embeddings, (b) stack **DenseNet-style** so
later trees split on earlier trees' decisions, or (c) sit inside an end-to-end / multi-modal pipeline. That
is exactly the relational setting the thesis cares about — so "ranks last on flat tables" and "worth
building" are compatible; the axis that matters is *does the tree need to plug into other learned
structure?*

## What shipped
- **Lesson** `lessons/0044-node.html`: warm-up (upTo:44) → oblivious-tree recap (L016 callback) → the three
  "softenings" (entmax15 feature choice with the softmax→entmax15→sparsemax family; entmoid soft split;
  outer-product routing to all 2^d leaves) → the interactive tree viz → DenseNet-style stacking →
  predict-before-reveal on the **CatBoost** bake-off → rank table + cost + the "did not clear the bar ≠
  significantly worse" separation → validated-mechanism box (machine-precision deltas) → *when
  differentiability matters* → teach-back on **why differentiability is worth its cost** → thesis bridge
  (differentiable-tree lineage) → 5 quizzes → primary reading → inline lab → teacher/ask-teacher → nav.
- **One new viz** (`file://`-safe): `assets/node-tree-viz.js` — a depth-3 differentiable oblivious tree that
  routes one row to all 8 leaves **softly**; a **τ (temperature) slider** collapses the routing to a single
  argmax leaf (compare the **"hard tree"** toggle) or spreads it toward uniform; per-level gap sliders drive
  the entmoid splits; the readout shows leaf entropy + effective-leaves + `ŷ = Σ wₖ·Rₖ`. CSS prefix
  `.nod-`. The pure math (`entmoid15`, `leafWeights`, `model`) is exposed for headless testing.
- **Headless checks:** `labs/_viz_check_l044.js` — **all 69 pass** (entmoid properties incl. exact 0/1
  saturation; outer-product routing sums to 1 and equals the per-level product; τ→0 one-hot == hard-tree
  argmax; τ↑ flattening; every verified number present in the lesson; each honesty statement).
  `labs/_check_pedagogy.js` — **all 40 pass** (incl. the new `popov2019` deck card). Browser MCP unavailable
  → node verification only (consistent with L021–L043).
- **Lab notebook** `labs/0044-node.ipynb` (+ solution, gitignored; + rendered `labs/html/0044-node.html`),
  built by `labs/_build_l044.py`. Four tasks: (1) implement **entmax15** (bisection) + **entmoid15**,
  validated against the `entmax` package; (2) implement the **ODST forward** (feature choice → soft split →
  outer-product routing), verifying leaf weights sum to 1; (3) the **baseline-first** bake-off NODE vs
  **CatBoost** vs MLP/ResNet with mean ranks + Friedman; (4) the **cost** measurement + a one-sentence
  verdict on *when NODE's differentiability would matter*. **Solution executed end-to-end with nbconvert —
  all CHECK assertions pass (17 PASS / 0 FAIL across 4 CHECK cells)** in **390 s** on CPU; its
  reduced-budget run (3 datasets, 2 seeds) reproduces the **direction**: NODE mean rank **3.33** (last, tied
  with ResNet) vs CatBoost 2.0 / MLP 1.33, Friedman **p = 0.145**, NODE beats CatBoost **1/3**, **~146×**
  slower on that table.

## Design choices
- **Tasks 1–2 are the load-bearing pieces, and only those.** The student writes entmax15, entmoid15, and the
  ODST forward — the three things that are *specifically NODE*; `DenseNODE`, training, and the baselines are
  provided in `relkit`. Implementing the whole stack would have been a typing exercise.
- **CatBoost is the headline comparator, not a generic GBDT.** The point of the lesson is that they are the
  *same tree shape*, so the only variable is differentiability. Substituting XGBoost would blur that.
- **The cost is measured, not hand-waved** (the ~70× / ~146× numbers). "Marginal wins, heavy compute" from
  `plan/year-2.md` is made concrete, which is what lets the honest verdict land.
- **`quarter: "Y2Q1"`** on the three new retrieval-pool items.

## Artifacts synced
- `assets/retrieval-pool.js` **+3**: `l044-entmax` (the entmax family), `l044-oblivious-routing`
  (outer-product routing), `l044-diff-cost` (misconception, mirrors M57 — when differentiability is worth
  its cost).
- `assets/paper-deck.js` **+1** card `popov2019` (the three softenings, DenseNet stacking, the
  machine-precision mechanism validation, and the honest CatBoost verdict).
- `misconceptions.md` **M56** (NODE learns better splits ⇒ should beat CatBoost → same tree shape, ranks
  last + ~70× on flat tables), **M57** (ranks last ⇒ pointless → value is composition a GBDT structurally
  cannot do).
- `reference/glossary.html` — Year 2 · Q1 **+6 terms**: NODE, oblivious (symmetric) decision tree, entmax
  (α-entmax), entmoid, outer-product leaf routing, differentiability as composition (not accuracy).
- `thesis-dossier.md` — narrative **L044** paragraph. BAR-raising with the sharpest FOR yet: an honest
  *loss* on flat tables that names precisely what structure-keeping (here, gradient flow through the tree)
  buys — composition — and why that only pays when there is surrounding structure to connect to.
- `lessons/manifest.json` — L044 entry with `labPath: labs/0044-node.ipynb` (44 entries).
- `requirements-labs.txt` **+`entmax>=1.3`** (validation only).
- L043 nav now links forward to L044.
- **New files:** `labs/relkit/node.py`, `labs/_check_l044.py`, `labs/_verify_l044.py` (+ results JSON),
  `labs/_build_l044.py`, `labs/0044-node.ipynb`, `labs/solutions/0044-node.ipynb` (executed, gitignored),
  `labs/html/0044-node.html`, `labs/_viz_check_l044.js`, `assets/node-tree-viz.js`.

## Open items
- **Self-supervised / semi-supervised NODE** is not a thing (unlike TabNet's pre-training), so nothing owed
  there. But NODE's real selling point — **composition** — is only *asserted* here, never demonstrated,
  because there is no module for it to compose with yet. That demonstration is a Year 2–4 burden (once an
  encoder / message-passing block exists, stack a differentiable tree head on it and show it co-trains).
- The paper's **benchmark-scale** result (thousands of trees, 40+ datasets) is out of scope for the lab; if
  the user wants it, it is a compute-heavy stretch, not a bolt-on.

## Next
User runs the lab and pastes the EXIT ticket (entmax15/entmoid validation, the ODST routing sum, the
NODE-vs-CatBoost rank table + cost, and the one-sentence "when differentiability matters" verdict) or says
"lab done". On completion, open **Lesson 045** per `CURRICULUM.md` / `plan/year-2.md` (continuing Y2 Q1 —
the next advanced tabular architecture), carrying the same baseline-first bar forward.
