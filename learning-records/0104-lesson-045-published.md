# 0104 — Lesson 045 published: TabTransformer (contextual embeddings + self-supervised pre-training)

**Date:** 2026-08-28
**Status:** Published **with lab** (awaiting user completion).
**Curriculum:** Year 2 · Q1 · lecture 045 (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md` / `plan/year-2.md`: *TabTransformer — train it, show contextual categorical
embeddings help on categorical-rich data, and name its limitation (numeric features bypass attention)*;
curriculum lab: *run TabTransformer vs the L031 static-embedding MLP · table + "what context bought / where
it didn't."*
**Primary reading:** Huang, Khetan, Cvitkovic & Karnin, 2020 — *TabTransformer: Tabular Data Modeling Using
Contextual Embeddings* ([arXiv:2012.06678](https://arxiv.org/abs/2012.06678) ★), Fig. 1 (architecture),
§3.1–3.2 (column embedding + Transformer), §3.3 (RTD pre-training).
**Companion:** Vaswani et al. 2017 — *Attention Is All You Need*
([arXiv:1706.03762](https://arxiv.org/abs/1706.03762), the self-attention kernel).

## Single skill
Train a **TabTransformer** from scratch and hold it to **L042's baseline-first rule**: show that its
**contextual** categorical embeddings edge the **context-free** static-embedding MLP of L031/L032 (the
`n_layers=0` ablation) — but only by a hair, and both lose to **CatBoost** — then **name the limitation**
(numeric features bypass the attention) and use **RTD self-supervised pre-training** to buy a small
**label-efficiency** lift a GBDT structurally cannot.

## Why this was the ZPD
L032 built TabTransformer's architecture **forward-only** and previewed it as "matches trees, real wins are
robustness + a semi-supervised lift." L045 is the promised close of that preview: *train* it, *pre-train*
it, and *measure* the two claims. It is the natural next rung after L043 (TabNet) and L044 (NODE) — both
put through the same bar and both losing to trees on flat tables — because it isolates one clean variable
the others could not: flipping `n_layers` between 3 and 0 turns contextualisation on and off inside the
**same class**, so "what does context buy over a static entity embedding?" becomes a controlled experiment.
Heavy prior recall: entity embeddings (L031), self-attention `softmax(Q·Kᵀ/√d)·V` (L032), baseline-first +
shared frame + mean-rank/Friedman (L042/L023/L030), CatBoost native categoricals (L016), and the numeric
tokenisation gap that L041/L046 (FT-Transformer) exists to close.

## Standard #24 (paper-mirror) — how each axis was met
- **(A) Implementation.** `labs/relkit/tabtransformer.py` implements the load-bearing pieces **from
  scratch**, each annotated with its paper element: `scaled_dot_product_attention`,
  `MultiHeadSelfAttention`, `TransformerBlock` (residual attn + FFN, each LayerNorm'd), `TabTransformer`
  (column embeddings → N blocks → concat LayerNorm'd numerics → MLP head; `n_layers=0` = the context-free
  ablation), `corrupt_categorical` (RTD), `RTDHead` (per-column detector), `pretrain_rtd`, and
  `train_tabtransformer`/`tabtransformer_auc`. torch's own `F.scaled_dot_product_attention` and
  `nn.MultiheadAttention` are the **checkers only** (never used to build).
- **(B) Datasets.** Bake-off on three **categorical-rich** Tier-A OpenML tables (`credit_g`, `adult`
  subsampled to 4000, `churn`) via `relkit.data.load_tier_a`. Gap statement shipped in lesson + lab: this
  is a **down-scaled demonstration**, *not* the paper's 15-dataset benchmark, so only the *mechanism*
  (validated exactly) and the *direction* of the contextual + semi-supervised results are claimed.
- **(C) Reproducibility.** Fixed seeds (splitter seeds included), one shared frame across arms,
  `labs/_verify_l045.py` → committed `labs/_verify_l045_results.json` (**433.5 s**). No new dependencies —
  torch (L042) + catboost (L044) already in `requirements-labs.txt`.

## Verified numbers (evidence of record — `labs/_verify_l045.py`, seeds 0/1/2, 60 epochs)
**Mechanism validated to machine precision** (from-scratch vs torch's kernels):
`scaled_dot_product_attention` max |Δ| **6.7e-16**, `MultiHeadSelfAttention` vs `nn.MultiheadAttention`
max |Δ| **1.1e-16**. **Contextual property on a real credit_g row:** `checking_status`'s vector moves
**0.259** (L2) when its neighbour `housing` is flipped **with** attention, and exactly **0.000** at
`n_layers=0`. (`labs/_check_l045.py` — **15/15 pass**: attention kernels + weight simplex, block shape +
LayerNorm-last, the contextual-vs-context-free move, RTD corruption/label correctness, per-column detector,
detector learns above chance (acc 0.825), and a learning sanity check — fits a 2-column interaction to
AUC 1.000.)

**Bake-off — contextual (TabTransformer, n_layers=3) vs context-free (n_layers=0) vs CatBoost.**

| dataset | TabTransformer | context-free MLP | CatBoost |
|---|---|---|---|
| credit_g | 0.787 ± 0.016 | 0.792 ± 0.027 | **0.804 ± 0.024** |
| adult (4000) | **0.866 ± 0.009** | 0.866 ± 0.009 | **0.903 ± 0.006** |
| churn | **0.886 ± 0.010** | 0.882 ± 0.010 | **0.922 ± 0.014** |
| **mean rank** | **2.33** | **2.67** | **1.00** |

Friedman **chi2 = 4.667, p = 0.097** (k = 3, N = 3). **Contextual beats context-free on 2/3**;
**TabTransformer beats CatBoost on 0/3.** The gain of context over the static embedding is real but
**small and within noise**; the robust result is that **trees win the flat-table metric**, because numeric
features never touch the attention.

**Semi-supervised RTD (adult, 16 000 rows, unlabeled pool ≈ 11 200, 3 seeds, 30 pretrain epochs, gentle
FT-LR 5e-4).**

| labeled fraction | from scratch | pre-train + fine-tune | lift |
|---|---|---|---|
| 3% | 0.825 ± 0.017 | 0.833 ± 0.015 | **+0.008** (all 3 seeds positive) |
| 10% | 0.861 ± 0.006 | 0.862 ± 0.011 | +0.001 |

The label-efficiency lever a GBDT lacks — but **small, and it shrinks as labels grow**. It **collapses to
negative** under two failure modes (both diagnosed while iterating): too small an unlabeled pool, and too
large a fine-tune LR (catastrophic forgetting).

## The honest takeaway — context is a hair, self-supervision is the real (small) lever (M58 / M59)
Two claims the lesson keeps apart. **(1) Contextual > context-free** is supported but *barely* (2/3, within
noise) — and it is a *different* claim from **neural > trees**, which is **false here** (0/3). The reason is
structural and is the lesson's headline limitation: **only categoricals are contextualised; numeric
features are LayerNorm'd and concatenated, never attending to anything** — exactly what FT-Transformer
(L046) removes. **(2) Self-supervision** (RTD) is the genuinely new capability — learn from *unlabeled*
rows, which a tree cannot — but the payoff is a modest, fragile label-efficiency edge, not a revolution.
The detector only works because a swapped token looks wrong *only in context*, so the pretext sharpens the
very contextualisation the architecture adds.

## What shipped
- **Lesson** `lessons/0045-tabtransformer.html`: warm-up (upTo:45) → static-vs-contextual recap (L031/L032
  callback) → the architecture (column embed → N blocks → numeric bypass → MLP) with the reused arch +
  contextual viz → predict-before-reveal on the contextual-vs-context-free vs CatBoost bake-off → rank
  table + the "small edge / trees still win / numeric-bypass" reading → **RTD** section with the new
  `rtd-pretrain-viz` → **label-efficiency** section with the new `label-efficiency-viz` and the honest
  "small & shrinks" note → validated-mechanism box (machine-precision deltas) → teach-backs (contextual;
  RTD) → thesis bridge (within-row attention = across-table message passing) + the **L046** motivation →
  quizzes → primary reading → inline lab → teacher/ask-teacher → nav.
- **Two new viz** (`file://`-safe, headless-testable pure functions): `assets/rtd-pretrain-viz.js` (`.rtd-`)
  — a replace-probability slider corrupting a toy row, showing which cells a detector must flag and the
  effective replaced fraction `p·(1−1/card) < p`; `assets/label-efficiency-viz.js` (`.leff-`) — the
  scratch-vs-pretrain payoff curve over labeled fraction, anchored on the verified +0.008 / +0.001.
  Reused: `tabtransformer-arch-viz` and the contextual/attention viz from L032.
- **Headless checks:** `labs/_viz_check_l045.js` — **all 82 pass** (both new viz's pure functions;
  lesson↔asset CSS coupling; every verified number present; each honesty statement).
  `labs/_check_pedagogy.js` — **all 40 pass** (incl. the new `huang2020-l045` deck card). Browser
  verification **skipped at the user's explicit request** this session (headless-only, consistent with
  L021–L044).
- **Lab notebook** `labs/0045-tabtransformer.ipynb` (+ solution; + rendered `labs/html/0045-tabtransformer.html`),
  built by `labs/_build_l045.py`. Four tasks: (1) implement **`corrupt_categorical`** (the RTD pretext),
  validated vs the reference + the effective-fraction property; (2) the **RTD pre-training step** (corrupt
  → contextual encoder → per-column detector → BCE), verified by detector ROC-AUC ≈ 0.82 and the
  contextual-vs-context-free move; (3) the **baseline-first** bake-off contextual vs context-free
  (`n_layers=0`) vs CatBoost with mean ranks + Friedman; (4) the **label-efficiency** experiment
  (pre-train + gentle fine-tune vs from-scratch). **Solution executed end-to-end with nbconvert — all
  CHECK assertions pass (0 FAIL across the CHECK cells)** in **211 s** on CPU.

## Design choices
- **`n_layers=0` is the context-free comparator, by construction.** The context-free MLP is not a separate
  model — it is the same `TabTransformer` class with the Transformer stack removed, which is *exactly* the
  L031/L032 static entity embedding. One flipped knob isolates contextualisation; a bespoke MLP would blur
  it.
- **The lab reproduces the DIRECTION honestly, not the paper's numbers.** At the down-scaled budget
  (2 tables, 2 seeds) the contextual-vs-context-free gap is genuinely within noise and here **inverts**
  (context-free ranks ahead). Rather than hide this, the Task-3 verdict *teaches* it: a thin edge cannot
  survive two-table noise, so read the robust shape (CatBoost wins; numerics bypass attention) and defer
  the direction to the fuller `_verify_l045.py` run. This is the #23 discipline made pedagogical.
- **Task 4 needed a big unlabeled pool to be honest.** A first cut (adult capped at 3000 → ~2100 unlabeled)
  produced a **−0.061** lift with one seed crashing 0.795→0.673 — catastrophic forgetting. Fixed by giving
  Task 4 a dedicated 14 000-row frame (pool ≈ 9800) and the gentle FT-LR 5e-4, which lands the lift near
  zero (−0.002) with no crash — a faithful, non-broken down-scaling of the verified +0.008.
- **`quarter: "Y2Q1"`** on the three new retrieval-pool items; the new deck card is `huang2020-l045` so it
  coexists with the L032 preview card `huang2020`.

## Artifacts synced
- `assets/retrieval-pool.js` **+3**: `l045-contextual-vs-free` (what contextualisation changes),
  `l045-rtd` (why RTD is label-free and needs context), `l045-numeric-bypass` (misconception, mirrors M58 —
  contextualises categoricals yet loses to CatBoost because numerics bypass attention).
- `assets/paper-deck.js` **+1** card `huang2020-l045` (contextual embeddings, RTD, the machine-precision
  mechanism validation, and the honest small-edge/numeric-bypass verdict).
- `misconceptions.md` **M58** (contextualises categoricals ⇒ should beat CatBoost → 0/3, numeric bypass),
  **M59** (RTD ⇒ always helps → small, shrinks with labels, collapses under small pool / aggressive FT-LR).
- `reference/glossary.html` — **+2 terms** (Replaced Token Detection; self-supervised pre-training / label
  efficiency) and the **TabTransformer** entry updated with L045's from-scratch validation + `n_layers=0`
  ablation.
- `thesis-dossier.md` — narrative **L045** paragraph. BAR (another deep single-table architecture that ties
  the static embedding and loses to trees) + double FOR (within-row attention *is* the across-FK message
  passing of RDL; self-supervision on unlabeled rows foreshadows Y5 relational foundation models).
- `lessons/manifest.json` — L045 entry with `labPath: labs/0045-tabtransformer.ipynb` (45 entries).
- `lessons/0044-node.html` nav + body now link **forward** to L045.
- **New files:** `labs/relkit/tabtransformer.py`, `labs/_check_l045.py`, `labs/_verify_l045.py` (+ results
  JSON), `labs/_build_l045.py`, `labs/0045-tabtransformer.ipynb`, `labs/solutions/0045-tabtransformer.ipynb`
  (executed), `labs/html/0045-tabtransformer.html`, `labs/_viz_check_l045.js`,
  `assets/rtd-pretrain-viz.js`, `assets/label-efficiency-viz.js`.

## Open items
- **Browser verification not run** this session (user asked to skip). The two new viz are node-verified
  only; a `file://` pass with the L043 puppeteer method is still owed if any layout doubt arises.
- **Numeric-bypass limitation** is named and quantified (0/3 vs CatBoost) but its *fix* is L046
  (FT-Transformer tokenises numerics too) — the direct motivation for the next lesson.
- **The RTD lift is small and fragile.** The paper's larger semi-supervised gains need benchmark scale
  (more unlabeled data, more pretrain epochs) — a compute-heavy stretch, not a bolt-on.
- The thesis's *composition* FOR (contextual embedding = message passing) is still **asserted**, not
  demonstrated across tables — the same Year 3–4 burden noted for L044.

## Next
User runs the lab and pastes the EXIT ticket (RTD validation, the pretext-step detector AUC, the
contextual-vs-context-free probe, the bake-off ranks, the label-efficiency lift, and the one-sentence
"what context buys / where it stops" verdict) or says "lab done". On completion, open **Lesson 046**
(FT-Transformer) per `CURRICULUM.md` / `plan/year-2.md` — which removes exactly the numeric-bypass
limitation this lesson named, carrying the same baseline-first bar forward.
