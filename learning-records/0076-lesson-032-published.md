# 0076 — Lesson 032 published: TabTransformer Preview

**Date:** 2026-07-19
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 032 — matches the CURRICULUM row (032: "TabTransformer preview ·
Huang et al. 2020 · Read architecture figure"). Second lesson of Q4 (Consolidation & bridge to neural
tabular, L031–L040).
**Primary reading:** Huang, Khetan, Cvitkovic, Karnin 2020, *TabTransformer: Tabular Data Modeling Using
Contextual Embeddings* (`2012.06678`) — Fig. 1 + §2. Optional background: Vaswani et al. 2017 (`1706.03762`)
§3.2 for scaled dot-product attention.

## Single skill
Read and reconstruct the TabTransformer architecture — categorical embeddings → a stack of Transformer
self-attention layers → **contextual** embeddings, while continuous features **bypass** the Transformer and
are concatenated before an MLP head — and state its honest empirical verdict: it **matches** tree ensembles
(does not beat them) on supervised tabular, its measured gains are over other deep methods and in robustness
/ semi-supervised pre-training, and the idea that carries forward is that attention makes a column's
representation depend on its row-context.

## Why this was the ZPD
L031 ended by naming the entity embedding as the atom of RDL but left it *context-free*. L032 is the
curriculum's designated next step ("TabTransformer preview") and the natural bridge: it takes exactly the
L031 embeddings and shows what a Transformer does to them (contextualisation via self-attention). It is a
**preview** — the deep TabTransformer treatment + training lab live at Y2 lec 045 — so the scope is
"read the architecture figure," not "train the model." It also lets the tie pattern (L028/L029/L030/L031)
recur once more and, crucially, introduces attention/aggregation-over-related-vectors, the conceptual seed
of GNN message passing that Year 3–4 RDL is built on.

## What shipped
- **Lesson** `lessons/0032-tabtransformer-preview.html` (~40 min, standard #17 thoroughness): full
  vocabulary section (Transformer, token, self-attention, attention weights + the `softmax(Q·Kᵀ/√d)V`
  formula, context-free vs contextual, FFN/MLP head, LayerNorm — each from first principles) → the
  architecture stage-stepper viz → the contextual-attention viz → predict → the **honest verdict** table
  (the +1.0% is over other DL; matches trees; robustness; +2.1% semi-supervised) → teach-back → thesis
  bridge (contextual = within-row message passing; RDL = across-table) → 5 subtleties → 3 quizzes → primary
  reading → lab. Forward-references to L045/L046 are plain text (those lessons don't exist yet).
- **Two reusable viz (standard #9, one per mechanistic beat):**
  - `assets/tabtransformer-arch-viz.js` — the Fig. 1 data-flow as a 4-stage stepper (embed / Transformer→
    contextual / continuous bypass / concat+MLP), highlighting the boxes + arrows for each stage. Correct
    use of a stepper = one figure, walked stage by stage.
  - `assets/attention-context-viz.js` — context-free vs contextual toggle on a 3-token row; click a query
    column to see its softmax attention weights and its contextual vector assemble as a weighted blend of
    the others. Illustrative numbers (a fixed toy), captioned as such.
  - Headless check `labs/_viz_check_l032.js` — **14/14 pass**. **Browser MCP unavailable** (headless env;
    no chrome-devtools server) → node verification only, consistent with L021–L031.
- **Lab** `labs/0032-tabtransformer-preview.ipynb` — **Tier A** (real `credit_g`), **PyTorch**,
  **forward-only** (no training). *Redesigned mid-session* to follow the new **standard #18** (labs
  implement the paper) after the user asked labs to be paper-faithful and very informative, and flagged
  (correctly) that TabTransformer returns at L045 — so the implementation is **split**: this lab builds the
  *architecture*, L045 trains/benchmarks it. Three load-bearing blanks, each annotated to a paper element:
  Task 1 = scaled dot-product self-attention `softmax(Q·Kᵀ/√d)·V` (Vaswani §3.2); Task 2 = the **Transformer
  block** (wire the two residual + LayerNorm sub-layers of Huang Fig. 1); Task 3 = the **full TabTransformer
  forward** (embed → *N* blocks → flatten contextual ⊕ LayerNorm(continuous) → MLP head), forward-run on all
  1000 rows + the contextual-property proof. A real 18,671-param module, not a toy. Student blank
  (6 `____`, 0 outputs); solution executed clean & gitignored (`labs/solutions/` is in `.gitignore`).
  Built via `_build_l032.py`; rendered to `labs/html/0032-*.html`. (User decisions: full Transformer block,
  forward-only; torch not numpy.)
- **Verify** `labs/_verify_l032.py` — ran clean; numbers below.

## Verified live (`_verify_l032.py` + executed solution; torch, real credit_g, deterministic seed 0)
- Self-attention returns `(B, m, d)` with weight rows summing to 1 (matches `softmax(Q·Kᵀ/√d)·V` exactly).
- Transformer block preserves shape `(B, 13, 16)` and applies **LayerNorm last** (token mean |·| = 4.5e−8).
- Full TabTransformer forward on **1000 real rows** → logits `(1000,)`, **18,671 params** (untrained).
- **Contextual property:** with `checking_status` held fixed and only `housing` changed,
  `checking_status`'s **contextual** vector moved **0.160** (L2) while its context-free entity embedding was
  byte-identical — the concrete "context-free ≠ contextual" demonstration, now on real data.
- Data-flow: 13×16 embeddings → contextual → flatten (208) ⊕ 7 numeric → MLP → 1 logit.

## Honest framings kept
- **TabTransformer matches trees; it does not beat them.** The +1.0% mean AUC is over *other deep methods*
  (the L028 honest-baseline story), not over GBDTs, which it ties. The real wins are robustness to
  noisy/missing features and a +2.1% **semi-supervised** lift from unlabeled data — a lever trees lack.
- **The tie is the thesis, not a letdown.** Fifth flat-table tie in a row (L028/L029/L030/L031/L032):
  single-table cleverness (search, architecture, representation, attention) is exhausted. A contextual
  embedding is a weighted aggregate of related vectors — the same operation a GNN runs over an entity's
  foreign-key neighbours. TabTransformer attends *within a row*; RDL attends *across tables*. Same idea,
  one relational level up.
- **Preview, not full treatment.** Only categoricals are contextualised in the 2020 design (FT-Transformer,
  Y2, tokenises numerics too); heads/residuals/pre-training objective are deferred to Y2 lec 045.

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l032-contextual`, `l032-verdict` [misconception], and via the deck).
- `assets/paper-deck.js` +1 (`huang2020`, the architecture + honest verdict).
- `misconceptions.md` **M30** (TabTransformer beats trees → it *matches* them; +1.0% is over other DL) and
  **M31** (contextual = a bigger entity embedding → it's a *different* object: row-dependent, built by
  `softmax(Q·Kᵀ/√d)·V`), each with a matching pool item where applicable.
- `thesis-dossier.md` +1 ledger line (L032, BAR + FOR, C3/C1/C4 — attention over columns ties, but is
  literally within-row message passing) and **Current verdict updated to "after L032 / TabTransformer
  preview."**
- `reference/glossary.html` — Q4 section +3 terms (self-attention, contextual embedding, TabTransformer).
- `lessons/manifest.json` regenerated → **32 entries** (L032 Q4, published); `labs/html/0032-*.html`
  rendered.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l032.js` 14/14 clean.

## Next
Lesson 033 — **When to stop feature engineering** (Domingos 2012, *A Few Useful Things to Know about
Machine Learning*; FE time-budget exercise), continuing Q4's consolidation before the "relational data
without RDL" thread (L034–L035) that sets up the star-schema / what-joins-destroy bridge.
