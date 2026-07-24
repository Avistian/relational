# 0082 — Lesson 035 published: What Joins Destroy

**Date:** 2026-07-24
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 035 — matches the CURRICULUM row (035: "What joins destroy · Fey ICML
2024 §2 (feature engineering cost) · List lost structure"). Fifth lesson of Q4 (Consolidation & bridge to
neural tabular, L031–L040) and the **close of the L034–L035 "relational data without RDL" thread**. Plan
entry: `plan/year-1.md` §035 — flagged there as "the intellectual pivot of Year 1" and "core of the Y1 exit
essay."
**Primary reading:** Fey, Hu, Huang, Lenssen, Ranjan, Robinson, Ying et al. — *Relational Deep Learning:
Graph Representation Learning on Relational Databases* (2024),
[arXiv:2312.04615](https://arxiv.org/abs/2312.04615) / [ICML 2024](https://proceedings.mlr.press/v235/fey24a.html),
**§1–2 only** (★ preview): the five issues with manual feature engineering and the definition of the
relational entity graph.

## Single skill
Enumerate the structural information a flatten-then-aggregate pipeline discards — cardinality, event
identity, temporal order within a neighbour set, and higher-order (multi-hop) paths — and recognise an
**aggregation collision**, where two entities with different neighbourhoods map to the same fixed-width
feature vector.

## Why this was the ZPD
L034 made the flatten *literal* (build the join + PIT aggregation) and ended on "aggregation is lossy by
construction." L035 is the curriculum's designated next step and the natural payoff: turn "lossy" from a
slogan into a **demonstrated** fact via a collision the learner can build and a fitted model that provably
cannot separate the pair. It reactivates L001 (the single-table assumption — now shown to be a *choice*),
L009 (each recovery feature is a hand-built DFS primitive), L019/L025–L027 (trees can't help once the signal
is gone from the columns), and L033 (the FE ceiling). It is the cleanest possible close of Year 1's
diagnostic arc and the setup for the Y1-exit essay and Y3–Y4's constructive burden.

## What shipped
- **Lesson** `lessons/0035-what-joins-destroy.html` (~40 min, standard #17 thoroughness): full vocabulary
  from first principles (lossy vs. lossless map, aggregation collision, cardinality, event/entity identity,
  temporal order within a neighbour set, higher-order/multi-hop path, relational entity graph) → the
  collision viz with a **predict-before-reveal** on "can a tuned model separate Ada and Bo?" → a taxonomy
  table of the four discarded structures (what/which-aggregates-lose-it/in-the-collision) → **Fey §1–2 in
  their words** (the verbatim five-issue enumeration, issue (4) as the load-bearing claim, the Gabor→pixels
  CV analogy, the REG definition) → teach-back → thesis bridge (closes L001→L035; loss is *information*, not
  capacity) → 5 subtleties (lossy≠useless; the add-more-aggregates treadmill; lossy≠leaky; trees don't
  escape it; the graph reframing isn't free yet) → 3 quizzes → primary reading → lab. Forward references
  to Y3–Y4 are plain text.
- **One new reusable viz (standard #9, the lesson's single mechanistic beat):**
  - `assets/flatten-loss-viz.js` — the aggregation collision: two customer subgraphs (customer → 3 orders in
    time order, each carrying amount + product) collapse under JOIN+AGGREGATE to two **identical** flat rows
    (`n=3/total=90/avg=30/max=50`, flagged with a red "⚠ identical" banner). A "Reveal the lost structure"
    toggle annotates each subgraph with its spend trend (↗ rising / ↘ falling) and distinct-product count
    (3 vs 1) and rewrites the readout to name what a graph keeps. Documented expected states in the header
    comment; CSS prefix `.fl-`.
  - Headless check `labs/_viz_check_l035.js` — **13/13 pass**. **Browser MCP unavailable** (headless env; no
    chrome-devtools server) → node verification only, consistent with L021–L034.
- **Lab** `labs/0035-what-joins-destroy.ipynb` — **Tier C** (small synthetic customers/orders/order_items),
  pandas + scikit-learn, runs in seconds. Standard #18 scope stated in the intro: Fey is a *position*
  preview, not an architecture, so the lab **demonstrates issue (4)** as a runnable collision rather than
  implementing a model. Deterministic DB, cutoff t = 2024-06-01, all orders pre-t (PIT was L034's job — this
  isolates the *loss*). Three crucial fragments: Task 1 = the L034 flatten + confirm Ada/Bo collide; Task 2 =
  fit a `LogisticRegression` on the flat features and prove P(churn) is identical for the pair (0.502) though
  their true labels differ (0 vs 1); Task 3 = recover structure with `spend_trend` (last−first: +40 vs −40,
  restores temporal order) and `n_distinct_products` (2-hop join: 3 vs 1, restores identity). Stretch = a
  third customer (Zoe) that collides again, making the treadmill visible. Student blank (7 `____`, 0
  outputs); solution executed clean (all 3 CHECK pass, EXIT + stretch correct) & gitignored. Built via
  `_build_l035.py`; rendered to `labs/html/0035-*.html`.

## Verified live (executed solution; deterministic synthetic DB, cutoff t = 2024-06-01)
- **Task 1** (flatten + collision) — Ada and Bo both → `n_orders=3, total_spend=90, avg_basket=30,
  max_basket=50` (byte-identical); other customers distinct (Cy 2/40, Di 1/100, Ez 4/20, Fi 3/120).
- **Task 2** (model can't separate) — fitted `LogisticRegression` gives **P(churn)=0.502307 for both** Ada
  and Bo, though true labels are **0 (Ada) vs 1 (Bo)**. Identical input ⇒ identical output; guaranteed wrong
  on at least one.
- **Task 3** (recover structure) — `spend_trend` = **+40 (Ada) / −40 (Bo)** restores temporal order;
  `n_distinct_products` = **3 (Ada) / 1 (Bo)** restores event identity; the pair no longer collides.
- **Stretch** (treadmill) — a third customer Zoe (orders 30/10/50) also collides on the flat four
  (`3/90/30/50`), showing no finite aggregate set pre-empts every collision.

## Honest framings kept
- **The loss is upstream of the model (M36).** An aggregation collision is an *information* loss in the
  flatten, not a capacity/tuning/leakage problem; "just use XGBoost / a deeper net" cannot recover an
  identical input.
- **Adding aggregates is a treadmill, not a fix (M37).** Each bespoke feature recovers one dimension for one
  task and must be re-derived + re-leak-checked under drift (Fey issue 5); the space of collisions is
  unbounded. Also: "lossy" ≠ "leaky" — a pipeline can be perfectly PIT-correct and still discard structure.
- **A demonstration of cost, not a win.** L035 shows the single table *destroys recoverable signal*; no
  result yet shows a graph model *recovering* it to beat the honest bar (tuned GBDT + ResNet + AutoML,
  L028–L030). That is the Y1-exit essay's argument and the Y3–Y4 empirical burden.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l035-collision` [misconception], `l035-thesis`).
- `assets/paper-deck.js` +1 (`fey2024-fecost` — the five issues + issue (4) + the REG, keyed to L035; the
  broad `fey2024` card stays at L001).
- `misconceptions.md` **M36** (a better model can recover a lossy flatten → the loss is before the model) and
  **M37** (add more aggregates until complete → unbounded per-task treadmill; lossy ≠ leaky).
- `reference/glossary.html` — Q4 section +5 terms (lossy map, aggregation collision, fine-grain signal,
  structure a flatten discards, relational entity graph).
- `thesis-dossier.md` +1 ledger line (L035, FOR + BAR, C1/C2 — the collision made concrete + the honesty
  guard) and **Current verdict updated to "after L035 / what joins destroy."**
- `RESOURCES.md` — enriched the Fey 2024 (arXiv:2312.04615) entry into the L035 primary-reading note (§1–2,
  the five issues, issue (4), the REG, the Gabor→pixels analogy).
- `lessons/manifest.json` regenerated → **35 entries** (L035 Q4, published); `labs/html/0035-*.html` rendered.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l035.js` 13/13 clean; `_viz_check_l034.js`
  regression clean.

## Next
Lesson 036 — **Revisit your homework pipeline** (audit the user's own `homework/report.md` against the
Q1–Q3 leakage-spine checklist; reuse `checklist.js` + `pipeline-viz.js`). No new concept — it personalises
the exit exam by applying the rubric to the learner's prior work (find one real leak, fix it, re-measure),
and feeds L037 (package the fixed pipeline). Note: L036 needs the user's homework artifact — prompt for it
at the start of that session.
