# 0080 — Lesson 034 published: Relational Data Without RDL

**Date:** 2026-07-21
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 034 — matches the CURRICULUM row (034: "Relational data without RDL ·
Star schema & joins (Kimball summary) · Sketch 3-table join"). Fourth lesson of Q4 (Consolidation & bridge
to neural tabular, L031–L040) and the opener of the **L034–L035 "relational data without RDL" thread**. Plan
entry: `plan/year-1.md` §034.
**Primary reading:** Ralph Kimball & Margy Ross — *Fact Tables and Dimension Tables*
([Kimball Group](https://www.kimballgroup.com/2003/01/fact-tables-and-dimension-tables/)) + the
[Dimensional Modeling Techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
summary (from *The Data Warehouse Toolkit*, 3rd ed., Ch. 2).

## Single skill
Read a three-table schema (a fact table plus its dimensions, with primary and foreign keys) and write the
**join + point-in-time aggregation** that flattens it into one design matrix at a chosen entity grain —
keeping every aggregate strictly to information available before the prediction time.

## Why this was the ZPD
L033 ended on "the returns moved across the join" but left the *join* metaphorical; L031/L032 built ever-
richer single-table representations. L034 is the curriculum's designated next step and the natural pivot:
make the flatten **literal** so the thesis's target (C1 — "flattening is the lossy step RDL replaces") stops
being a slogan and becomes a mechanism the learner can build and audit. It reactivates the L002/L021/L022
leakage thread (point-in-time correctness, now at the *feature* step) and the L009 DFS thread (each aggregate
is a hand-built primitive), and it is the cleanest possible setup for L035 ("What joins destroy", Fey 2024
§2): once the learner has *built* a lossy flatten by hand, "here is the structure it throws away" lands as a
demonstrated fact.

## What shipped
- **Lesson** `lessons/0034-relational-data-without-rdl.html` (~45 min, standard #17 thoroughness): full
  vocabulary from first principles (relational database, entity, PK, FK, fact/dimension table, star schema,
  grain, one-to-many cardinality, join, aggregation, design matrix, PIT correctness) → the 3-table schema viz
  → "the ML row is one entity at one time" + the flatten viz → predict → the SQL **and** pandas join+aggregate
  with the three correctness gates (grain fixed by GROUP BY, LEFT JOIN keeps zero-order entities, `order_ts<t`
  guard) → the PIT-leakage beat (reused `leakage-viz.js`) → teach-back → thesis bridge (flatten = hand-built
  per-task lossy DFS pipeline; RDL keeps PK/FK as a graph) → 5 subtleties → 3 quizzes → primary reading → lab.
  Forward-references to L035 are **plain text** (lesson not yet published), per the L033 convention.
- **Two new reusable viz (standard #9, one per beat):**
  - `assets/star-schema-viz.js` — the toy schema (customers/orders/order_items) as boxes with PK/FK column
    badges, one-to-many ∞/1 cardinality glyphs, fact-vs-dimension labels, per-table grain captions, and four
    relationship-highlight buttons (whole schema / cust→ord / ord→item / item→products dangling FK).
  - `assets/join-flatten-viz.js` — the flatten mechanism: a chosen customer's nested orders collapse into one
    fixed-width aggregate row (n_orders / total_spend / avg_basket / max_basket) on "Flatten"; a customer
    toggle (C1: 3 orders, C7: 1) shows every entity yields the same columns; the readout names the discarded
    structure (which products, order, identity) — an explicit L035 hook.
  - Reused `assets/leakage-viz.js` (L002) for the point-in-time beat; its CSS (not in the global stylesheet)
    was copied into the lesson's inline `<style>`.
  - Headless check `labs/_viz_check_l034.js` — **17/17 pass**. **Browser MCP unavailable** (headless env; no
    chrome-devtools server) → node verification only, consistent with L021–L033.
- **Lab** `labs/0034-relational-data-without-rdl.ipynb` — **Tier C** (three small synthetic tables), pandas-
  only, runs in seconds. Kimball is a modeling summary, not an architecture, so the standard #18 scope (stated
  in the intro) is **operationalise the flatten**, not implement a model. Deterministic toy DB (cutoff
  t = 2024-06-01; C1 carries a future $999 order as the leakage trap; C4 has only a future order and C5 none —
  the LEFT-JOIN/fillna test). Three crucial fragments: Task 1 = PIT filter + `groupby("customer_id")`
  aggregation; Task 2 = the leak-free design matrix (`LEFT JOIN` + `fillna(0)`, one row per customer); Task 3 =
  prove the guard is load-bearing (leaky all-orders aggregate). Stretch = a two-hop `customer→orders→items`
  distinct-product count. Student blank (8 `____`, 0 outputs); solution executed clean (all 3 CHECK pass, EXIT
  + stretch correct) & gitignored. Built via `_build_l034.py`; rendered to `labs/html/0034-*.html`.

## Verified live (executed solution; deterministic synthetic DB, cutoff t = 2024-06-01)
- **Task 1** (PIT + customer-grain aggregation) — C1: n_orders_before_t=**3**, total_spend=**125**,
  avg_basket≈**41.67**, max_basket=**55** (the future $999 July order correctly excluded). C2: 1/120. C3:
  2/45. C6: 2/160. C4 absent (its only order is after t).
- **Task 2** (design matrix) — exactly **6 rows, one per customer**; C4 and C5 kept via LEFT JOIN with
  0-order aggregates filled to 0. Grain: one row = one customer as of t.
- **Task 3** (guard is load-bearing) — dropping `order_ts < t` changes C1 from **n=3/total=125** to
  **n=4/total=1124** (the future $999 leaks in). Safe ≠ leaky, as asserted.
- **Stretch** (two-hop) — n_distinct_products from pre-t items: C1=2, C2=1, C3=1, C6=1, C4/C5=0.

## Honest framings kept
- **The flat table is manufactured, not given (M34).** It is a per-task, hand-written pipeline of DFS-style
  primitives that deliberately discards relational structure — the point of the lesson, and the thesis's target.
- **PIT is about every feature, not just the label (M35).** An aggregate leaks if it includes post-t events
  even when the label column is untouched; the `order_ts < t` filter must guard *every* aggregate, *every*
  re-flatten. Dropping the timestamp afterwards does not undo the leak.
- **The demonstration is mechanical, not yet a win.** L034 shows the *cost* of flattening; no result yet shows
  a model *recovering* the discarded structure to beat the fair bar — that is L035's setup and the
  Y1-exit → Y3–Y4 burden.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l034-flatten` [misconception], `l034-thesis`).
- `assets/paper-deck.js` +1 (`kimball2013` — fact/dimension, PK/FK, grain, and the flatten recipe).
- `misconceptions.md` **M34** (flat table is a given → hand-built lossy flatten) and **M35** (PIT is only
  about the label → every aggregate must filter to before t).
- `thesis-dossier.md` +1 ledger line (L034, FOR, C1/C2 — the flatten made literal and auditable) and
  **Current verdict updated to "after L034 / relational data without RDL."**
- `reference/glossary.html` — Q4 section +9 terms (relational database, PK, FK, fact/dimension table, star
  schema, grain, one-to-many, flatten, PIT correctness).
- `RESOURCES.md` — Year 1 +1 (Kimball & Ross dimensional modeling: the free article + techniques summary).
- `lessons/manifest.json` regenerated → **34 entries** (L034 Q4, published); `labs/html/0034-*.html` rendered.
  `NOTES.md` Session 34 logged.
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l034.js` 17/17 clean.

## Next
Lesson 035 — **What joins destroy** (Fey et al., *Position: Relational Deep Learning*, ICML 2024 §2 — the
feature-engineering cost; ★ preview): enumerate the structure a flatten-then-aggregate pipeline discards
(cardinality, identity, higher-order paths, temporal order within a neighbour set). The intellectual pivot of
Year 1. Planned new viz `flatten-loss-viz.js` (two entities with different neighbour structure collapsing to
the same feature row — aggregation collision).
