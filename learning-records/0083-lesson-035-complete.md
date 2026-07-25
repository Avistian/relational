# 0083 — Lesson 035 complete: What Joins Destroy

**Date:** 2026-07-25
**Status:** Complete (self-reported by the user: "lesson 35 done")
**Curriculum:** Year 1 · Q4 · lecture 035 — fifth lesson of the Q4 bridge (L031–L040) and the close of the
L034–L035 "relational data without RDL" thread. **The intellectual pivot of Year 1.**

## How it was marked complete
The user said "lesson 35 done" (alongside the request to create Lesson 036), with no lab EXIT-ticket text
pasted — so, per the L017–L034 precedent, there is no rubric score for the lab; treated as self-reported
complete. Full teaching content, the `flatten-loss-viz` collision widget, the Tier-C collision lab, and the
verified numbers are in the published record [[0082-lesson-035-published.md]].

## Retained (assumed, from the lesson design)
- **Aggregation is a lossy map.** JOIN + GROUP BY sends many neighbour rows to a fixed-width vector; two
  entities with genuinely different neighbourhoods can land on the **same** row — an **aggregation
  collision**. Verified: Ada and Bo both flatten to `n=3 / total=90 / avg=30 / max=50`.
- **Four structures a flatten discards:** cardinality, event/entity identity (*which* items, not how many),
  temporal order within a neighbour set, and higher-order (multi-hop) paths.
- **The loss is upstream of the model (M36).** A fitted `LogisticRegression` gives **P(churn) = 0.502 for
  both** Ada and Bo though their true labels differ (0 vs 1) — identical input ⇒ identical output. No amount
  of capacity, tuning, or a deeper net recovers information the flatten already destroyed.
- **Adding aggregates is a treadmill, not a fix (M37).** `spend_trend` (+40 / −40) and
  `n_distinct_products` (3 / 1) do separate Ada and Bo — but a third customer (Zoe: 30/10/50) collides again
  on the same four columns. Each bespoke feature recovers one dimension for one task; the space of collisions
  is unbounded. Also: **lossy ≠ leaky** — a pipeline can be perfectly PIT-correct and still discard structure.
- **A demonstration of cost, not a win.** L035 shows the single table destroys recoverable signal; nothing
  yet shows a graph model *recovering* it to beat the honest bar (tuned GBDT + ResNet + AutoML, L028–L030).
  That gap is the Y1-exit essay's argument and the Y3–Y4 empirical burden.

## Next
Lesson 036 — **Revisit your homework pipeline**: audit the user's *own* prior work
(`/home/avist/Projects/homework`, the ReAction L&D response-prediction submission) against the Q1–Q3
leakage spine, then fix one real defect and re-measure. Published this same session; see
[[0084-lesson-036-published.md]]. Turns the diagnostic arc L001–L035 back on the learner's own code — the
first lesson in the workspace whose subject matter *is* the learner.
