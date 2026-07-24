# 0081 — Lesson 034 complete: Relational Data Without RDL

**Date:** 2026-07-24
**Status:** Complete (self-reported by the user: "lesson 34 done")
**Curriculum:** Year 1 · Q4 · lecture 034 — fourth lesson of the Q4 bridge (L031–L040) and the opener of
the L034–L035 "relational data without RDL" thread.

## How it was marked complete
The user said "lesson 34 done" (alongside the request to create Lesson 035), with no lab EXIT-ticket text
pasted — so, per the L017–L033 precedent, there is no rubric score for the lab; treated as self-reported
complete. Full teaching content, the star-schema + join-flatten viz, the Tier-C 3-table flatten lab, and
the verified numbers are in the published record [[0080-lesson-034-published.md]].

## Retained (assumed, from the lesson design)
- **The flatten = join + point-in-time aggregation.** To feed any Q1–Q3 model you pick a **grain** (one row
  per entity at a prediction time `t`), LEFT JOIN the neighbour fact tables via foreign keys, and aggregate
  the one-to-many rows into fixed-width columns (`COUNT`/`SUM`/`AVG`/`MAX`) — with `order_ts < t` guarding
  **every** aggregate so no feature uses the future.
- **Three correctness gates:** the `GROUP BY`/`groupby` key fixes the grain (else you train on duplicated
  order-level rows); `LEFT JOIN` + `fillna(0)` keeps zero-order entities (else a biased training set); the
  PIT filter lives *inside* the aggregation, not bolted on afterwards (dropping the timestamp column does
  not undo a leak already baked into an aggregate).
- **Verified (synthetic DB, t = 2024-06-01):** C1 safe = n=3/total=125/avg≈41.67/max=55 (the future $999
  July order excluded); leaky (no cutoff) = n=4/total=1124 — the difference is the leaked future, proving the
  guard is load-bearing. Design matrix = exactly 6 rows, one per customer; C4/C5 survive via LEFT JOIN.
- **Thesis hook (M34/M35):** the flat table is *manufactured*, not given — a per-task, hand-written pipeline
  of DFS-style primitives that deliberately discards relational structure. That set-up is exactly what L035
  now measures the cost of.

## Next
Lesson 035 — **What joins destroy** (Fey et al. 2024 §2, ★ preview): enumerate the structural information a
flatten-then-aggregate pipeline discards. Published this same session; see [[0082-lesson-035-published.md]].
The intellectual pivot of Year 1.
