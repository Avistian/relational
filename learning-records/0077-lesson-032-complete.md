# 0077 — Lesson 032 complete: TabTransformer Preview

**Date:** 2026-07-21
**Status:** Complete (self-reported by the user: "lesson 32 done")
**Curriculum:** Year 1 · Q4 · lecture 032 — second lesson of the Q4 bridge (L031–L040).

## How it was marked complete
The user said "lesson 32 done" (alongside the request to create Lesson 033), with no lab EXIT-ticket text
pasted — so, per the L017–L031 precedent, there is no rubric score for the lab; treated as self-reported
complete. Full teaching content, the two viz, the forward-only torch lab, and the verified numbers are in
the published record [[0076-lesson-032-published.md]].

## Retained (assumed, from the lesson design)
- **The TabTransformer data-flow:** categorical entity embeddings (L031) → a stack of *N* Transformer
  self-attention layers → **contextual** embeddings; continuous features **bypass** the Transformer
  (LayerNorm only) and are concatenated with the flattened contextual vectors before an MLP head. The only
  new machinery is the Transformer in the middle.
- **Contextual ≠ context-free (M31):** self-attention rewrites each column as `softmax(Q·Kᵀ/√d)·V` — a
  weighted blend of every column's value vector — so a column's vector now depends on the rest of the row
  (the "bank" = river vs savings analogy).
- **Honest verdict (M30):** TabTransformer **matches** tree ensembles on supervised tabular; the +1.0% is
  over other deep methods, and its real wins are robustness to noise/missingness and a +2.1%
  semi-supervised lift. Fifth flat-table tie in a row (L028/L029/L030/L031/L032).
- **Thesis hook:** a contextual embedding is a weighted aggregate of related vectors — within-row message
  passing; RDL runs the same operation across tables via foreign keys.

## Next
Lesson 033 — **When to stop feature engineering** (Domingos 2012). Published this same session; see
[[0078-lesson-033-published.md]]. Continues the Q4 consolidation before the "relational data without RDL"
thread (L034–L035).
