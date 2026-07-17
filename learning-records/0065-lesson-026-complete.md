# 0065 — Lesson 026 complete: Inductive Bias — Rotation

**Date:** 2026-07-17
**Status:** Complete (self-reported by the user: "lesson 26 completed")
**Curriculum:** Year 1 · Q3 · lecture 026 — third lesson of the Grinsztajn arc (024–027), second of the
three inductive-bias mechanism lessons.

## How it was marked complete
The user said "lesson 26 completed" with no EXIT-ticket text pasted, so — per the L017–L025 precedent —
there is no rubric score for the lab; treated as self-reported complete. The published record with the
full teaching content and verified numbers is [[0064-lesson-026-published.md]].

## Retained (assumed, from the lesson design)
- Rotational invariance (Ng 2004): same orthogonal `Q` on train + test leaves an MLP's score unchanged
  (`W·(Qx)=(WQ)·x`); a tree is not invariant (axis-aligned splits tied to the original basis), nor is an
  FT-Transformer (pointwise tokenizer).
- The rotation experiment reverses the ranking (tree 0.987→0.747 vs MLP 0.862→0.869, +0.008) because a
  rotation is lossless — it only destroys the alignment of the signal with the meaningful axes.
- Ng's bridge to L027: an invariant learner's worst-case sample complexity grows ≥ linearly in the number
  of uninformative features.

## New teaching-standard directive captured this session
The user asked that **everything introduced in a lesson be thoroughly explained, even if it makes the
lesson longer**, and that the decision persist for all future lessons. Recorded as **NOTES.md Preferences
standard #17** and a new §6 in the `lesson-pedagogy` skill. First applied in L027.

## Next
Lesson 027 (Inductive bias: uninformative / junk features — Grinsztajn 2022 **§5.3**, Finding 2), the
last of the three mechanism lessons, linked to L026 by Ng's theorem. Then the Grinsztajn arc closes and
Q3 continues (L028 = MLP/ResNet tabular baselines).
