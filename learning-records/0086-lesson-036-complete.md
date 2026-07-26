# 0086 — Lesson 036 complete: Revisit Your Homework Pipeline

**Date:** 2026-07-26
**Status:** Complete (self-reported by the user: "lesson 36 done")
**Curriculum:** Year 1 · Q4 · lecture 036 — sixth lesson of the Q4 bridge (L031–L040) and the first lesson in
the workspace whose subject matter is the learner's own code.

## How it was marked complete
The user said "lesson 36 done" (alongside the request to create Lesson 037), with no lab EXIT-ticket text
pasted — so, per the L017–L035 precedent, there is no rubric score for the lab; treated as self-reported
complete. The full teaching content, the `nested-calib-viz` group-straddle widget, the Tier-A audit lab, and
every verified number are in the published record [[0084-lesson-036-published.md]].

## Retained (assumed, from the lesson design)
- **An audit's output is a ranked triage, not a count of smells.** Each finding is assigned a
  **consequence class** *before* its size is estimated: does it inflate the **reported number**, degrade the
  **shipped artifact**, change the **decision**, or is it only a **declarable limitation**?
- **"Leak" ≠ "inflated metric" (M38).** `CalibratedClassifierCV(cv=5)` takes an ungrouped nested split inside
  a correctly person-grouped outer fold (675 of 4,851 labelled persons straddle; 1,411 rows). Because the
  mis-split is confined to the training block, the reported metrics stayed honest — the damage was to the
  calibrator. Fixed: log-loss 1.4248 → 1.4232, ECE 0.0363 → 0.0360.
- **Consequence class ≠ measured size.** The transductive encoder — the one class that *can* inflate a
  reported number — was re-measured in-fold at **1.4237 vs 1.4248**, i.e. the honest arrangement is
  0.0011 nats *better*: an effect of zero, drifting opposite to the leakage story. The class says where to
  look; only the re-measurement says how much.
- **The most expensive defect broke no measurement (M39).** M2a shipped over M1 on Δ = 0.0032 nats = 8 % of
  one fold's σ, losing 2 of 5 folds, with the winner flipping when fold 2 is dropped. Every number was
  correct and the wrong artifact went to production. Also: the report's only "significant" claim
  (M1−M0, naive p = 0.0146) dies under Nadeau–Bengio (0.0514) and Holm (0.0583) while the *effect* survives.
- **An audit must be able to return "correct."** Seven spine questions passed, two above standard practice.
- **Reproducibility findings make every other finding checkable.** `n_jobs=-1` cost 210 s per multiclass fit
  against 9.2 s at `n_jobs=6` on this 12-core box — the difference between re-measuring a finding and
  skipping it. This observation is what L037 is built on.

## Next
Lesson 037 — **Document a baseline package**: turn the audited submission into an artifact that can
regenerate its own headline number on demand, and can tell you when it failed to. Published this same
session; see [[0087-lesson-037-published.md]].
