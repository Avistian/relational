# Lesson 011 complete — decision trees as partitions

User completed `labs/0011-decision-trees-partitions.ipynb` ("lab 11 done"). Warm-up earlier this session **4/4** on Q1-checkpoint retrieval.

**Lab score (lab-authoring rubric, 0–2 × 5 = /10):**
- Correctness — 2 (depth-1 split read from `tree_`; hand ΔGini matched sklearn within tol after fixing the objective)
- Leakage discipline — n/a for a mechanism lab (tree fit on full data by design to expose the partition); scored 2 for not over-claiming generalization
- Conceptual takeaway — 2 (split = axis-aligned partition; ΔG = purity gain)
- Mid-zone effort — 2 (wrote the Gini function + weighted reduction unaided after one nudge)
- Reproduction — n/a
**Total ≈ 9/10.**

**Real fix that stuck:** Gini is `1 − p² − (1−p)²` (not the positive rate `p`), and ΔG is **parent minus weighted children** (not the weighted children alone). Both were live mistakes the user corrected. Callback: watch that ΔG-vs-weighted-children sign in Lesson 013 (boosting gain) and Lesson 014 (XGBoost regularized gain).

**Process note:** user flagged mid-lesson that labs were too thin on in-notebook explanation → new lab-intro standard (NOTES #8, `lab-authoring` § Introductory content); Lab 011 retrofitted with a concept recap + per-task goal/why.
