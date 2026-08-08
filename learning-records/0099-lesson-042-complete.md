# 0099 — Lesson 042 complete: MLP & ResNet baselines (do these first)

**Date:** 2026-08-08
**Status:** Complete (self-reported by the user: "lesson 42 done")
**Curriculum:** Year 2 · Q1 · lecture 042 (Advanced Tabular Deep Learning).
Topic: *MLP & ResNet baselines (do these first)*; curriculum lab: *Train ResNet baseline*.

## How it was marked complete
The user said "lesson 42 done" (alongside the request to create Lesson 043). The lab's three test scores,
the corrected-test verdict, and the "why run the baselines first" sentence were not pasted into chat, so
per the L017–L041 precedent there is no fresh hostile-reader rubric score; treated as self-reported
complete. The full teaching content, the verified numbers, and the rigor retrofit are in
[[0096-lesson-042-published.md]] and [[0097-rigor-upgrade-rtdl-and-multidataset.md]].

## Retained (assumed, from the lesson design)
- **The baseline-first rule (M51).** Beating a GBDT is not the same as clearing the neural bar. The strong
  simple baselines (tuned MLP, tuned ResNet) are run *first*, because a "win" over a weak or
  unequally-tuned net measures a **tuning-effort gap**, not model quality (L038).
- **Shared frame vs per-model search space.** What must be identical across arms is the *frame* — split,
  metric, search budget, selection rule. The search *space* is legitimately per-model. Holding the frame
  fixed is what makes two numbers comparable at all.
- **Build from scratch; libraries validate (#22).** The from-scratch `TabResNet` matched rtdl to
  |Δ| = 0.000 (credit_g) and 0.001 (diabetes) — the library is a *validation point*, not the teacher.
- **Never conclude from one dataset (#23).** The four-dataset run (mean ranks MLP 1.25 / ResNet 1.75 /
  GBDT 3.00, Friedman p = 0.039) looked like "nets significantly beat the GBDT" — but 3 of the 4 tables
  were all-numeric, exactly the regime Grinsztajn shows favours nets. A tiny biased sample can produce a
  "significant" p that does not generalise; the representative "no universal winner" verdict is grounded
  in Grinsztajn's ~45 datasets, not in our 4.
- **The degradation problem (M52).** A deep *plain* MLP scoring worse than a shallow one is not
  overfitting — training error rises too; skip connections are what make depth trainable (L028).

## Publishing note (this session)
Closing out L042 surfaced that the Pages deploy for the **paper-mirror doctrine commit** (`be919fbe`) had
failed, though the site itself was never stale for L042. Root cause and fix are recorded in
[[0100-pages-deploy-fix.md]].

## Next
Lesson 043 — **TabNet (sequential attention)** (Arik & Pfister 2019, `1908.07442`; curriculum lab:
*train + read masks*). The first "novel" architecture of Y2 Q1, and the first chance to hold one to the
baseline-first rule established here — plus the first paper unit to apply the paper-mirror doctrine (#24)
end to end.
