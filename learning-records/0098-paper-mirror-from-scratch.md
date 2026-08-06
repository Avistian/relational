# 0098 — Paper-mirror doctrine: from-scratch impl + paper datasets + reproducibility

**Date:** 2026-08-06
**Status:** Adopted as durable standard #24; consolidates #18 / #22 / #23 / #20 honesty.
**Trigger:** User (`/teach`):
> make sure that future lessons/labs will always try to mirror paper implementations from scratch,
> so we can learn as much as possible from papers. Also in terms of datasets and reproducibility
> point of view.

## Decision

When a lesson's core source is a paper, the unit is a **paper mirror**, not a loosely inspired exercise.
Three axes are required unless a narrow exception is stated in the lab intro:

1. **Implementation** — load-bearing algorithm/architecture from scratch (#18 + #22); reference libs
   validate only; cells cite paper figure/section/equation.
2. **Datasets** — prefer the paper's own data/splits/preprocessing; if substituted, name what was
   skipped and which claim cannot be reproduced; comparative claims still obey multi-dataset rigor (#23).
3. **Reproducibility** — seeds (model + splitter), versions/lockfile, identical protocol across arms,
   `_verify` + results JSON, EXIT = paper metric within tolerance or honest fail (L037 ledger pattern;
   #20 honesty on downscales).

## Implications for future sessions

- Authoring checklist lives in `NOTES.md` Preferences #24 and `.agents/skills/lab-authoring/SKILL.md`.
- Does not replace mid-zone difficulty, lab-with-lesson (#21), or compute escalation — it specifies
  *what* those standards should build when the source is a paper.
- Harness code still accumulates in `relkit/`; that is not license to import the model from a library.
- Reference stack already in the curriculum: L032 (architecture mirror), L037 (repro ledger),
  L042 (from-scratch + multi-dataset + rtdl validate). Next paper unit after L042 should apply #24
  end-to-end and call it out in the learning record.
