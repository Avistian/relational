# 0097 — Two standing directives: build from scratch (rtdl validates) + never conclude from one dataset

**Date:** 2026-08-05
**Status:** Adopted as durable standards; applied retroactively to L042 as the reference implementation.
**Trigger:** User (`/teach`), two remarks after L042 shipped:
> 1. use rtdl as a validation point, at least for some models that are available there, most of the models to learn anything should be built from scratch.
> 2. Do you think that comparing and making conclusions on one dataset is enough? Make sure that future lessons will be as research rigorous as possible. Find solutions.

## The decisions (now binding on all future lessons)

- **NOTES standard #22 — build from scratch; libraries only validate.** The load-bearing model/mechanism
  is implemented by the learner; a reference library (rtdl for MLP/ResNet/FT-Transformer) is used *only*
  to validate it (same forward pass on copied weights, or scores within a tolerance under the same
  protocol). Reusable from-scratch models are promoted into `labs/relkit/` so later labs build on them.
  Exception: a lesson whose skill *is* the tool (e.g. L041 "set up rtdl").
- **NOTES standard #23 — never conclude from one dataset.** Any comparative claim ("A beats/ties B",
  above all "no universal winner") needs ≥3 real datasets, per-dataset mean ± std over ≥3 seeds, and a
  **rank-based** cross-dataset summary (mean ranks + Friedman + a Nemenyi critical-difference diagram —
  reuse `assets/cd-diagram-viz.js`). When our compute can't reach the N the strong claim needs, cite the
  large published benchmark (Grinsztajn 2022 ≈45 datasets; Gorishniy 2021) and separate "verified here on
  k" from "established in the literature on N". A single-dataset result is a *demonstration*, never proof.
- Both are also written into the `lab-authoring` skill.

## What changed in L042 (the reference implementation)

- **New from-scratch module** `labs/relkit/nets.py` — `TabResNet` / `TabMLP` (promoted from L028) + a fair
  early-stopping `train_net`. Registered 5 small OpenML tables in `relkit.data` (diabetes, blood_transfusion,
  kc1, phoneme, churn) for cheap multi-dataset runs.
- **Re-verified** (`labs/_verify_l042.py`, 4 datasets × 3 seeds, from-scratch models **validated against
  rtdl**, `_verify_l042_results.json`):
  - rtdl validation: from-scratch ResNet ≈ rtdl exactly — |Δ| = **0.000** (credit_g), **0.001** (diabetes).
  - Per-dataset (mean ± std): credit_g MLP 0.802/ResNet 0.790/GBDT 0.780; diabetes 0.816/0.811/0.805;
    blood_transfusion 0.740/0.735/0.727; kc1 ResNet 0.814/MLP 0.795/GBDT 0.789.
  - **Mean ranks** MLP 1.25, ResNet 1.75, GBDT 3.00; **Friedman p = 0.039**.
- **Honesty catch (the real rigor lesson).** Taken at face value the four-dataset run says "the tuned nets
  significantly beat the GBDT" — which *contradicts* the literature. The sample is the problem: 4 tables,
  3 of them all-numeric — exactly the regime Grinsztajn shows favours nets. A tiny, biased sample yields a
  "significant" p that does not generalise. So the lesson now (a) reports credit_g as one datapoint, (b)
  shows the four-dataset rank table, (c) explicitly flags the numeric-skew and refuses to over-claim, and
  (d) grounds the representative "no universal winner" in Grinsztajn's ~45 datasets. This turned the
  single-dataset weakness into the teaching point.
- **Lab rebuilt** (`labs/_build_l042.py` → notebooks + rendered HTML): Task 1 = the fair early-stopping
  loop from scratch; **Task 2 = validate the from-scratch ResNet against rtdl** (|Δ| < 0.04); Task 3 =
  multi-dataset bake-off + mean-rank verdict. Solution executed clean (Task 2 |Δ|=0.022 VALIDATED; ranks
  ResNet 1.33 / MLP 1.67 / GBDT 3.00; winner flips between the nets).
- **Reconciled** everywhere to the new numbers + framing: lesson HTML (new "across datasets" section,
  build-from-scratch defbox, two new quizzes on multi-dataset rigor and rtdl-validation, updated
  predict/teachback), `protocol-bakeoff-viz.js`, `misconceptions.md` M51, `thesis-dossier.md` L042 entry +
  narrative, `requirements-labs.txt`. Cache-buster bumped to `4`.

## Note for future me

- L028's earlier evidence-of-record still shows the *less-tuned* nets losing to GBDT on credit_g
  (0.793 vs 0.752/0.743). L042's better-tuned, rtdl-validated nets tie/edge it. Left L028's assets alone
  (their numbers are that run's truth), but be aware the two runs differ because of tuning — a possible
  future clean-up is to re-verify L028 under the fair protocol.
- The `cd-diagram-viz.js` asset exists (L023/L030) and is the intended tool for the CD-diagram half of
  standard #23 in future comparison lessons.
