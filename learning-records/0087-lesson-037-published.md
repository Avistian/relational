# 0087 — Lesson 037 published: Document a Baseline Package

**Date:** 2026-07-26
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 037 — seventh lesson of Q4 (Consolidation & bridge to neural tabular,
L031–L040) and the second built on the learner's own submission. Where [[0084-lesson-036-published.md]] asked
whether the reported number was **right**, this asks whether it is **stable** — whether running the pipeline
again gives it back, and whether you would find out if it did not.
**Primary reading:** Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox & Larochelle,
*Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility
Program)*, JMLR **22(164)**, 2021 — [jmlr.org/papers/v22/20-303.html](https://jmlr.org/papers/v22/20-303.html);
§5 and the ML Reproducibility Checklist in Appendix Fig. 8. Companions: Raff 2019
([arXiv:1909.06674](https://arxiv.org/abs/1909.06674), 255 papers hand-implemented, **63.5 %** reproduced) and
Sculley et al. 2015 §6 on configuration debt. Terminology from
[ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current).

## Single skill
Turn an audited pipeline into a package that **regenerates its own headline number on command and fails loudly
when it cannot** — by measuring which inputs actually move the number rather than asserting it, naming the
number precisely enough that two runs can be compared at all, and pre-registering a reproduction tolerance.

## Why this was the ZPD
L036 left the learner holding a triaged list of defects in their own code and a fix diff. The obvious next
move — "apply the fixes" — is not a lesson; it is an afternoon. The non-obvious one is the precondition
underneath both L036 and the entire Y3–Y4 comparison: an audit is only worth doing on an artifact that can be
re-run, and a "win" is only evidence if the loser's number can be regenerated. The learner already had the
statistical discipline (L023 variance, L030 honest benchmark reports) but had never asked the *engineering*
question — and had a concrete, personally-owned instance of it to hand, since the L036 audit was nearly
skipped over a 23× thread-contention penalty. That observation is what this lesson is built on.

## What shipped
- **Lesson** `lessons/0037-document-a-baseline-package.html` (~40 min, standard #17): cold-open warm-up →
  the three rungs (repeatability / reproducibility / replicability) and why the nouns are unreliable →
  constraint vs pin vs lock → the perturbation ledger, with a prediction gate before the reveal → the
  estimator-of-record problem and the noise floor → which seed → choosing a tolerance → the package layout
  (`pyproject.toml`, `uv.lock`, `src/`, `baseline.yaml`, `make verify`, the emitted manifest) → the packaging
  rubric checklist → teach-back → thesis bridge → subtleties → 3 quizzes → primary reading → lab.
- **Three new reusable viz (standard #9), all `file://`-safe plain scripts:**
  - `assets/repro-probe-viz.js` — the pipeline as a six-stage chain; each knob perturbs exactly one stage and
    the OOF fingerprint chip either matches the reference or does not. Nine measured configurations. CSS
    prefix `.rp-`.
  - `assets/ece-estimator-viz.js` — the same predictions binned pooled vs per-fold vs one 107-row slice,
    drawing the populated bins so the mechanism is visible (17 rows in the sparsest pooled bin; **2 rows**
    producing a 0.40 gap per-fold), with the measured noise floor quoted against each. CSS prefix `.ee-`.
  - `assets/tolerance-gate-viz.js` — the four measured landmarks on a log axis with a draggable `tol`,
    naming what a gate there catches and what it waves through. CSS prefix `.tg-`.
  - Reused `checklist.js`, `quiz.js`, `predict.js`, `teachback.js`, `retrieval-bank.js`.
  - Headless check `labs/_viz_check_l037.js` — **51/51 pass**; `_check_pedagogy.js` clean; every CSS class
    emitted by the three widgets verified present in the lesson's stylesheet. **Browser MCP unavailable**
    (headless env) → node verification only, consistent with L021–L036.
- **Lab** `labs/0037-document-a-baseline-package.ipynb` — **Tier A**, seeded Tier-C stand-in when
  `~/Projects/homework` is absent. Standard #18: the primary reading is a *protocol*, so the crucial fragments
  are the machinery that enforces it — Task 1 `content_hash()` + `build_manifest()`, Task 2 `fingerprint()` +
  `probe()`, Task 3 `ece()` + a named-estimator registry + `noise_floor()` + `assert_reproduces()` (made to
  fail on purpose). Stretch: write the workspace's missing lockfile. 120 trees / 40-column sample so five CV
  runs finish in ~2 min; stated in the intro.
- **`labs/relkit/repro.py`** — deliberately partial: `git_state()`, `env_versions()`, `host_info()` only. The
  load-bearing half is what the student writes, and the lab tells them to append it here afterwards.
- **Harnesses** (every number reproducible): `labs/_repro_l037.py` (threads × `deterministic`, 9 configs),
  `_repro2_l037.py` (histogram mode, row order, dtype, model seed), `_repro3_l037.py` (splitter seed),
  `_repro_env_l037.py` (cross-environment), `_ece_estimator_l037.py` (estimator + noise floor), plus the
  matching `_*_results.json` and `_timing_l037_results.json`.

## Verified live (real homework data, 5,587 labelled rows, person-grouped 5-fold, LightGBM 400 trees, uncalibrated)
Reference configuration: `n_jobs=6`, seed 0, float32 input, lightgbm 4.6.0 / scikit-learn 1.9.0 / numpy 2.5.0
→ OOF sha256 **`d2f0e4bf9b4fd761`**, mean log-loss **1.632168**, fold σ **0.035009**.

| Perturbation | Verdict | Detail |
| --- | --- | --- |
| rerun, byte-identical command | **identical** | rung 1 passes |
| `n_jobs` ∈ {1, 2, 12} | **identical** | wall time only |
| `deterministic=True` + `force_row_wise` at all 4 thread counts | **identical** | insurance already held |
| `force_row_wise` vs `force_col_wise` | **identical** | the mode LightGBM otherwise picks *by timing your machine* |
| shuffled training-row order | **identical** | 255-bin histograms make the sums order-independent |
| model seed ∈ {1, 2, 3, 4} | **identical** | the seed is **inert** |
| **float32 → float64** | **moves** | sha `a4377f2a443dc970`; **258/5,587 argmax flips (4.6 %)**; max \|Δp\| **0.326**; Δ mean log-loss **+0.00133** |
| **lightgbm 4.6.0 → 4.5.0** | **crashes** | `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'` |
| **splitter seed 0 → 1–4** | **moves** | mean log-loss **1.6191–1.6357, range 0.0166 nats** |

Clean best-of-3 per-fit timings on a quiet box: **9.81 s** at `n_jobs=1`, 6.01 at 2, **3.52 at 6**, 13.74 at 12;
`deterministic=True` costs ~30 % (4.58 s at 6).

**Estimator of record.** Recomputed from the submission's own saved `oof_M2a_lgbm_km.npz`: pooled ECE
**0.017760** vs mean-over-folds **0.033200** — the same metric on the same predictions, **1.87×** apart, while
log-loss agrees to five decimals either way. Noise floor from a control that is perfectly calibrated by
construction (labels resampled from the model's own probabilities, true ECE = 0):

| n | mean-ECE of the model | noise floor | signal |
| --- | --- | --- | --- |
| 5,587 (pooled) | 0.0178 | **0.0149** | +0.003 |
| 1,117 (one fold) | 0.0344 | **0.0335 ± 0.0087** | +0.0009 |
| 107 (the failing ship-gate slice) | 0.094 | **0.1071 ± 0.0297** | *below the floor* |

Re-cutting the pooled rows into five **random** blocks of 1,117 — no fold structure, no grouping — gives
0.0343 ± 0.0034, inside which the real folds' 0.0332 sits: the gap is the binned estimator's small-sample
bias, not a fold effect. The floor follows `a + b/√n` closely (fit on the two largest n alone: `a = 0.0042`,
`b = 1.017`, predicting the other four within 3.5 % — an extrapolation of a heuristic form, flagged as such).

## Honest framings kept
- **Eight of nine perturbations changed nothing, and that is the result.** The lesson resists the temptation to
  manufacture drama: on this pipeline the reproducibility risk is *not* where the folklore says it is. Saying
  "thread count is free here" is only permitted because it was run.
- **The seed is inert, and "all randomness is seeded" is true and uninformative (M41).** LightGBM reads its RNG
  only when it samples, and this config does not. Meanwhile the *same literal* `RANDOM_STATE = 0` handed to the
  splitter is the largest controllable term in the whole report.
- **Aggregate damage and artifact damage are different quantities (M42).** The dtype cast moved mean log-loss
  by +0.00133 — which most gates would wave through — while changing what 258 people are predicted to do. It is
  also **42 % of the 0.0032 margin that chose which model shipped** (L036), so the "negligible" number is not
  negligible against the decision it feeds.
- **The most likely failure mode is not a different number, it is a crash.** The version rollback did not drift;
  it raised `TypeError`. Two packages, each satisfying this workspace's own constraints, that cannot be in the
  same room.
- **A gate that cannot fail is a comment.** `tol` is set to 0.0 here *because bitwise was achievable* — eight of
  nine perturbations were byte-identical, so the strictest gate produces no false alarms. That will stop being
  true the moment a GPU reduction or early stopping enters (Y2+), and the lesson says so rather than
  generalising from one lucky pipeline.
- **The estimator choice matters least when the number is bad.** Measured in the lab: on the uncalibrated
  120-tree model (ECE ≈ 0.093, ~8× its floor) the two estimators agree within 1 %; on the calibrated shipped
  model they differ by 1.87×. So the ambiguity bites exactly when you are about to ship — and exactly when you
  would most like to believe the smaller figure.
- **One failed ship-gate cannot be passed by anything.** §4.7's CONDITIONAL verdict on the 107-row
  `age = missing` slice compares 0.094 to a 0.05 threshold, when a perfect model scores 0.1071 ± 0.0297 there.
  There may be a real problem in that slice; this measurement cannot see it.

## Lab verified live (executed solution, real homework data, reduced model)
All three CHECK cells pass, EXIT ticket and stretch print correctly. Reduced config (120 trees, 40-column
sample → mean log-loss 1.628215, fold σ 0.0125), stated in the intro:
- **Task 1:** manifest emits git `9d73c5fecca0` (dirty), the resolved env, host (12 cores, aarch64 WSL2),
  data hash `6012c6563ade6bd3`, config hash `eb572736f54c1969`. Checks confirm the config hash is blind to key
  order but sensitive to a bumped `n_estimators`, and that the data and config hashes move independently.
- **Task 2:** rerun / `n_jobs 4→1` / model seed `0→7` all hash to `ea2b12e9d719af7e`; float64 gives
  `d20116a42b98d4f7` with **431 flips (7.7 %)**, max |Δp| 0.195, Δ log-loss −0.00077. The flip count differs
  from the full-config 258 (4.6 %) because it is a different, cheaper experiment — noted rather than smoothed.
- **Task 3:** pooled 0.0929 (floor 0.0120) vs mean-over-folds 0.0936 (floor 0.0267) — the **values** agree
  within 1 % while the **floors** differ by 122 %, which is the reframed finding above. The gate passes on a
  faithful rerun at `tol = 0.0` and fails on the float64 run with a message naming expected, observed, delta
  and tol.
- **Stretch:** wrote `requirements-labs.lock.txt` — **121 pinned packages** behind the 8 lines of
  `requirements-labs.txt`.

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l037-inert-seed` [misconception], `l037-fingerprint`, `l037-noise-floor`);
  pool now 56 items, ids unique, all validated.
- `misconceptions.md` **M41** (a seed makes a pipeline reproducible), **M42** (agreeing on the headline metric
  means the runs reproduced), **M43** (the same ECE reported twice must say something about the model).
- `reference/glossary.html` — Q4 section +8 terms (repeatability/reproducibility/replicability, constraint vs
  pin vs lock, run manifest, output fingerprint, perturbation probe, inert seed, estimator of record, noise
  floor, reproduction tolerance, configuration debt).
- `assets/paper-deck.js` +1 card `pineau2021` (standard #16 — a new ★ paper).
- `thesis-dossier.md` — current-verdict section extended: L037 completes the **credibility precondition**, and
  the splitter-seed spread (0.0166 nats) sets a floor on how small a Y3–Y4 RDL-vs-GBDT delta can be and still
  mean anything.
- `RESOURCES.md` — Pineau et al. 2021 added under Year 1 with the Raff and Sculley companions and the ACM
  terminology note.
- `NOTES.md` **#19** extended — the full L037 reproducibility ledger, the clean timing table, and the new
  lockfile, so L038–L040 and the Y1 exit can rely on them.
- `lessons/manifest.json` → **37 entries** (L037 Q4, published); `labs/html/0037-*.html` rendered.
- **New in the repo root:** `requirements-labs.lock.txt` (121 packages), written by the lab's stretch task.

## Next
Lesson 038 per `CURRICULUM.md`. The packaging instruments built here — manifest, fingerprint, probe, named
estimator, noise floor, tolerance gate — are the ones the Y1 exit essay (L040) and every Y3–Y4 RelBench
reproduction will lean on; `labs/relkit/repro.py` is where the learner's completed versions belong.
