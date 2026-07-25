# 0084 — Lesson 036 published: Revisit Your Homework Pipeline

**Date:** 2026-07-25
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 036 — matches the CURRICULUM row (036: "Revisit your homework pipeline ·
Your `homework/report.md` · Audit CV + missingness"). Sixth lesson of Q4 (Consolidation & bridge to neural
tabular, L031–L040) and the **first lesson in the workspace whose subject matter is the learner's own code**.
Turns the L001–L035 diagnostic apparatus back on the person who built it.
**Primary reading:** the learner's own `~/Projects/homework/report.md` §3.1–§3.4 read beside
`src/modeling.py` — as a reviewer, not as the author. Method backing (no new ★ paper): Nadeau & Bengio,
*Inference for the Generalization Error*, Machine Learning 52(3) 2003,
[doi:10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366) — the `(1/n + n_test/n_train)`
correction already assigned at [L023](../lessons/0023-statistical-comparison.html).

## Single skill
Audit a finished pipeline against the Q1–Q3 leakage spine and **triage each finding by consequence class** —
does it *inflate the reported number*, *degrade the shipped artifact*, *change the decision*, or is it only a
*declarable limitation*? — then re-measure before believing any of it.

## Why this was the ZPD
L035 closed the diagnostic arc conceptually (the flatten is lossy). The spine itself — grouped CV (L003/L004),
preprocessing inside the fold (L005), PIT/label leakage (L002/L022), proper scoring + calibration (L008),
paired-fold variance (L023), nested selection (L017), temporal protocols (L021) — has been assembled piece by
piece but never **run as one instrument on real, personally-owned code with something at stake**. The learner
had a finished, careful, already-submitted pipeline sitting outside the workspace; the highest-value next step
was not another concept but the transfer test: can the rubric find real defects in work its author believed
was clean, and can each finding be priced rather than merely announced? It also sets up L037 (package the
*fixed* pipeline) and supplies the standard the Y3–Y4 RDL comparison will be held to.

## What shipped
- **Lesson** `lessons/0036-revisit-your-homework-pipeline.html` (~45 min, standard #17 thoroughness):
  cold-open warm-up → the audit posture (source order, not report order) → vocabulary from first principles
  (audit, consequence class, nested split, group straddle, transductive vs inductive, winner's curse,
  selection stability) → **what the pipeline gets right** (7 spine questions pass, 2 above standard practice)
  → four findings, each with its consequence class stated *before* its size → the triage table → the audit
  rubric checklist → teach-back → thesis bridge (the honest-baseline standard + `person_id` as the surviving
  trace of a one-to-many relationship, closing back to L035) → 6 subtleties → 3 quizzes → primary reading →
  lab.
- **One new reusable viz (standard #9):**
  - `assets/nested-calib-viz.js` — the group-straddle mechanism: an outer person-grouped fold whose 13
    training rows get split *again* by `CalibratedClassifierCV(cv=5)`; toggling **ungrouped → grouped** shows
    the same persons landing on both sides of the base-model/calibrator boundary and then not, with a readout
    that quotes the re-measured ECE. Documented expected states in the header comment; CSS prefix `.nc-`.
  - **Generalised** `assets/paired-diff-viz.js` (built for L023) with `metricLabel` / `domain` / `ticks` /
    `verdictNaive` / `verdictCorrected` options so it can carry log-loss and a *non*-significant verdict —
    reuse over duplication (skill rule); L023's own mount is unchanged and its check still passes.
  - Reused `pipeline-viz.js` (transductive encoder) and `checklist.js` (the audit rubric).
  - Headless check `labs/_viz_check_l036.js` — **19/19 pass**; `_viz_check_l023.js` regression clean;
    `_check_pedagogy.js` clean. **Browser MCP unavailable** (headless env) → node verification only,
    consistent with L021–L035.
- **Lab** `labs/0036-revisit-your-homework-pipeline.ipynb` — **Tier A (the learner's own data)**, with a
  seeded Tier-C synthetic stand-in of the same structure auto-generated when `~/Projects/homework` is absent,
  so it still runs on Colab/Binder. Standard #18 scope stated in the intro: this lesson assigns no
  architecture, so the crucial fragments are the **audit instruments** — Task 1 `group_straddle_report()`
  (reusable group-leak detector for *any* splitter; clears the outer CV, catches the library default), Task 2
  a person-grouped inner calibration CV measured before/after, Task 3 the Nadeau–Bengio corrected resampled
  t-test plus a leave-one-fold-out winner check on the submission's own saved per-fold metrics. Stretch: refit
  the encoder in-fold and measure the transductive optimism. Cheaper LightGBM than the submission (120 trees,
  `n_jobs=4`) so both arms finish in ~a minute; both arms use it, so the comparison is internally valid.
- **Audit harness** (kept, so every number is reproducible): `labs/_audit_l036.py` (four regimes A/B/C/D over
  the real pipeline), `labs/_selection_l036.py` (paired tests + Holm + leave-one-fold-out on the saved
  artifacts), `labs/_bench_l036.py` (the `n_jobs` finding).

## Verified live (real homework data, 5,587 labelled rows, person-grouped 5-fold, LightGBM 400 trees + isotonic)
Baseline reproduction is **bit-exact** against the submission's reported M1 arm: log-loss
**1.4248 ± 0.0392**, ECE **0.0363**.

| Regime (M1 arm) | log-loss | ECE top |
| --- | --- | --- |
| A — as shipped (ungrouped inner calibration, encoder on all rows) | 1.4248 ± 0.0392 | 0.0363 ± 0.0053 |
| B — person-grouped inner calibration only | 1.4232 ± 0.0386 | 0.0360 ± 0.0094 |
| C — encoder refit inside each outer fold only | 1.4237 ± 0.0382 | 0.0350 ± 0.0059 |
| D — both fixes | 1.4352 ± 0.0499 | 0.0411 ± 0.0061 |

**The four arms span 1.4232–1.4352 — a range smaller than one fold's standard deviation.** Arm D is *worse*
than as-shipped, driven almost entirely by fold 1 (1.5237 vs 1.4946): isotonic calibration on ~900 rows per
inner fold is itself a variance source, and log-loss charges an unbounded price when it maps a test row to a
near-zero probability. Kept in the lesson as the honest headline rather than smoothed away — the two fixes are
justified by reproducibility and defensibility, not by the metric, and resolving differences this small would
need repeated (5×5) CV per L023.

- **Finding 1 (artifact, not number).** `CalibratedClassifierCV(estimator, method="isotonic", cv=5)` takes an
  *ungrouped* `StratifiedKFold` inside the correctly person-grouped outer fold, so a person's situations
  straddle the base-model/calibrator boundary (**675 of 4,851 labelled persons repeat**; 1,411 rows). Because
  the mis-split is confined to the training block, the reported metrics were **honest**; the fix moved
  log-loss −0.0016 and ECE −0.0003, both far inside the 0.039 fold σ. Real defect, negligible measured cost,
  one-line fix.
- **Finding 2 (decision, not measurement).** M2a shipped over M1 on **Δ = −0.0032 nats = 8 % of one fold's
  σ**, losing 2 of 5 folds, with its entire margin from fold 2; naive paired p = 0.64, corrected p = 0.75; and
  the winner **flips to M1 when fold 2 is dropped**. Separately, the report's only "significant" claim
  (M1−M0, naive **p = 0.0146**) does **not** survive Nadeau–Bengio (**0.0514**) or Holm over its four tests
  (**0.0583**) — while the effect itself (5/5 folds, 0.055 nats, 17× the M2a margin) plainly survives.
  Also caught by reading the code against the prose: the report credits a tie-break rule that `pick_winner`
  never fired, since M2a was already the log-loss argmin.
- **Finding 3 (the only class that *can* inflate the number — and here it does not).** Encoder, KMeans and the
  propensity head are all fit on all **119,498** rows before any CV. Refitting the encoder inside each fold
  (training-fold persons only) gave **1.4237 vs 1.4248 — the honest arrangement is 0.0011 nats *better***, ~3 %
  of fold σ, drifting opposite to the leakage story: a measured effect of zero. Still worth fixing for
  reproducibility, and the *defence* ("unsupervised ⇒ safe") is what must be retired, since the same sentence
  is reused to license KMeans centroids placed with test rows (in the **shipped** M2a) and an ungrouped inner
  OOF split in the propensity model.
- **Finding 4 (declarable only).** No event timestamp exists anywhere in the schema — only
  `responses.observed_at`, recorded 4–8 weeks *after* the event and only for labelled rows — so the L021
  temporal protocol is impossible even though the deployment memo specifies nightly scoring and **weekly
  retraining**. Not a bug and not measurable: a limitation owed to the reader beside the ship-gate table, plus
  a request to the data team.
- **Reproducibility finding (not leakage).** `src.modeling._lgbm()` sets `n_jobs=-1`: **210 s per multiclass
  fit vs 9.2 s at `n_jobs=6`** on this 12-core WSL2 box — a 23× thread-contention penalty on 5.6k rows.
  Nothing about the model changes; it is the difference between re-measuring a finding and skipping it.

### Stage 3 — the KMeans block, priced separately (approximate reconstruction)
| M2a arm (KMeans regime) | log-loss | ECE top |
| --- | --- | --- |
| centroids fit once on all 119,498 rows (as shipped) | 1.4271 ± 0.0368 | 0.0357 ± 0.0070 |
| centroids refit inside each outer fold | 1.4221 ± 0.0259 | 0.0380 ± 0.0046 |

Same verdict as Finding 3: honest is **0.0050 better**, inside fold noise. **Caveat carried into the lesson
text:** this arm is a *reconstruction* of the SSL block (cluster one-hots + centroid distances) and does **not**
reproduce the submission's published M2a row (1.4215 vs 1.4271 here), so the two rows are comparable to each
other and to nothing else. Finding 2's verdict deliberately does not lean on them — it is computed from the
submission's own saved per-fold CSVs, which are exact. (The M1 arm, by contrast, *did* reproduce bit-exactly,
which is why the Finding 1 table is quoted as authoritative.)

## Lab verified live (executed solution, real homework data, reduced model)
All three CHECK cells pass, EXIT ticket and stretch print correctly. The lab deliberately runs a cheaper model
(120 trees, a 40-column feature sample → 73 encoded features) so both arms finish in ~a minute, which the lab
now states explicitly so the learner is not confused by the different level:
- **Structure detected:** 119,498 rows, 5,587 labelled (4.675 %), 4,851 persons, **675 with >1 labelled row =
  1,411 rows = 25.3 %** of the labelled set — the exposure figure quoted in the lesson, measured by the lab.
- **Task 1:** outer CV straddling persons `[0,0,0,0,0]` (clean); the inner `cv=5` split straddles
  `[177,189,178,175,185]` → **442 distinct persons / 930 rows in outer fold 0 alone**.
- **Task 2:** before 1.5654 ± 0.0053 / ECE 0.0160; after (grouped inner) 1.5676 ± 0.0044 / ECE 0.0218; fixed
  inner split straddles **0**. Note the delta's **sign is opposite** to the full-config audit (+0.0023 vs
  −0.0016) and is inside the noise band either way — a better demonstration of "measure, don't assume" than a
  tidy result would have been; the lab text now says to expect exactly this.
- **Task 3:** correction factor `0.200 + 0.250 = 0.450` vs naive `0.200`; M1−M0 Δ = −0.0554, p 0.0146 → 0.0514,
  5/5 folds; M2a−M1 Δ = −0.0032, p 0.6409 → 0.7539, 3/5 folds; leave-one-fold-out winners
  `['M2a','M1','M2a','M2a','M2a']` → **UNSTABLE**.
- **Stretch:** in-fold encoder refit changes log-loss by **−0.0001 nats** — the same "effectively zero" as the
  full-config audit's −0.0011.

## Honest framings kept
- **"Leak" ≠ "inflated metric" (M38).** Name the contaminated quantity before estimating damage; a leak inside
  the training block cannot inflate a metric computed on the outer test fold. Finding 1 is a genuine group
  leak that inflates nothing.
- **Consequence class ≠ measured size (Finding 3).** The class was correctly assigned and the measured
  inflation was still nil. The class tells you where to look and what kind of damage is possible; only the
  re-measurement tells you the size, and "indistinguishable from zero" is a legitimate finding you can only
  earn by running it. Kept in the lesson deliberately uncomfortable rather than tidied away.
- **The most expensive defect broke no measurement (M39).** Selection on noise put the wrong artifact in
  production with every number correct — leakage checklists cannot catch it; L023's variance discipline and
  L017's nested CV can.
- **An audit must be able to return "correct."** Seven spine questions pass, two above standard practice
  (the explicit allow-list encoder; refusing to drop features by mutual information). Crediting the passes is
  what makes the failures credible.
- **Severity is a fact about this dataset, not a law.** Finding 1's cost is small because the label rate is
  4.68 % and only 675 persons repeat; as labels arrive the exposure grows.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l036-consequence` [misconception], `l036-selection`).
- `misconceptions.md` **M38** (finding a leak ⇒ the number was inflated), **M39** (lowest mean CV score = best
  model), **M40** ("unsupervised, so fitting on all rows is safe").
- `reference/glossary.html` — Q4 section +5 terms (pipeline audit, consequence class, group straddle /
  nested-split leak, transductive preprocessing, winner's curse).
- `thesis-dossier.md` +1 ledger line — L036 contributes to the **credibility precondition** under C1–C4, not
  to C1–C2 themselves: the consequence-class triage is exactly what the Y3–Y4 RDL-vs-GBDT comparison will be
  held to.
- `RESOURCES.md` — new "Your own work (Tier-A artifact)" entry for `~/Projects/homework` under Year 1.
- `NOTES.md` **#19** — the homework artifact's location, shape and the four findings, so L037/L040 can build
  on the fixed pipeline; plus the `n_jobs` harness note.
- `assets/paper-deck.js` — **no new card** (standard #16 applies to new ★ papers; Nadeau & Bengio is already
  covered by the Demšar card from L023).
- `lessons/manifest.json` → **36 entries** (L036 Q4, published); `labs/html/0036-*.html` rendered.

## Next
Lesson 037 — **Package the pipeline**: turn the audited submission into a reusable, installable artifact
(config, seeds, one-command reproduction, the fixed splits from this lesson), so Q4's remaining lessons and the
Y1 exit essay stand on code that survives its own audit. The three fixes measured here (grouped inner
calibration, in-fold preprocessing, ship M1 not M2a) are its starting diff.
