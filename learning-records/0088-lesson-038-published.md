# 0088 — Lesson 038 published: Peer Review Your Evaluation

**Date:** 2026-07-29
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 038 — eighth lesson of Q4 (Consolidation & bridge to neural tabular,
L031–L040) and the third built on the learner's own submission. The arc closes: [[0084-lesson-036-published.md]]
asked whether the reported number was **right** (audit), [[0087-lesson-037-published.md]] whether it was
**stable** (reproduce), and this asks whether it would **survive a hostile competent reader** — i.e. it hands
the learner the *reviewer's* hat and turns the self-audit stance outward.
**Primary reading:** Michael A. Lones, *Avoiding common machine learning pitfalls*, **Patterns** 5(10):101046,
2024 — [doi:10.1016/j.patter.2024.101046](https://doi.org/10.1016/j.patter.2024.101046) (open access; the
continuously-updated preprint is [arXiv:2108.02497](https://arxiv.org/abs/2108.02497)). Its middle three
sections *are* the lesson's three axes. Companions (checklists the deliverable descends from): Kapoor &
Narayanan, *Leakage and the reproducibility crisis in ML-based science*, Patterns 2023
([doi](https://doi.org/10.1016/j.patter.2023.100804)); REFORMS (Kapoor, Cantrell, … Lones, Malik et al.,
Science Advances 2024, [doi](https://doi.org/10.1126/sciadv.adk3452)).

## Single skill
Run a **structured, severity-graded peer review of an evaluation across three axes — leakage, tuning, metrics —
and reach a defensible verdict that names what must change**, holding the baseline to exactly the standard you
would demand of a result that beat you. The load-bearing sub-skill is **triage**: sorting findings on two
*independent* axes (conclusion-impact vs artifact severity) rather than enumerating them.

## Why this was the ZPD
L036 gave the learner a triaged list of defects in their own code; L037 turned the pipeline into a package that
regenerates its number. Both were *inward* — the learner auditing their own work with the source in hand. The
non-obvious next move, and the one the whole mission runs on, is the *outward* stance: a reviewer usually has
only what was **reported**, extends no benefit of the doubt, and must end on a **verdict**, not a list. This is
a genuinely different skill (reasoning from omissions, grading severity, triaging on two axes) and the learner
had every input for it already measured — so L038 is a **method** lesson that re-reads the verified L036/L037
findings through a reviewer's eyes, introducing **no new numbers**. It also sets up the comparative discipline
Y3–Y4 needs: a claim that RDL beats a GBDT needs *two* pipelines reviewed to *one* standard, and the baseline
you want to beat is the one you are least motivated to scrutinise.

## What shipped
- **Lesson** `lessons/0038-peer-review-your-evaluation.html` (~50 min): cold-open warm-up → why review (the
  mission's immune system; the comparative-claim two-pipeline argument; Kapoor & Narayanan's 17-field / 329-paper
  finding) → a full vocabulary block built from first principles (reviewer's stance, blocker/major/minor/nit,
  the two triage axes, verdict, sequential overfitting, HP budget parity, multiplicity, meaningful vs naive
  baseline, significance vs effect size, model info sheet / REFORMS) → the evaluation under review (the homework
  `report.md`'s three headline claims) → **Axis 1 leakage sweep** (walk the 8-type taxonomy, "leak ≠ inflated"
  callout) → **Axis 2 tuning/selection sweep** (sequential overfitting, the winner's-curse blocker, budget
  fairness / the float32 cast) → **Axis 3 metrics sweep** (estimator of record, noise floor) → **triage grid**
  (the load-bearing viz) → predict-the-verdict gate → the verdict (major revision, with required changes per
  claim) → the reusable peer-review checklist (the deliverable) → teach-back → thesis bridge → subtleties →
  3 quizzes → primary reading → lab (a written review, not a notebook).
- **One new reusable viz** (`file://`-safe plain script): `assets/review-triage-viz.js` — a 2-axis scatter
  (conclusion-impact × artifact severity) placing **7 defects** drawn straight from L036/L037, coloured by
  checklist axis (leakage / tuning / metrics / reporting-repro), with quadrant guides and a click-to-read-out
  panel. CSS prefix `.rt-`. Default selection is the winner's-curse blocker (T1). The whole point of the visual
  is that the two axes *come apart* — the ungrouped split sits high-but-left (severe, doesn't move the number),
  the 0.0032-nat margin sits low-but-right (every number correct, decides the claim).
  - Reused `leakage-taxonomy-viz.js` (L022), `checklist.js`, `quiz.js`, `predict.js`, `teachback.js`,
    `retrieval-bank.js`.
  - Headless check `labs/_viz_check_l038.js` — **52/52 pass**; `_check_pedagogy.js` clean; every `.rt-` /
    `.ltv-` class emitted by the widgets verified present in the lesson's stylesheet, and every readout number
    asserted against its L036/L037 measured value. **Browser MCP unavailable** (headless env) → node
    verification only, consistent with L021–L037.
- **No lab notebook.** The "lab" is a review the learner *writes* against the homework `report.md` (or a
  supplied write-up), pasted into chat for the teacher to review — deliberately no dangling `.ipynb`; `labPath`
  is `null` in the manifest.

## The review, as it stands on the verified evidence (no new numbers)
Every figure below was measured in L036/L037 and is only re-read here through the two triage axes.

| Code | Axis | Finding | Conclusion-impact | Artifact severity |
| --- | --- | --- | --- | --- |
| **T1** | tuning | M2a over M1 on **0.0032 nats** (8 % of fold σ, loses 2/5 folds, naive p=0.64, **N–B corrected p=0.75**, flips to M1 on dropping fold 2) | **HIGH** — *is* claim 1 | MID — numbers all correct |
| **M1** | metrics | "the ECE" is **0.0332** (per-fold) or **0.018** (pooled) — 1.87× on an unnamed estimator; selected on one, ship-gated on the other | **HIGH** — *is* claim 2 | MID |
| **M2** | metrics | failed ship-gate (107-row `age=missing`, ECE 0.094 vs 0.05) sits **below** its **0.1071 ± 0.0297** noise floor | HIGH | MID |
| **T2** | tuning | undeclared `.astype(np.float32)` = **42 %** of the 0.0032 margin (+0.00133 mean, 258/5,587 argmax flips) | AMBIGUOUS — silently corrupts T1 | MID–HIGH |
| **R2** | repro | one fold draw reported as the result; splitter seed 0→4 spans **0.0166 nats** (5× the margin) | HIGH | MID |
| **L1** | leakage | ungrouped inner calibration split (outer CV *is* grouped); grouped re-measure 1.4232/0.0360 vs 1.4248/0.0363, inside 0.039 σ, *right* direction | **LOW** — leak confined to training block | **HIGH** — shipped isotonic map mis-shaped for strangers |
| **R1** | repro | no lockfile, `>=` constraints only; lightgbm 4.5.0 + sklearn 1.9.0 both satisfy them and **crash** | LOW on the number | HIGH for reproducibility |

**Verdict: major revision.** Not reject (engineering better than most published ML; every defect fixable
without redoing the study), not accept (all three headline claims fail as written). Claim 1 blocked by T1 (+42 %
from T2); claim 2 blocked by M1/M2; claim 3 ("reproducible") true-but-inert (the seed) and unbacked (R1) — a
major.

## Honest framings kept
- **A review triages, it does not enumerate (M44).** Two independent axes; grading only one either nukes a sound
  paper over a nit or waves through a fatal flaw dressed as a small number. Seven ungraded findings is a worse
  review than three correctly graded.
- **"Leak" does not imply "inflated number."** The load-bearing counter-intuition, carried straight from L036:
  the ungrouped calibration split is high artifact severity but low conclusion-impact, and the honest
  re-measure moves the number the *right* way. Name the contaminated quantity before estimating damage.
- **Omissions are findings.** An unnamed estimator, an undescribed split, a missing lockfile are defects — an
  unstated choice cannot be trusted on faith.
- **Hold the baseline to the model's standard (M45).** The comparative claim needs two pipelines reviewed to one
  standard; the honest researcher's failure mode is a generous review of the model they love and a lazy one of
  the baseline they want to beat. This is the discipline that makes an eventual RDL win credible.
- **Significance *and* effect size, both.** The winner's-curse blocker needed the corrected p=0.75 *and* the
  "flips on dropping one fold" magnitude to be damning.
- **Match the verdict to what revision can fix.** Reject only when no revision rescues the claim; here the
  machinery is sound, so the honest verdict is major revision.

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l038-triage` [misconception], `l038-baseline-parity`, `l038-omission`);
  pool now 59 items, ids unique, all validated.
- `misconceptions.md` **M44** (a good review is a thorough list of every flaw) and **M45** (scrutinise the novel
  model, it is the contribution) — both under a new "Q4 · peer review" heading.
- `reference/glossary.html` — Q4 section +11 terms (reviewer's stance, severity grades, conclusion-impact vs
  artifact severity, verdict, sequential overfitting, HP budget parity, multiplicity effect, meaningful vs naive
  baseline, significance vs effect size, REFORMS).
- `assets/paper-deck.js` +1 card `lones2024` (standard #16 — a new ★ paper).
- `thesis-dossier.md` — current-verdict section extended: L038 converts the self-audit stance into the reviewer's
  stance and names the two-pipelines-one-standard discipline the Y1 exit and every Y3–Y4 comparison run on.
- `RESOURCES.md` — Lones 2024 added under Year 1 with the Kapoor & Narayanan and REFORMS companions.
- `lessons/manifest.json` → **38 entries** (L038 Q4, published, `labPath: null`).

## Next
Lesson 039 per `CURRICULUM.md`. The peer-review checklist shipped here is the reusable instrument the Y1 exit
essay (L040 — "beat XGBoost on a flat task or explain why not") and every Y3–Y4 RelBench comparison will be
graded against; the triage grid is how the learner decides whether a leaderboard delta is believable or
enthusiasm.
