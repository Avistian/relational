# 0072 — Lesson 030 published: Q3 Checkpoint — Write a Benchmark Report

**Date:** 2026-07-18
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 030 — the **Q3 checkpoint**. Matches the CURRICULUM row (030:
"Q3 checkpoint · Grinsztajn 2022 full · Write 1-page benchmark report"). Consolidates the whole Q3 arc
(L021–L029); no new concept — a capstone, like L010 (Q1) and L020 (Q2).
**Primary reading:** Grinsztajn, Oyallon & Varoquaux 2022 (`2207.08815`), read **in full** (met in pieces
across L019/L024/L025/L026/L027).

## Single skill
Assemble the whole Q3 evaluation-rigor toolkit into one defensible **one-page benchmark report** — a
curated dataset under a deployment-matched split, a leakage audit, the right contestants (a tuned GBDT, an
honest neural baseline, and the AutoML bar) compared as a random-search **budget curve** rather than a
single tuned number, a **significance test** that proves any gap is not noise, and an inductive-bias
explanation of the ranking — ending in a verdict you would defend to a hostile reviewer, **including the
verdict "no significant winner."**

## Why this was the ZPD
Q2 taught the learner to *build* the incumbent (L020 checkpoint); Q3 taught them to *judge* it
(L021 splits, L022 leakage, L023 significance, L024 protocol, L025–L027 biases, L028 neural baseline,
L029 AutoML). The natural capstone is to assemble all of it into the exact artifact every future thesis
claim must take: a fair, temporal, leak-audited, significance-tested, bias-explained benchmark report.
It also closes Y1 Q3 and sets up the Q4 consolidation/bridge (L031+).

## What shipped
- **Lesson** `lessons/0030-q3-checkpoint.html` (~50 min, capstone; standard #17 thoroughness): intro →
  single skill → "What Q3 gave you, assembled" table (L021–L029 → report section) → the
  **benchmark-report contract** (Q2's five + Q3's three: deployment-matched split, audited provenance,
  proven-then-explained gap) → the reference report on credit_g (Section A budget curve, Section B
  head-to-head + significance, Section C bias explanation) → predict-before-reveal → "a tie is a valid
  verdict" → self-audit checklist (10-item benchmark-report rubric via `checklist.js`) → teach-back →
  thesis bridge → 3 quizzes → primary reading (Grinsztajn full) → lab.
- **Pedagogy widgets (standards #10–#14):** warm-up (`RetrievalBank upTo:30`), predict-before-reveal (on
  "what should the report conclude from +0.008?"), teach-back (the load-bearing "sections of a defensible
  report + why a tie is valid"), 3 quizzes, and the reusable `Checklist` rubric.
- **No new viz** — consistent with the L010/L020 checkpoint precedent. A checkpoint consolidates and
  introduces no new mechanism, so the `lesson-visuals` "one viz per *new* mechanism" rule is satisfied by
  the tables + the reused `checklist.js`; the Q3 mechanism viz live in L021–L029 and are linked.
- **Lab** `labs/0030-q3-checkpoint.ipynb` — Tier A (real credit_g via `relkit`), built via
  `_build_l030.py`. Two crucial fragments are the Q3 *evaluation* skills (not model internals):
  - Task 1 = the **L024 budget-curve selection** (`best_va, best_te = va, te` — keep best-by-validation,
    report its test), for the GBDT and the neural baseline;
  - Task 2 = the **L023 corrected resampled t-test** (`corrected_var = (1/n + 1/(k-1)) * var_d`);
  - Task 3 = assemble the one-page report (fill the leakage audit / model info sheet + the honest verdict).
  Student blank (6 `____`, 0 outputs); solution executed clean & gitignored. Neural contestant is sklearn
  `MLPClassifier` (torch-free, runs anywhere) standing in for the L028 ResNet — noted in the lab.
- **Verify** `labs/_verify_l030.py` — ran clean; numbers below.

## Verified live (credit_g, real Tier-A, via _verify_l030.py + executed solution)
- **Budget curve (single 60/20/20 split, K=15, select-by-validation):** GBDT default **0.804** → ceiling
  **0.809**; MLP default **0.819** → ceiling **0.805** (tuning *dipped* — a 15-config search on a ~200-row
  validation slice overfit; the L024 "not too small" criterion biting, disclosed as a finding). Both land
  ~0.80–0.82.
- **Head-to-head (paired 5×5 CV, 25 folds, defaults, identical folds):** GBDT **0.780** vs MLP **0.772**,
  mean paired gap **+0.0081** (sd 0.0321).
- **Significance:** naive paired t p=**0.218** (not sig); **corrected resampled t p=0.643** (not sig, and
  larger, as it must be — the correction can only raise p). **Verdict: no significant winner — a tie
  within noise.** GBDT is the cheapest strong default; the three biases (L025–L027) explain its small lean.
- Lab solution CHECK + EXIT all clean; the executed report matches the lesson numbers exactly.

## Honest framings kept
- **A tie is a valid, valuable verdict.** The +0.008 mean is not a win; reporting "GBDT wins" from it is
  exactly the L023 mistake. The checkpoint's deliverable *is* the honest tie.
- **The budget curve can dip** on small data (best-by-validation ≠ monotone in test) — disclosed, not
  hidden; a single tuned number would have concealed it.
- **MLPClassifier ≠ the L028 ResNet** — a torch-free portable stand-in with the same inductive biases;
  the checkpoint tests the *report*, not the architecture. (torch was not installed this session.)
- **Not contradicting L028's "GBDT leads credit_g."** The stable 5×5 CV still puts the GBDT nominally
  ahead (+0.008); the point is that the gap is not significant — the honest reading of a small dataset.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l030-tie` [misconception], `l030-report`).
- `misconceptions.md` **M27** ("a benchmark must crown a winner; a tie is a failed experiment" → a
  correctly-established tie is valid; prove the gap with a corrected test before claiming a win).
- `thesis-dossier.md` +1 ledger line (L030, BAR + FOR, C3/C4 — the full honest instrument is assembled;
  the tie + L029 show single-table search/architecture returns are nearly exhausted → upside is
  representational) and **Current verdict updated to "after L030 / Q3."**
- `reference/glossary.html` +3 terms (Benchmark report, Model info sheet, Statistical tie / no significant
  winner).
- `lessons/manifest.json` regenerated → 30 entries (L030 `checkpoint: true`); `labs/html/0030-*.html`
  rendered.
- No new paper flashcard — Grinsztajn 2022 already has cards (`grinsztajn2022` + the §-specific
  benchmark/smoothness/rotation/uninformative cards); the checkpoint reading is the same paper in full.
- `node labs/_check_pedagogy.js` clean.

## Env note
`.venv` present with sklearn 1.9.0 + scipy 1.18.0 + xgboost; **torch NOT installed** this session, so the
checkpoint lab/verify use sklearn `MLPClassifier` for the neural baseline (portable). Env-setup agent
should still preinstall the lab venv incl. CPU torch for Year 2+ (recurring since Sessions 22–28), but the
Q3 checkpoint deliberately avoids torch so it runs anywhere.

## Next
Lesson 031 — **Q4 opener** (embeddings for categoricals; entity embeddings, Guo & Berkhahn 2016; target
encoding pitfalls), beginning the Q4 consolidation + bridge to neural tabular. Q3 (L021–L030) is closed
once the user completes the L030 report.
