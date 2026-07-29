# 0092 — Lesson 040 published: Year 1 Exit Exam

**Date:** 2026-07-29
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 040 — **Year 1 exit exam** (checkpoint). Closes the 6-year
curriculum's first year. Curriculum lab: *Beat XGB on flat task OR explain why not*. Exit criterion:
reproducible tuned tree baseline + written understanding of Grinsztajn's three biases.
**Primary reading:** Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 — exit re-read of abstract + §5
([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)), with the L039 synthesis essay in hand.

## Single skill
Under a fair fixed protocol on a flat Tier-A task, regenerate a disclosed XGBoost baseline, attempt one
honest Year-1 challenger, classify BEAT / TIE / FAIL against a disclosed noise band, and — whether you
beat or not — write Grinsztajn's three inductive biases with evidence of record, ending in a regenerable
number and an explicit STAND or REVISE stance on the L039 synthesis claim.

## Why this was the ZPD
L039 wrote the claim. The exit is the experiment that stands on or revises it. The non-obvious skill is
not "tune harder until you win" — Year 1 already showed that fails — but treating **TIE / EXPLAIN as full
passes**, classifying gaps against a noise band (antidote to the L036 winner's curse), and producing both
curriculum deliverables (regenerable bar + written biases) without inventing an RDL fair-bar win.

## What shipped
- **Lesson** `lessons/0040-year-1-exit-exam.html` (~60 min): warm-up → Year-1 assembly table → six exit
  gates → three-way fork → evidence-of-record table → three biases restated → homework bridge → teach-back
  → thesis bridge → 3 quizzes → primary reading → lab → exit checklist.
- **Two new reusable viz** (`file://`-safe):
  - `assets/exit-verdict-viz.js` — BEAT / TIE / EXPLAIN × 3 points; default = tie-adult. CSS `.ev-`.
  - `assets/exit-gates-viz.js` — six protocol gates; default = regenerable baseline. CSS `.eg-`.
  - Reused `biases-viz.js`, `checklist.js`, pedagogy widgets.
  - Headless `labs/_viz_check_l040.js` — **all pass**; no new bake-off (cites L020/L030/L025–L035 numbers).
  - Browser MCP unavailable → node verification only (consistent with L021–L039).
- **Lab** `labs/0040-year-1-exit-exam.ipynb` (+ solutions/, builder `_build_l040.py`): Tier A OpenML
  `adult` under L020 protocol; Tasks = regenerate XGB ref → disclosed LGBM challenger →
  `classify_verdict(..., noise=0.002)` → three bias sentences + STAND/REVISE → EXIT ticket. Homework
  `make verify` is stretch (private artifact; cloud env lacks `~/Projects/homework`).

## Design choices
- **Adult, not homework, for the runnable bar.** NOTES #19 prefers homework for Q4 Tier A; the curriculum
  exit needs a *public regenerable* XGB number. Homework discipline (noise bands, named estimators,
  baseline parity) is reused in the fork classifier and stretch; L036–L038 already closed the private
  audit/package/review arc.
- **TIE is a pass.** Matches L030's "no significant winner" skill and L020's verified near-ceiling.
- **No new verified bake-off.** Lab regenerates the L020 reference band; lesson quotes prior figures only.

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l040-softbeat` [misconception], `l040-deliverables` [misconception],
  `l040-stance`).
- `misconceptions.md` **M48** (any positive delta = beat) and **M49** (must beat XGB to pass).
- `reference/glossary.html` — Q4 +4 terms (Year 1 exit exam, exit fork, noise band, essay stance).
- `assets/paper-deck.js` +1 card `grinsztajn2022-exit`.
- `thesis-dossier.md` — Evidence Ledger +L040 (FOR+BAR, C3/C4/C1); current verdict after L040.
- `RESOURCES.md` — Grinsztajn entry extended as L040 primary reading.
- `lessons/manifest.json` → **40 entries** (L040 Q4, `checkpoint: true`, lab path set).
- L039 nav now links forward to L040.

## Next
User runs the exit lab and pastes the EXIT ticket (or says "lab done"). On completion, mark Year 1 closed
and open Year 2 with Lesson 041 (neural tabular architectures — NODE / TabNet era).
