# 0090 — Lesson 039 published: Year 1 Synthesis Essay

**Date:** 2026-07-29
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q4 · lecture 039 — ninth lesson of Q4 (Consolidation & bridge to neural tabular,
L031–L040) and the penultimate Year 1 unit before the exit exam. The curriculum titles it *Year 1 synthesis
essay* with lab *Write: what trees beat and why*.
**Primary reading:** Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning
on tabular data?*, NeurIPS 2022 — [arXiv:2207.08815](https://arxiv.org/abs/2207.08815). Re-read **full** with
the essay skeleton in hand (previously met in pieces: L019 preview, L024 benchmark, L025–L027 mechanisms).
Companions while drafting: Domingos 2012 (FE ceiling, L033); Fey et al. 2024 §1–2 (what joins destroy, L035);
Lones 2024 (peer-review coda, L038).

## Single skill
Compose a **synthesis essay** that argues, with Year-1 evidence, what a competently-built tree ensemble beats
on flat tabular tasks and **why** — naming the three inductive biases, the exhaustion of single-table
cleverness, the conditions where the claim flips, and the relational frontier the flatten cannot speak to —
ending in a claim a hostile reviewer could grade.

## Why this was the ZPD
L036–L038 turned the evaluation apparatus inward (audit → package → peer review) on the learner's own
pipeline. The non-obvious next move before the exit exam is not another measurement — it is to **write the
argument** the exit will defend or revise. Year 1 has every input: the leakage spine (Q1), the incumbent
(Q2), Grinsztajn's three biases + honest bar (Q3), the exhaustion cascade + join collision + credibility
checklist (Q4). What the learner has not yet done is assemble them into a *falsifiable claim with
boundaries and an open burden* — the genre skill that separates a synthesis from a chronological recap
(M46) and "trees are more powerful" folklore from inductive-bias match (M47).

## What shipped
- **Lesson** `lessons/0039-year-1-synthesis-essay.html` (~55 min): warm-up → synthesis vocabulary (claim,
  boundary condition, open burden, evidence of record) → working claim → Year 1 as four moves → opponent
  ladder (win / tie / within-noise) → three biases with reused `biases-viz` → frontier map → credibility
  coda → essay skeleton → teach-back → thesis bridge → 3 quizzes → primary reading → written lab.
- **Two new reusable viz** (`file://`-safe):
  - `assets/y1-arc-viz.js` — 4 quarters × 3 milestone chips; each chip hands an *essay sentence* plus
    evidence of record. Default = Q3 / three biases. CSS prefix `.ya-`.
  - `assets/trees-frontier-viz.js` — three zones (WIN / EXHAUSTED / FRONTIER) × 3 points; keeps the claim's
    regimes apart. Default = win-smooth. CSS prefix `.tf-`.
  - Reused `biases-viz.js` (L019), `checklist.js`, `quiz.js`, `predict.js`, `teachback.js`, `retrieval-bank.js`.
  - Headless check `labs/_viz_check_l039.js` — **all pass**; every quoted number is a prior verified figure
    (no new bake-off). **Browser MCP unavailable** → node verification only, consistent with L021–L038.
- **No lab notebook.** The lab is the essay (800–1,200 words target), pasted into chat for a hostile-reader
  review — `labPath: null` in the manifest (same pattern as L038).

## Working claim the essay exists to make
On typical flat tabular tasks — medium-sized tables whose columns carry individual meaning, whose targets
are irregular, and whose feature sets contain junk — a leak-free, budget-tuned GBDT matches or beats honest
neural baselines and AutoML because its inductive biases fit that regime. Further single-table cleverness
(deeper nets, embeddings, attention, hand FE) has repeatedly tied or failed. The claim flips under smooth
targets, rotated bases, and low-junk regimes — and is silent on signal a lossy join already destroyed.
Whether a model that *keeps* relational structure can beat this fair flat bar is the open burden of Years 3–6.

## Honest framings kept
- **Synthesis ≠ recap (M46).** Organised by argument moves, not lesson numbers.
- **Inductive-bias match ≠ "more powerful" (M47).** Flip conditions are mandatory.
- **Win / tie / within-noise** distinguished on the opponent ladder (L010 crush of dummy; L030 corrected
  p=0.64 tie; L033 FE inside ±0.03).
- **Open burden stays open.** Year 1 demonstrated the *cost* of flattening, not a fair-bar RDL win.
- **Credibility coda** binds every comparative sentence to the L038 checklist (two pipelines, one standard).

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l039-recap` [misconception], `l039-biases` [misconception], `l039-frontier`).
- `misconceptions.md` **M46** (synthesis = chronological recap) and **M47** (trees win because more powerful).
- `reference/glossary.html` — Q4 section +7 terms (synthesis essay, claim, boundary condition, open burden,
  evidence of record, exhaustion cascade, opponent ladder).
- `assets/paper-deck.js` +1 card `grinsztajn2022-synthesis`.
- `thesis-dossier.md` — Evidence Ledger +L039 (FOR+BAR, C1/C3/C4); current verdict updated to after L039.
- `RESOURCES.md` — Grinsztajn entry extended as L039 primary reading.
- `lessons/manifest.json` → **39 entries** (L039 Q4, published, `labPath: null`).
- L038 nav now links forward to L039.

## Next
Lesson 040 — **Year 1 exit exam**: beat XGBoost on a flat task *or explain why not*, standing on (or revising)
the synthesis essay written here. Exit criterion per `CURRICULUM.md`: reproducible tuned tree baseline +
written understanding of Grinsztajn's three biases.
