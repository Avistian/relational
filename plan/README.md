# Lesson Decomposition Plan

Agent-facing, per-lesson breakdown of the [expert curriculum](../CURRICULUM.md). Where
`CURRICULUM.md` gives a one-line topic row and a quarter-level paper list, this plan expands
**every lesson from 031 through 240** into a concrete lesson spec — enough that any future
session can open the file and build the lesson HTML + lab without re-deriving scope.

Lessons 001–019 are already **published** (`lessons/`), and 020–030 have adequate one-line rows
plus published-lesson precedent to follow; the decomposition therefore starts at **Y1 Q4 (031)**,
where the granular breakdown in `CURRICULUM.md` thins out and the student-facing
`reference/curriculum.html` collapses whole quarters into paper lists.

## Why this exists

The published lessons (001–019) all share the same anatomy — a *single skill*, a *specific paper
section*, a *reusable viz asset*, a *lab with one "crucial fragment"*, an *honest myth-buster*, and
a *thesis bridge*. `CURRICULUM.md` rows do not carry that detail for later lessons, so "create the
next lesson" still required re-reading the papers each time. This plan front-loads that work:
papers are decomposed into the lessons that will teach them, grounded in the actual paper
contributions (verified via the arXiv MCP — see IDs inline).

## Per-lesson schema

Each lesson is one entry:

> **NNN · Title** — *paper(s), tier, arXiv id §section*
> - **Skill** — the one testable competency (mirrors the "single skill" line in published records).
> - **Teach** — 2–4 concept beats the lesson HTML covers (knowledge before skill).
> - **Lab** — dataset [Tier A/B/C] · the *crucial fragment* the student implements → deliverable.
> - **Viz** — reuse an existing `assets/*-viz.js` or a one-line spec for a new reusable component.
> - **Bridge** — retrieval callback to prior lessons + thesis/mission tie + what it sets up.

**Checkpoints** (every 10th lesson) and **exit exams** (every 40th) replace *Lab/Viz* with a
**Deliverable** (the capstone artifact) and are scored against the exit criterion, not a new concept.

Tiers, cell conventions (PROVIDED/TODO/CHECK/EXIT), the 4-block reproduction structure, and the
0–10 lab rubric all follow [`.agents/skills/lab-authoring/SKILL.md`](../.agents/skills/lab-authoring/SKILL.md).
Viz decisions follow [`.agents/skills/lesson-visuals/SKILL.md`](../.agents/skills/lesson-visuals/SKILL.md).

## How to use this when building a lesson

1. Read the lesson entry here + its paper row in the [verified index](../CURRICULUM.md#verified-paper-index-arxiv-ids).
2. Run `curriculum-currency` if it's the first lesson of a new quarter (frontier papers move monthly).
3. Deep-read the cited paper **section** (not the whole paper) via the arXiv MCP.
4. Build the lesson HTML (single skill), reuse/author the viz, build the lab (one crucial fragment).
5. Verify live (numbers in the record), publish, update `lessons/manifest.json`, log a learning record.

The **Skill / Teach / Lab / Viz / Bridge** fields are a *starting contract*, not a straitjacket —
update the entry if the paper's live reproduction reveals a better framing (as with any planned scope).

## Index

| Year | Quarter arc | File |
|------|-------------|------|
| 1 | Q4 · Consolidation & bridge to neural tabular (031–040) | [year-1.md](./year-1.md) |
| 2 | Advanced tabular DL — architectures → honest baseline → TFMs → SSL/bridge (041–080) | [year-2.md](./year-2.md) |
| 3 | Graph ML — message passing → hetero → temporal → mastery (081–120) | [year-3.md](./year-3.md) |
| 4 | Relational deep learning — REG → baselines → next-gen → expertise (121–160) | [year-4.md](./year-4.md) |
| 5 | Foundation relational models — concepts → pretraining → frontier → synthesis (161–200) | [year-5.md](./year-5.md) |
| 6 | Novel research — hypothesis → execution → communication → launch (201–240) | [year-6.md](./year-6.md) |

**Design principles carried through every year** (from `CURRICULUM.md` critical framing):
- Every architecture is taught **with its failure mode**, never as hype.
- **GBDTs are not dead** — the relational thesis must beat *tuned* baselines under *temporal* splits.
- Front-loaded learning, back-loaded research: Y1–Y2 concept-dense, Y3–Y4 reproduction-dominated,
  Y5–Y6 research-dominated (a numbered unit is a *unit*, not a day).
