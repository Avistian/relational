---
name: lab-authoring
description: Author mid-difficulty lab notebooks with PROVIDED/TODO/CHECK/EXIT cells, paper-reproduction structure, dataset tiers, and agent scoring when the user completes a lab.
---

Use when creating, retrofitting, or reviewing lab notebooks in `labs/`.

## Cell convention

| Tag | Student sees | Agent writes |
|-----|--------------|--------------|
| **PROVIDED** | Complete code — run only | Imports, data load, helpers, leaky baseline for contrast |
| **TODO** | Blanks (`____`, `# TODO`) | 2–4 focused blanks per task on the *skill* being practised |
| **CHECK** | Auto assertions — do not edit | Immediate feedback |
| **EXIT TICKET** | Final deliverable print | Verifiable summary the user pastes to teacher |

**Never:** complete solution in TODO code cells or markdown hints. Hints describe *what* to achieve, not *how* to code it.

**Teacher solutions:** store in `labs/solutions/NNNN-<slug>.ipynb` (gitignored). Student notebooks stay blank.

## Difficulty — mid zone

- Not easy: student must write non-trivial lines (not copy-paste from markdown).
- Not hard: peripheral boilerplate is PROVIDED; one skill per task.
- Optional **stretch** cell at end (ungraded) for fast finishers.

## Dataset tiers

| Tier | When | Examples |
|------|------|----------|
| **A — Real, small, open** | Default Q2+ training/eval labs | OpenML (`fetch_openml`), UCI; see `labs/data/README.md` |
| **B — Real, relational** | Y3+ / RelBench | RelBench tasks via PyTorch Frame |
| **C — Synthetic** | Mechanism isolation only | MCAR/MAR/MNAR demos, pure-noise leak demos |

Document tier and rationale in the lab intro markdown.

## Paper-reproduction labs (Q2 onward)

Four blocks per reproduction lab:

1. **Paper step** — numbered algorithm step from the paper (with section ref)
2. **Crucial fragment** — student implements one non-obvious function (split gain, ICL batch layout, etc.)
3. **Harness** — `import relkit` from `labs/relkit/` (CV, metrics, leakage-safe pipelines)
4. **Reproduction target** — metric within tolerance of paper, or documented honest fail

## Agent review when user says *lab done*

Score 0–2 per axis (max 10):

| Axis | 0 | 1 | 2 |
|------|---|---|---|
| **Correctness** | EXIT/CHECK fail | Partial pass | All CHECK + EXIT clean |
| **Leakage discipline** | Obvious leak | Minor doubt | Pipeline/fold boundaries correct |
| **Conceptual takeaway** | Missing/wrong | Vague | One accurate sentence |
| **Mid-zone effort** | Copy-paste or blank | Too easy or stuck | Appropriate challenge |
| **Reproduction (if applicable)** | Not attempted | Wrong metric | Within tolerance or honest fail documented |

Return: total /10, one concrete improvement, log notable gaps in `NOTES.md` or a learning record.

## Template

Follow `labs/LAB-TEMPLATE.ipynb` and `labs/README.md`.
