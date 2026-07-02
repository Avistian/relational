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

## Introductory content — required (from L011 onward)

Labs are not a silent worksheet. The lesson HTML teaches; the notebook **bridges** lesson → hands-on practice. Every new lab must include markdown that a student can follow **without reopening the lesson** for the core idea.

### Required sections (markdown cells)

1. **Header** — lesson link, skill, exit criteria, how PROVIDED/TODO/CHECK/EXIT works, environment (see template).
2. **Concept recap** (after header, before setup) — 3–6 short paragraphs:
   - Restate the *one skill* in plain language
   - Define key terms and formulas used in the lab (with LaTeX or code-style math)
   - One **worked micro-example** on toy numbers (not the lab's answer — e.g. Gini on `[0,0,1,1]`, not the German-credit split)
   - Link back to the lesson HTML for the full viz/reading
3. **Before each task** — a markdown cell with:
   - **Goal** — one sentence: what you will produce
   - **Why it matters** — tie to mission/thesis or the lesson's failure mode
   - **Hint boundary** — describe *what* to compute, never the completed code

### Balance with mid-zone difficulty

- Intro markdown **explains concepts**; TODO cells still hold the **implementation** blanks.
- Do not paste the solution into markdown or prefilled TODO code.
- A formula in the recap is fine; the student still writes the Python that implements it.
- Target: ~30% of notebook cells are explanatory markdown; the rest are PROVIDED/TODO/CHECK/EXIT.

### Retrofit rule

When publishing a new lesson, if the matching lab lacks a concept recap, add one before marking the unit published. Retrofit the current lesson's lab when the user flags thin intros (see `NOTES.md` Preferences).
