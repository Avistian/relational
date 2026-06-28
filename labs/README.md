# Labs

Each lesson's exit-ticket lab ships as a Jupyter notebook here, named `NNNN-<slug>.ipynb`
to match its lesson (`lessons/NNNN-<slug>.html`).

## The fill-in convention

Every lab follows [`LAB-TEMPLATE.ipynb`](./LAB-TEMPLATE.ipynb):

| Cell tag | Meaning |
|----------|---------|
| **PROVIDED** | Complete boilerplate (imports, data, helpers). Just run it. |
| **TODO** | Has blanks — `____` or `# TODO`. You fill these in. This is where the learning happens. |
| **CHECK** | Auto-runs assertions / prints so you get *immediate* feedback. Don't edit. |
| **EXIT TICKET** | The final deliverable cell. When it prints cleanly, the lab is done. |

Design intent: blanks sit only on the **skill** being practised; everything peripheral is
provided so working memory stays on the one idea (see `../NOTES.md` preferences and the
`teach` skill's fluency-vs-storage-strength note). CHECK cells make the feedback loop tight.

## Environment (one-time setup)

From the repo root (`relational/`):

```bash
bash labs/setup-env.sh
```

This creates `.venv/`, installs [`requirements-labs.txt`](../requirements-labs.txt), and registers the Jupyter kernel **Relational Labs (.venv)**.

**In Cursor / VS Code:** open a lab notebook → **Select Kernel** → choose either:
- **Relational Labs (.venv)** (Jupyter kernel), or
- the interpreter at `.venv/bin/python`

**In a terminal:**

```bash
source .venv/bin/activate
jupyter lab    # then open labs/NNNN-<slug>.ipynb
```

Re-run `bash labs/setup-env.sh` after pulling new dependencies.

## Running

## Reproduction labs build incrementally (future)

Concept labs (like `0006`) are self-contained. But **paper-reproduction / experiment labs**
(RelBench baselines, GBDT/RealMLP/TabM, RDL) must build on what earlier labs already wrote —
a cumulative, reusable codebase, not isolated one-offs. When that phase starts:

- Promote shared code out of notebooks into a small importable package (e.g. `labs/relkit/`):
  data loaders, the CV/eval harness, leakage-safe pipelines, metrics.
- Reproduction notebooks `import` that package and extend it; each lab leaves the harness
  stronger and better-tested for the next.

This keeps the thesis baselines trustworthy by the time results matter.

## Index

- `0006-missingness.ipynb` — classify MCAR/MAR/MNAR; show complete-case bias and the
  missing-indicator fix (Lesson 006).
- `0007-class-imbalance.ipynb` — accuracy paradox, the SMOTE-before-CV leak vs the imblearn
  Pipeline fix, and leak-free class weights (Lesson 007).
