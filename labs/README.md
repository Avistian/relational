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

**Introductory content (from L011 onward):** each lab opens with a **concept recap**
(terms, formulas, one toy worked example) and a short **goal / why** block before every task.
The notebook should bridge lesson → practice without forcing the student to re-read the HTML
for core definitions. See `.agents/skills/lab-authoring/SKILL.md` § Introductory content.

**No prefilled answers:** TODO code cells use `____` only — never the completed solution.
Teacher copies with filled answers live in [`solutions/`](./solutions/) (gitignored).

**Agent scoring:** when you say *lab done*, your teacher scores the lab with the rubric in
`.agents/skills/lab-authoring/SKILL.md`.

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
- `0008-metrics-calibration.ipynb` — ROC-AUC vs PR-AUC against the prevalence baseline, and
  reliability-curve + Brier calibration with CalibratedClassifierCV (Lesson 008).
- `0009-feature-engineering.ipynb` — ratio, cyclical datetime, leak-free target encoding (Lesson 009).
- `0010-baseline-checkpoint.ipynb` — Q1 capstone reproducible baseline (Lesson 010).
- `0011-decision-trees-partitions.ipynb` — tree splits on OpenML credit data (Lesson 011).
- `0012-bagging-random-forest.ipynb` — RF vs single tree, OOB, variance drop (Lesson 012).

## View / run in the browser

- **View:** static HTML renders live in `labs/html/` (regenerate with `bash scripts/render_notebooks.sh`).
- **Run:** the [Notebooks page](../notebooks.html) links each lab to **Binder** (real `relkit` env + OpenML fetch) and **Colab**.
