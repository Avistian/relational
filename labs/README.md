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

## Running

```bash
pip install jupyterlab scikit-learn pandas numpy   # once
jupyter lab            # then open labs/NNNN-<slug>.ipynb
```

No Jupyter? Open the `.ipynb` directly in Cursor/VS Code and run cells with the Python
extension. Paste the EXIT TICKET output back into chat, or say "lab done".

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
