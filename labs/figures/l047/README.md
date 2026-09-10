# SAINT notebook figures

These PNGs are embedded as inline `data:image/png;base64,...` markdown images in
both Lesson 047 notebooks, matching Colab's image representation. Jupyter's
`attachment:` links do not render in Colab and must not be used here.
They remain visible without executing cells or copying this directory alongside
the downloaded notebook.

Regenerate from the repository root:

```bash
MPLCONFIGDIR=/tmp/l047-mpl .venv/bin/python labs/_figures_l047.py
.venv/bin/python labs/_build_l047.py
```

The first command exports and checks the lesson's SVG states with Node, renders
six mechanism diagrams with system librsvg/cairo, and plots three measured
reference panels with Matplotlib. `provenance.json` records source and PNG hashes.
The second command builds the student and solution notebooks and the prepared
student HTML, adjusting its relative lesson/reference links.

Check image packaging after rebuilding:

```bash
.venv/bin/python labs/_check_notebook_images_l047.py
```

This validates the eight active mechanism/result PNGs here plus the current
`../architecture-revision/0047-saint.png` overview (nine payloads) against the notebooks.
The older `architecture.png` and `model-architecture.png` are superseded by that overview.
It checks both Markdown images and the overview's HTML image with its horizontal-scroll container in the notebooks
and prepared HTML. It is a format/payload check, not a live Colab browser test.

Mechanism figures contain synthetic illustrations. `scores`, `ranks`, and
`context-results` use the committed `_verify_l047_results.json`; their captions
label them as author-reference evidence, separate from a student's current run.

Building resets the solution's outputs. Execute it again to verify changed code:

```bash
OMP_NUM_THREADS=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  labs/solutions/0047-saint.ipynb --ExecutePreprocessor.timeout=300
```

The original rendering used librsvg and Matplotlib. For current browser verification
and remaining delivery limits see `reviews/lesson-quality-audit-047-070/047.md`.
