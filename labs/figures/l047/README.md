# SAINT notebook figures

These PNGs are embedded as markdown attachments in both Lesson 047 notebooks.
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

Mechanism figures contain synthetic illustrations. `scores`, `ranks`, and
`context-results` use the committed `_verify_l047_results.json`; their captions
label them as author-reference evidence, separate from a student's current run.

Building resets the solution's outputs. Execute it again to verify changed code:

```bash
OMP_NUM_THREADS=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  labs/solutions/0047-saint.ipynb --ExecutePreprocessor.timeout=300
```

Browser rendering was not available for this revision. The figures were rendered
and visually inspected using the installed SVG renderer and Matplotlib; this is
not a browser verification claim.
