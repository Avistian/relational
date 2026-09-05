# L049 reproduction guide

The main artifact is [the student notebook](0049-excelformer-trompt.ipynb). Its model and training loop are visible, and its attention/augmentation functions stay in the trained path. Trompt is scoped to Eq.4–5; no full Trompt encoder is trained here.

Run from the repository root with the existing lab environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_reference_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_paper_repro_l049.py --preset closer --out labs/data/cache/l049-new-closer
```

The reference checker downloads one pinned source file for validation. The data loader downloads the authors' three small split files when absent and requires exact SHA-256 fingerprints. MovieLens uses the already-established checked archive. Neither downloader disables TLS checks.

The short experiment uses three author-split numeric tasks and a separate MovieLens random/time probe, three seeds each. The larger run is a single-task Pima attempt with greater architecture capacity; the `paper` preset increases training caps but still does not reproduce the published search or full suite. The source target is ExcelFormer v5 **Table 14**, **Pima-Indians-Diabetes**, Feat-Mix default AUROC **0.8356**. Do not substitute the differently named Pima rows.

Keep the three evidence levels separate: checked implementation path; local scores; cited paper claim. Upstream preprocessing fit scope, augmentation sampler equivalence, original multi-run partitioning and the tuning search remain gaps. The larger-run verdict is **INCOMPARABLE**, even if a rounded score is close. A chronological transfer ranking is not a refutation of an IID benchmark.

The larger runner saves a contract, per-seed results, selected model weights, test probabilities and histories. It resumes only a matching contract; choose a fresh output directory after code/configuration/environment changes. The author closer directory was created before helper-class fingerprint coverage was expanded, so preserve it as historical evidence and use a new directory for new runs.

For a student GPU run, enable the notebook's `RUN_PAPER_REPRO` gate. For unattended execution, the separate operator is:

```bash
modal run --detach modal/l049_paper_repro.py --preset closer
modal volume get relational-artifacts l049 ./l049-artifacts
```

No cloud operator was run during authoring. Local smoke execution and resume/rejection checks cover the operator's Python runner, not a deployed Modal image.

To regenerate teaching artifacts after producing results:

```bash
.venv/bin/python labs/_publish_l049_evidence.py
.venv/bin/python labs/_figures_l049.py
.venv/bin/python labs/_build_l049.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 labs/solutions/0049-excelformer-trompt.ipynb
.venv/bin/python labs/_render_l049.py
.venv/bin/python labs/_delivery_check_l049.py
node labs/_viz_check_l049.js
node labs/_check_pedagogy.js
```

The teacher solution directory is gitignored, following the course convention. The prepared HTML is generated from the student notebook and keeps exercises unfilled. Embedded PNGs use Colab's data-URL representation; browser and live-Colab checks were not performed because no browser was installed. Arithmetic, widget handlers, diagram text bounds and notebook image integrity are checked separately.
