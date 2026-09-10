# L049 reproduction guide

The main artifact is [the student notebook](0049-excelformer-trompt.ipynb). Its model and training loop are visible, and its attention/augmentation functions stay in the trained path. The numeric Trompt path now includes Fig.3 prompt fusion, numeric embeddings, learned expansion/ReLU/GroupNorm/residual, repeated cells, the shared Fig.4 downstream head and Eq.9 summed cell losses. Its separate Pima execution probe is not a benchmark comparison.

Run from the repository root with the existing lab environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l049.py
.venv/bin/python labs/_check_trompt_l049.py
.venv/bin/python labs/_reference_trompt_l049.py
.venv/bin/python labs/_verify_trompt_l049.py
.venv/bin/python labs/_check_resume_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_reference_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l049.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_paper_repro_l049.py --preset closer --out labs/data/cache/l049-new-closer
```

The reference checker downloads one pinned source file for validation. The data loader downloads the authors' three small split files when absent and requires exact SHA-256 fingerprints. MovieLens uses the already-established checked archive. Neither downloader disables TLS checks.

The short experiment uses three author-split numeric tasks and a separate MovieLens random/time probe, three seeds each. The larger run is a single-task Pima attempt with greater architecture capacity; the `paper` preset increases training caps but still does not reproduce the published search or full suite. The source target is ExcelFormer v5 **Table 14**, **Pima-Indians-Diabetes**, Feat-Mix default AUROC **0.8356**. Do not substitute the differently named Pima rows.

Keep the three evidence levels separate: checked implementation path; local scores; cited paper claim. Upstream preprocessing fit scope, original multi-run partitioning and tuning remain gaps. The exact sampler mismatch is now identified: the release draws one Beta rate per row with NumPy, while the preserved local loop draws one rate per minibatch with Torch masks. The release also uses ten warmup epochs and cosine decay over a 500-epoch cap and selects by validation task metric; the local loop uses fixed learning rate and log-loss selection. The larger-run verdict is **INCOMPARABLE**, even if a rounded score is close. A chronological transfer ranking is not a refutation of an IID benchmark.

The repaired runner uses semantic-code-v1 identities, stable across notebook filenames and execution bookkeeping. Historical raw-source/marshal contracts require a fresh output directory; their evidence is preserved. The larger runner saves a contract, per-seed results, selected model weights, test probabilities and histories. It resumes only a matching contract; choose a fresh output directory after code/configuration/environment changes. The author closer directory was created before helper-class fingerprint coverage was expanded, so preserve it as historical evidence and use a new directory for new runs.

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

The teacher solution directory is gitignored, following the course convention. The prepared HTML is generated from the student notebook and keeps exercises unfilled. Embedded PNGs use Colab's data-URL representation; the 2026-09-10 repair exported and inspected the Trompt diagram in local Chromium. Parent audit artifacts record independent desktop/mobile and copied-Pages checks; live Colab remains unchecked. Arithmetic, widget handlers, diagram text bounds and notebook image integrity are checked separately.

## Trompt source and execution contract

`relkit/trompt_l049.py` is separate from `claim_models.py`, preserving the original ExcelFormer result/source fingerprint. The paper provides the zero initial previous state (§5.1), the two softmax axes, recurrence, normalization placement (Fig.3/4) and loss/prediction aggregation. The numeric implementation makes explicit choices where source code is unavailable: a bias-free scalar→P expansion, two GroupNorm groups, .01 normal embedding initialization, AdamW at .001, batch64, validation cross-entropy selection, and logits averaged before softmax. Categorical features and the paper's full training/search protocol are not implemented.

The independent PyTorch Frame component at commit `3ed76b132242be6ce5f63851052e114e57aaec5d` supplies a copied-weight validation point for the cell after encoding and the shared head. It is **not an identified author release**. Its full model starts from learned state, whereas our mirror follows the paper's zero state. `_reference_trompt_l049_results.json` records source SHA256 values and observed errors; `_check_trompt_l049_results.json` records structural interventions and gradients.

The separate measured Pima probe is `_verify_trompt_l049_results.json`; the notebook reruns it through the student's functions, records live routing calls, and includes its evidence in EXIT. It is **INCOMPARABLE** to Trompt's Grinsztajn45 accuracy/R² results. Do not add it to the older ExcelFormer table: that would change the comparison without an aligned multi-dataset experiment.

A further capacity check is executable (but not run in this audit):

```bash
.venv/bin/python labs/_verify_trompt_l049.py --capacity --device cuda --out /tmp/l049-trompt-capacity.json
```

It uses d=P=128,L=6 and 100 epochs on Pima, not the paper benchmark. The corresponding notebook gate stays off. Full Trompt reproduction still needs its benchmark IDs/partitions, categorical path, preprocessing and 40 unique search configurations; the paper's oversampled search curve must not be interpreted as hundreds of unique fits.
