# L072 reproducibility contract

Run from the repository root with the course environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l072.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_source_check_l072.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l072.py
.venv/bin/python labs/_build_l072.py
.venv/bin/python labs/_delivery_l072.py
```

The source audit fetches four files from pinned AstraZeneca/SubTab commit
`aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc` on first run and reuses the archived copies.
It checks NT-Xent outputs and input gradients, not the full model or optimizer trajectory.
SCARF is an independent reading of Algorithm 1 (arXiv v2); author-code parity is not established.

## Local protocol

Wine, breast cancer and digits are sklearn-bundled real datasets, available offline.
Data hashes, versions, splits, selected regularization, predictions and pretraining curves
are in `_verify_l072_results.json`. Original SCARF OpenML-CC18 and SubTab benchmark
protocols are not reproduced. Digits is an image-derived teaching dataset; a score on it is
not evidence of general success on business tables.

Three paired split/initialization seeds; approximate 70/10/20 stratified split; 25% of the
training labels fit each probe, all validation labels select C in [0.1,1,10]. Validation
labels count in the budget. All training features may fit the input scaler and pretrainer;
held-out features may not. Probe standardization fits labeled training features only.
Fixed 40 pretraining epochs, batch64, Adam .001, width64, projection32, SCARF rate.6/tau1.
Six recipes produce 54 evaluation records and 162 candidate probe fits.

SCARF uses exactly floor(c*d) chosen positions and independent marginal donors from training.
Its N-way CE differs from the displayed paper objective by +log(batch size); gradients agree.
The local SCARF architecture is compact and the downstream encoder is frozen, unlike paper
fine-tuning. No categorical embedding or missing-value pathway is implemented.

SubTab follows released contiguous windows with floor division and overlap .75.
For d not divisible by 3, tail features can be missing from all encoder inputs; they remain
reconstruction targets. The local decoder and encoder widths, training-only Gaussian noise,
normalization, fixed schedule and datasets are explicit teaching choices. Reconstruction and
projection-distance use sum-of-squares per row; losses average views/pairs. The same subset
encoder runs on every view; noise is disabled at inference and latents average over views.
No batch normalization or dropout: reuse of computed views across pairs preserves the
pair-average objective in this implementation. Full-model parity is not claimed.

## More compute, same visible code

```bash
.venv/bin/python labs/_run_l072.py --preset smoke
.venv/bin/python labs/_run_l072.py --preset closer
modal run --detach modal/l072_paper_repro.py --preset closer
```

`smoke`: wine/one seed/two epochs; `closer`: all three datasets, five seeds, 200 epochs.
The latter is a convergence/stability follow-up, **not** a paper-fidelity preset. It retains
the compact architecture, frozen probes and substitute suite. `paper` is deliberately
rejected: implementing exact full benchmarks and paper-specific supervised paths remains
unrun work, and a misleading preset would hide it. Outputs are written to a new result file;
there is no resume/cache of fitted results and no cloud job is launched by the builder.
The notebook's gated follow-up calls the student's live functions with this same configuration.

Ledger: equations/operator checks = PASS when recorded; local scores = measured;
original paper table = INCOMPARABLE; larger run = NOT_RUN until executed;
live Colab/deployment = NOT_CHECKED. Browser and solution execution are separately reported
in `_delivery_l072_results.json`. No learner mastery is inferred from author execution.
