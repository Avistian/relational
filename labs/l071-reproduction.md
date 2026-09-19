> **Reproduction repair, 2026-09-19.** The former larger-run recipe below is a teaching extension. The main notebook now includes a separate, complete inline paper/release implementation and executable protocol. See [the reproduction guide](reproductions/README.md) for targets, commands and unresolved gaps. Source parity, a longer teaching run and a close score do not certify exact paper reproduction.

# L071 reproduction contract

## Existing teaching implementation

A numeric-data PyTorch port of the released one-layer d→d encoder, two d-output pretext
heads, and two-100-hidden-layer downstream predictor. Corruption uses independently
permuted columns and **actual-change** mask targets. Pretraining uses mean BCE + 2×mean
MSE over all features and one fixed corruption bank. The semi-supervised arm freezes the
encoder and uses population **logit** variance across three augmented views, with beta=1.
Fine-tuning is an explicit extension; the released downstream code freezes the encoder.

Source: https://github.com/jsyoon0823/VIME/tree/996c58cf4c570061b30c38ecf2a754a9af85aafd
Paper: https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf
Supplement: https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Supplemental.pdf

The paper uses sampled masks in its equations and a clean-prediction consistency reference.
The released code recomputes changed cells and minimizes augmented-view logit variance.
These differences are taught and checked. Exact corruption parity on the saved fixture
establishes only that operation; historical Keras/TensorFlow execution is now tracked separately in `reproductions/vime-release-results.json`.

## Local experiment

From the repository root:

```bash
.venv/bin/python labs/_check_l071.py
.venv/bin/python labs/_source_check_l071.py
.venv/bin/python labs/_verify_l071.py
.venv/bin/python labs/_build_l071.py
.venv/bin/python labs/_execute_l071.py
```

Offline sklearn digits (1,797 × 64), intensities divided by the documented maximum 16.
Three seeds, 450 test rows per seed, 809 fixed unlabeled rows and 538 label-reserve rows.
Label budgets 50/150/500 include 20% validation labels. Budget selections are nested;
train/validation roles are resampled inside each budget. Unused reserve rows stay unused.
Five paired arms share split, dimensions, head initialization, supervised minibatch ordering,
60 downstream epochs and validation selection. Pretraining has 30 epochs, batch 128,
p=0.3, alpha=2, RMSprop lr=.001, alpha=.9, eps=1e-7. Downstream Adam lr=.001, batch 64.
The selected model minimizes validation cross-entropy; all fits finish before test scoring
for that fit. Test outcomes never feed selection or subsequent training choices.

`_verify_l071_results.json` records configuration, environment, input hash, implementation
hash, split row IDs, pretext loss traces, selected epoch, encoder movement, test accuracy
and log loss. Dots are seeds and error bars are sample SD, not population confidence
intervals. More unlabeled data and compute are allowed to the pretrained arms; this is a
label-budget comparison. One image-derived table is not a tabular benchmark suite.

## Public MNIST follow-up: after EXIT

The same code loads the standard Keras MNIST archive (60,000 train / 10,000 test),
checks its archive checksum, flattens 28×28 images to 784 numeric features and divides
by 255. Test rows are kept separate throughout pretraining and selection.

```bash
# Quick operator check on digits
.venv/bin/python labs/_run_l071.py --preset smoke --output /tmp/l071-smoke.json
# Public dataset, more labels, same implementation (CPU or CUDA)
.venv/bin/python labs/_run_l071.py --preset closer --device cuda --output l071-mnist.json
# Unattended GPU operator; result persists in Modal volume relational-l071-results
modal run --detach modal/l071_paper_repro.py --preset closer
modal volume ls relational-l071-results
# Substitute the actual filename printed by the job
modal volume get relational-l071-results closer-TIMESTAMP.json ./l071-mnist.json
```

The notebook includes the corresponding `RUN_TEACHING_EXTENSION = False` cell with the full loop
already visible. Enable only when ready for the longer run. The `extended` teaching preset increases
to ten seeds and 100/100 pretrain/downstream epochs; its name does not certify fidelity.

| Track | Data / seeds / epochs | Evidence status at creation |
|---|---|---|
| Local | digits / 3 / 30 + 60 | See measured JSON |
| smoke | digits / 1 / 2 + 3 | Operator smoke check |
| closer | MNIST / 3 / 30 + 60 | NOT_RUN |
| extended | MNIST / 10 / 100 + 100 | NOT_RUN |

## Named published target (not reproduced)

Paper Table 2, MNIST, accuracy mean ± SD over ten runs: supervised-only 0.9387 ± 0.0014;
self-SL-only 0.9406 ± 0.0019; full VIME 0.9577 ± 0.0022. The self-only gain is 0.0019,
and the full-VIME gain is 0.0190 over supervised-only. Do not compare the local digits curve
to these as a replication; do not accept a MATCH based only on nearby numbers.

## Remaining gaps before any paper-table claim

The public paper protocol uses 10% labeled and 90% unlabeled training rows; this controlled
follow-up keeps a 60% unlabeled pool and 40% label reserve. The paper optimizes architecture
width/depth and hyperparameters; this runner fixes them. Optimizer and initialization
semantics differ across frameworks. The scratch control retains an extra randomly initialized encoder to match the pretrained arm;
it differs from the paper's two-layer-perceptron supervised-only baseline. The original baseline
roster and Income/Blog experiments are absent. Genomic and clinical access/protocols are not supplied. To reproduce a paper table,
select its exact experiment, implement its split/preprocessing/search and baselines, record
source/data identities, freeze choices, then compare the same metric over aligned repetitions.
Do not use a digits accuracy as a comparison to a published MNIST, clinical or genomic score.

Conclusion ledger: **verified here** = tested operators and local fits; **paper claim** = cited
original experiments; **scale-up** = public follow-up NOT_RUN. Paper-table verdict remains
**INCOMPARABLE**. Live Colab, Modal execution and remote Pages deployment are independent
checks and remain NOT_CHECKED unless separately executed.

Operator references: [Modal local files](https://modal.com/docs/guide/local-files) and [persistent volumes / downloads](https://modal.com/docs/guide/volumes). These APIs were checked while preparing the operator; no remote job was launched.

Public MNIST data access was checked: see `_mnist_access_l071_results.json`. The archive
identity is the SHA-256 published by [Keras's loader](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py). That data-access check did not train MNIST; the later historical replay has its own measured evidence.
