# L071 authoring and evidence review

## Teaching contract

The planned skill is to implement masked tabular pretraining, fine-tune with few labels,
and produce a label-efficiency curve. The package includes the released frozen-encoder
path and marks fine-tuning as an extension. It does not assume a positive gain.

The lesson begins with retrieval, defines feature/label/encoder/head/pretext tasks,
walks a collision example through actual-change targets, derives the two losses, shows
both pretext heads and the prediction-time path, and separates the paper's clean-reference
consistency equation from the release's logit variance. Three TODO functions feed actual
training and receive immediate arithmetic/gradient checks. The notebook exposes every
model and training function. The EXIT combines a complete experiment with a written defense.

## Evidence

- Source pinned to 996c58cf4c570061b30c38ecf2a754a9af85aafd; individual file hashes in
  `labs/_sources_l071.json`. Original NumPy corruption and the torch operation agree exactly
  on the tested donor-collision fixture. No historical Keras/TensorFlow training replay.
- `labs/_verify_l071_results.json`: 45 fresh downstream fits on sklearn digits, 5 arms,
  3 budgets, 3 paired seeds. Each seed has 809 unlabeled and 450 held-out test rows.
  Labeled budgets include validation. Frozen encoders have zero recorded movement;
  updated encoders have nonzero movement.
- Mean fine-tuned VIME minus scratch accuracy is −1.63, −2.30, −1.11 percentage points
  at 50, 150 and 500 labels. Report every seed difference. This is a local negative
  finding with a fixed recipe, not a refutation of the paper.
- Named paper target: Table 2 MNIST, supervised-only .9387 ± .0014, self-only
  .9406 ± .0019, full VIME .9577 ± .0022. These are separate cited observations.
  All local-paper comparisons remain INCOMPARABLE.
- The executed-notebook and copied-delivery outcomes are recorded separately in
  `labs/_execution_l071_results.json` and `labs/_delivery_l071_results.json`.

## Visual and access review

The existing missingness visual teaches missing-data mechanisms. This lesson's corruption
trace instead exposes the donor value, sampled selection, corrupted row and actual-change
target together. Its controls preserve the clean baseline. A separate consistency control
shows how changing the clean reference alters anchor MSE while leaving view variance fixed.
The architecture distinguishes the training heads from the retained inference path. Loss
contributions are computed from the same worked example. Results retain seed points and a
paired-gain detail view; standard deviation bars are labeled correctly.

Headless Chromium checks the controls, keyboard changes, image loads and page overflow at
1000 and 375 pixels. Screenshots were inspected; mobile table headings were shortened and
spacing corrected. Wide figures use an explicitly labeled, keyboard-focusable scroll region.
The static links work without JavaScript, and HTTP navigation finds L071 through the updated
manifest. Notebook figures are inline PNG data URIs. Copied staging follows the relevant
Pages directories and supplementary evidence copies.

## Remaining boundaries

Live Colab: NOT_CHECKED. Public MNIST closer/paper runs: NOT_RUN. Modal execution: NOT_RUN.
Remote Pages publication: NOT_RUN. The full original benchmark, search and baseline roster
are not reproduced. Authored availability does not update the learner's mastery/completion.

Final local delivery: 45 solution fits exactly replayed, 92 staged links/anchors passed, five embedded figures matched their generated files, and source/solution AST parity passed. Public MNIST archive SHA-256, shapes and scaling also passed; training remains NOT_RUN.
