# Lessons 071–074: paper reproduction tracks

Each main lab now includes the complete executable source for its reproduction track after EXIT. The earlier compact experiments remain separate teaching extensions. The file-writing cells expose the model and training code and reconstruct exactly what the following runner imports.

The [execution ledger](execution_evidence.json) records completed scope, commands, environments, result hashes, rejected evidence and unrun work. Delivery checks live in [the package audit](../_paper_reproduction_audit_results.json) and [the incremental notebook execution record](../_paper_appendix_execution_results.json).

| Lesson | Reproduction target | What changed from the teaching experiment |
|---|---|---|
| 071 | VIME Table 2 MNIST, complete released three-arm pipeline at 6000 labels | Original Keras/TF optimizers and models; 54000 unlabeled rows; 10 self epochs; 1000 semi-supervised updates; frozen transfer; full test set |
| 072 | SCARF Tables 2 and 4; SubTab MNIST release compared with Table A3 | Full SCARF model and encoder fine-tuning; original feature corruption; full SubTab architecture, six-pair loss, noise, schedule and probe sweep |
| 073 | SCARF 100% versus 25% training labels | Same paper implementation as L072, 30 paired 70/10/20 splits, fixed validation/test fractions |
| 074 | CARTE released benchmark rows for three wine datasets | Full released graph transform, model, prediction head, checkpoint loading, stopping, fifteen-model ensemble and published per-split hyperparameters |

## What exact reproduction can and cannot mean here

These operators attempt identified experiments. They do not infer protocol identity from a close score. A smoke result is always `INCOMPARABLE`. Full outputs retain paper reference values separately from measurements and enumerate unresolved details.

- **VIME:** supplementary Section 5 describes validation-selected architectures, but the repository supplies a fixed architecture without the table's selected configurations. The historical release replay is executable; exact Table 2 protocol remains incomplete. No invented architecture search is presented as the authors' search.
- **SCARF:** Algorithm 1 and Section 4 specify architecture, optimizer, stopping, corruption, label fractions and repetition count. Author seeds/code and some initialization/preprocessing details are not published. We declare reconstruction seeds 0–29, unstratified random splits, train-fitted numerical scaling, and PyTorch initialization. Missing-value imputation and categorical vocabulary use the full table, preserving the paper's stated transductive imputation. The two-layer pretraining head ends in ReLU. This is an independent paper reconstruction, not author-code parity.
- **SubTab:** the complete model, objective, gradients and one AdamW update are checked against the released implementation. Evaluation disables noise as the release's train.py and eval.py entry scripts require and reports every regularization value; it does not select the best test score as an unbiased estimate. The release does not identify a checkpoint or C for Table A3. Random stream ordering and modern numerical libraries may differ.
- **CARTE:** full downstream release replay uses the released checkpoint, not freshly reproduced YAGO pretraining. The default pins June 18, 2024 source (`e1079b5f470094427935763daab74e33a23e3ed0`), contemporary with the paper; `--release current` retains the later release as an explicit alternative. The authors do not identify an exact execution commit. The original scatter-sum has a checked native-PyTorch fallback when the compiled extension is absent. Input and checkpoint hashes are recorded. `initial_x` is handled by the release loader, with no silent substitution of the teaching track's strict transfer. Numerical failure is recorded per row rather than silently changing preprocessing. A compatibility shim restores `_estimator_type` when running the release with scikit-learn 1.8+; it does not change learning. Other dataset families and joint source-table transfer are not covered by these single-table rows.

The unresolved points above are remaining reproduction gaps, not waived requirements. Never describe all four original papers as fully reproduced.

## Measured discrepancies to investigate

The completed SCARF reconstruction contains 360 fits: three datasets, 30 splits, two label fractions, and two arms. It does **not** reproduce every reported mean. On OpenML 40975, the full-label control reaches 72.92% versus the paper's 95.27%; 27 of its 30 runs stop after four epochs and predict one class. At 25% labels, 23 of 30 SCARF runs predict one class, yielding 74.28% versus 90.55%. The stored validation traces expose this failure under the declared initialization and patience rule. These observations do not identify the cause or justify changing patience after seeing test scores. Other reconstruction choices and unreleased author settings remain candidates for a separately specified diagnostic experiment.

The original VIME pipeline completed one full-data trial: supervised 93.49%, self 95.47%, semi 95.63%, against Table 2 means 93.87%, 94.06%, 95.77%. This verifies execution; it does not complete the declared ten repetitions or recover the paper's unpublished architecture choices. Saved JSON files and the notebooks retain all arm comparisons.

The June 2024 CARTE downstream replay completed the `wine_vivino_price / 32 labels / split 1` row with all fifteen models and full training/stopping limits: R² **0.53704**, released reference **0.53813**, gap **−0.00109**. It uses all 13,834 input rows and the complete test split. This close selected-row result does not establish full benchmark or pretraining reproduction; source/environment and historical-input uncertainties remain in the ledger.

The corrected SubTab run completed all 15 epochs on 60,000 MNIST training rows and evaluated all nine released probe regularizers on 10,000 test rows. Accuracy ranges from 95.02% to 98.31%; all nine scores appear in lesson 72 alongside the 97.86% Table A3 reference. No best-test setting is selected as an unbiased held-out estimate, and the unpublished Table A3 regularizer remains unresolved.

## Run locally

From the repository root, the modern tracks use the course environment plus `torch-geometric==2.8.0`, `fasttext-wheel==0.9.2`, `torcheval==0.0.7`, `skrub==0.10.1` and `pyarrow`. The historical VIME requirements are isolated in their own image.

```bash
# Execution checks, explicitly INCOMPARABLE; full architecture, shortened resources.
.venv/bin/python labs/reproductions/scarf.py --smoke --output scarf-smoke.json
.venv/bin/python labs/reproductions/subtab.py --smoke --output subtab-smoke.json

# SCARF: 3 named datasets × 30 splits × 2 label fractions × 2 arms = 360 fits.
.venv/bin/python labs/reproductions/scarf.py --output scarf-paper.json
# All 69 paper datasets; substantial runtime and downloads.
.venv/bin/python labs/reproductions/scarf.py --all-data --output scarf-all69.json

# SubTab: full 60000/10000 MNIST, 15 epochs, seed 57, nine probe regularizers.
.venv/bin/python labs/reproductions/subtab.py --device cuda --output subtab-paper.json

# Original VIME: x86_64 is required by historical TensorFlow wheels.
docker build --platform linux/amd64 -t relational-vime-paper -f labs/reproductions/Dockerfile.vime labs/reproductions
mkdir -p paper-evidence
docker run --rm --platform linux/amd64 -v "$PWD/paper-evidence:/evidence" relational-vime-paper

# CARTE: no 384-row cap. Download the complete FastText English model first.
.venv/bin/python labs/reproductions/carte.py --fasttext /path/to/cc.en.300.bin --checkpoint labs/data/l074/kg_pretrained.pt --device cuda --output carte-paper.json
# One named row, keeping all fifteen models and full training/stopping limits:
.venv/bin/python labs/reproductions/carte.py --fasttext /path/to/cc.en.300.bin --checkpoint labs/data/l074/kg_pretrained.pt --datasets wine_vivino_price --budgets 32 --seeds 1 --output carte-row.json
```

SCARF and CARTE persist completed records after each fit. SubTab prints every epoch and persists the final sweep. The paper default cannot quietly shrink rows or epochs; reduced resources require the explicit smoke flag. A selected CARTE row is a selected row, not the entire benchmark.

## Colab and unattended compute

The main notebooks include every implementation and a gated run cell. Run all PROVIDED source cells first. For modern tracks use the course dependencies. VIME's `colab_vime.py` installs a separate Python 3.7 environment using a pinned micromamba release; it preserves the notebook's own Python kernel. This historical path requires x86_64 and has a separate runtime check from local Docker. CARTE requires the full 7.2 GB FastText binary, available from the [official FastText download page](https://fasttext.cc/docs/en/crawl-vectors.html), plus enough host memory for the model and graphs. The teaching cache is insufficient.

```bash
modal run --detach modal/paper_071_074.py --method subtab
modal run --detach modal/paper_071_074.py --method scarf
modal volume get relational-paper-071-074 subtab-paper-results.json ./subtab-paper-results.json
# The corrected run also preserves its trained weights:
modal volume get relational-paper-071-074 subtab-paper-results.pt ./subtab-paper-results.pt
# CPU alternatives verified separately from GPU access:
modal run --detach modal/paper_subtab_cpu.py
modal run --detach modal/paper_vime_cpu.py --one-trial
```

The Modal operator preserves artifacts on a named volume. GPU availability and payment requirements are governed by the account; prepared code is not evidence of execution. VIME has its isolated historical container; CARTE can run locally or in a GPU runtime with the full model file. The earlier per-lesson `closer` operators are teaching extensions, not these paper tracks.

## Check source and protocol

```bash
.venv/bin/python labs/reproductions/test_protocols.py
```

The checks test original-column categorical corruption, SCARF's actual paper loss convention, full SubTab forward/loss/gradient/optimizer parity, and architecture/configuration invariants. `source_manifest.json` records source pins and SHA256 digests. Results and delivery status must be reported separately from these tests.

Primary sources: [VIME released code](https://github.com/jsyoon0823/VIME/tree/996c58cf4c570061b30c38ecf2a754a9af85aafd), [SCARF paper](https://arxiv.org/html/2106.15147v2), [SubTab released code](https://github.com/AstraZeneca/SubTab/tree/aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc), [CARTE June 2024 source](https://github.com/soda-inria/carte/tree/e1079b5f470094427935763daab74e33a23e3ed0) and [later data/score archive](https://github.com/soda-inria/carte/tree/f54690da4cddbedd1e1a9113a312f85783d2c125).

## Historical CARTE environment

`requirements-carte-paper.txt` preserves the June 2024 dependency versions where available. `Dockerfile.carte` isolates them from a current course environment. This additional container recipe is separate from the modern-runtime checks; consult the evidence status before assuming it has been executed.

```bash
docker build --platform linux/amd64 -t relational-carte-paper -f labs/reproductions/Dockerfile.carte labs/reproductions
docker run --rm --platform linux/amd64 -v /path/to/cc.en.300.bin:/input/cc.en.300.bin:ro -v "$PWD/labs/data/l074/kg_pretrained.pt:/input/kg_pretrained.pt:ro" -v "$PWD/paper-evidence:/evidence" relational-carte-paper --fasttext /input/cc.en.300.bin --checkpoint /input/kg_pretrained.pt --datasets wine_vivino_price --budgets 32 --seeds 1 --output /evidence/carte-row.json
```

The June 2024 checkpoint was downloaded and byte-matched against the shipped checkpoint; see `carte-checkpoint-identity.json`. A current-release wine split produced nonfinite graph values and is recorded in `carte-preprocessing-failure.json`. Do not replace this failure with a successful teaching-lane normalization and call it the same experiment.

## SubTab evaluation correction

The first author run retained corruption during evaluation. Both official `train.py` and `eval.py` entrypoints explicitly set `add_noise=False`, so that run is retained only as `subtab-noisy-evaluation-diagnostic.json`, with an INCOMPARABLE verdict. The corrected implementation tests clean evaluation windows directly and saves its trained state alongside the result JSON. Only `subtab-paper-results.json` is eligible for the paper comparison.

CARTE's full-table vector cache stores exact outputs of the fixed FastText model. It checks the original model hash, sampled vector equality, and the cache hash. Numeric transformations are still fitted separately inside each split. This cache reduces repeated model-loading cost without changing representations or using target labels.
