# L080 reproducibility contract

This is the curriculum's cross-paper Year 2 exit experiment, **not a published experiment from any one paper**. The full declared local experiment is executable and its raw inputs are packaged. The architecture sources are visible in the notebook. Larger local runs are explicitly separate from original paper benchmark reproduction.

## Exact replay

From a course checkout, use Python 3.12 and `requirements-labs.txt`; the exact author package snapshot is `requirements-l080-observed.txt`. The observed snapshot is evidence of the environment, not a cross-platform dependency lock. Raw subsets under `labs/data/l080` include all numeric/binary features and exact row IDs/targets from the hash-pinned TabReD release.

```bash
.venv/bin/python labs/_check_l080.py
.venv/bin/python labs/_verify_l080.py --preset smoke --output /tmp/l080-smoke-new.json
.venv/bin/python labs/_verify_l080.py --preset exam --output /tmp/l080-exam-new.json
.venv/bin/python labs/_build_l080.py
.venv/bin/python labs/_execute_l080.py
```

Every result path must be new. There is no resume. A failure leaves `IN_PROGRESS`; never summarize it as complete. The operator checks the packaged raw-data SHA256 and the complete v2 checkpoint SHA256. If absent, the historical checkpoint is downloaded from its pinned Hugging Face revision. No API inference or paid service is used.

To reconstruct the subsets from original release archives rather than use packaged inputs:

```bash
.venv/bin/python labs/_prepare_l080.py --preset exam
```

`_fetch_l055.py` verifies official archive hashes. The packaged extraction records exact sampled IDs and full split time ranges. Data files are compressed without lossy rounding. Numeric data retain float32 as in the release; labels are int64. Sampling seeds 550/551/552 are fixed, label-blind and distinct from model seeds.

## Frozen local protocol

- Two real classification tasks: Ecom Offers and Homesite Insurance. Released random-0 and sliding-window-0 partitions. These are different train/test populations, not a controlled causal drift intervention.
- `exam`: 128 train, 48 validation, 48 test rows per panel; seeds 0/1/2; four arms; exactly two candidates each; 48 selected test records. `smoke`: 64/24/24, one seed, two epochs and 20 trees, 16 records. `extended`: 512/192/192, three seeds, 128 epochs and 800 trees; not run during authoring.
- All available numeric/binary columns (119 and 276), no feature selection. Categorical columns are omitted and counted. Train-only median imputation (all-missing columns → zero); neural gradient-trained arms additionally use train-only standardization. TabPFN receives imputed raw numeric values and its documented context-fitted numeric encoding.
- Binary log loss of P(y=1); float64-epsilon clipping. Candidates and gradient checkpoints use validation only; exact ties choose first. Test labels never reach fit or selection. No train+validation refit, repeated test selection or class-label context contamination.
- XGBoost: 120 rounds, depth 3/6, learning rate .05, subsample/column sample .8, CPU hist. This is **two-candidate tuning**, not a 4-hour GBDT search.
- FT-Transformer: full pinned released feature tokenizer, three PreNorm blocks with first attention norm omitted, ReGLU, CLS readout; width32, 4 heads, FFN factor4/3, attention/FFN dropout .1, residual dropout0; no compression. TabM: corrected numeric full BatchEnsemble variant, k8, width64, depth3, dropout .1; common shuffled batches. Both use AdamW, lr .001/.003, decay1e-4, batch32, 16 epochs with first best-validation checkpoint.
- Historical TabPFN v2: all 12 layers and every checkpoint tensor, frozen; temperature .7/1 selected on validation. No new pretraining. Simple numeric wrapper, no default feature/class ensemble, no categorical preprocessing/fingerprints. One whole validation query call, one whole test query call; query batch is part of the protocol. Model seed controls random feature identifiers, not pretraining uncertainty.
- Same rows, targets, metric and two declared candidates are matched. Candidate count does **not** equalize FLOPs, validation adaptivity, historical pretraining cost or serving latency. Per-arm seconds include candidate fitting/validation and selected prediction, exclude shared imputation, checkpoint download/load and common data loading. They are diagnostic host times, not a hardware-normalized efficiency ranking.
- Mean and sample SD across the three downstream seeds within each dataset/regime; ranks within dataset after seed averaging. Two datasets cannot support a broad winner or a reliable population significance claim. No Friedman/Nemenyi significance is asserted.
- Temporal row order does not prove every released aggregate and label was available at prediction time. Full split boundaries and strict-versus-tied status are recorded. Features are inherited from a release, not reconstructed from raw event histories.

## Source and paper protocol audit

| Component | Reproduced here | Difference from original results |
|---|---|---|
| FT-Transformer | Pinned released full architecture; copied-weight output and gradient check | Small fixed recipe, numeric-only task subset; no paper tuning/benchmark |
| TabM | Corrected full numeric BatchEnsemble, shared weights/adapters and mean member loss | k8 vs paper k32, common batches, small fixed grid, no numerical embeddings/quantile recipe/full tuning |
| TabPFN v2 | Full historical weights, all layers, simple declared numeric inference | No new pretraining, restricted wrapper, two temperatures, different evaluation data/protocol |
| Tuned GBDT | Fresh two-depth XGBoost selection | Neither paper-optimal tuning nor 4-hour Nature comparator |
| TabReD | Hash-pinned release data and split0, exact sampled row identities | Two tasks, caps, categorical omission, log loss; not Figure2's roster, task metrics, all splits and 15 seeds |
| TabICL | Written mechanism assessment with primary source | No TabICL inference arm is requested by this exam; NOT_RUN |

Original FT, TabM, TabPFN and TabReD paper-result reproduction: **NOT_RUN / INCOMPARABLE**. A matching score would not change that status. No fabricated paper target exists for this four-way cross-paper protocol. Follow the prior lesson contracts (L054, L055, L064, L066) for their individual published targets and unresolved gaps. Their old `paper` preset labels are not certificates of fidelity.

## Larger local extension (not original paper reproduction)

```bash
.venv/bin/python labs/_prepare_l080.py --preset extended
.venv/bin/python labs/_verify_l080.py --preset extended --output /tmp/l080-extended-new.json
```

The notebook exposes the same complete live operator behind `RUN_FRESH_COMPARISON`. Default solution execution audits all frozen exam predictions and performs the full four-arm/two-regime smoke with the student's functions. To reproduce the full author experiment choose `exam`. Use a new path. The notebook records its actual executed cell source alongside the result; repository hashes alone do not identify edited kernel code.

Live Colab, cloud execution, deployment and original paper pretraining remain NOT_CHECKED/NOT_RUN. The student must submit a fresh run plus written reasoning; author evidence is not the student's exit ticket.
