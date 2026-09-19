# L073 reproducibility contract

This is an evaluation lesson, not a new architecture or paper-table reproduction.

## Verified local protocol

Three offline sklearn datasets: wine, breast_cancer, digits (flattened images). Split/model seeds 0,1,2. Stratified outer 70/10/20 splits; within training, one label-blind permutation seeded seed+73, prefixes floor(f*n), f=0.1,0.2,0.4,0.7,1.0. Dataset bytes hashed; exact split/subset IDs retained. Known class vocabulary comes from dataset metadata. No class-based label subset repair.

Six arms: raw standardized logistic; random SCARF encoder + frozen logistic; trained SCARF encoder + frozen logistic; random encoder fine-tuned from scratch; pretrained encoder fine-tuned; fixed HistGradientBoostingClassifier. All arms share full-training-pool feature standardization. Three logistic C values .1/1/10 chosen on validation; first wins ties. Probe scaler fits labeled rows only. Fixed trees use 100 iterations, 15 leaves, minimum leaf 5, L2=1, no early stopping. Neural encoder d→64→64 ReLU; projector 64→64→32. SCARF: 40 epochs, independent within-column donors, floor(.6*d) selected cells, N-way cosine CE, tau1, batch64, Adam .001. Fine-tuning: fresh copied encoder plus linear class head; 60 epochs, same batches/head initialization and Adam .001 for scratch and pretrained. Different budgets never warm-start each other. Extra pretraining compute is not matched.

270 selected test evaluations. Prediction vectors, targets, label budgets, C-selection trials, pretraining loss traces, training deltas, source hashes and environment versions are saved. Test labels are used only in scoring and reported analysis. Protocol was fixed before that analysis. Probe representations cannot move because sklearn receives detached feature arrays. Source definitions carried forward from L072 are visible in the notebook.

## Meaning and limitations

Primary contrast: scarf_ft minus scratch. Secondary mechanism contrast: scarf_frozen minus random. Raw and tree contrasts are recipe comparisons. Paired t95 intervals describe three overlapping split/model repetitions, not independent-dataset or simultaneous confidence. Crossings are all adjacent strict mean-sign changes; exact zero ties are separate. Dataset ranks average seeds first, then rank methods within dataset, with equal dataset weight. Friedman/Nemenyi summaries are exploratory on three datasets and five related budgets.

Shared train-feature preprocessing means the scratch baseline also uses unlabeled feature statistics. Available validation labels count in every development budget, even for fixed recipes that do not use them. Outer stratification uses benchmark labels; inner selection is label blind. Tiny samples may lack a class; one-class probes fail openly. The full unlabeled pool stays fixed, so unlabeled-quantity or shift effects are NOT_RUN.

## Paper / local / follow-up ledger

- Paper claim: SCARF pretraining followed by fine-tuning is evaluated in the authors' benchmark. Source: https://arxiv.org/html/2106.15147v2 . Cited, not reproduced.
- Local evidence: `_verify_l073_results.json` gives actual scores, repetitions, IDs and budget accounting. INCOMPARABLE to original paper tables.
- Deviations: 3 small sklearn datasets instead of the 69-task suite; 3 seeds instead of 30 trials; compact networks, fixed epoch counts, different selection recipe and budget grid. Digits is image-derived. No tuned broad baseline claim.
- Follow-up: notebook `RUN_LONGER=True` executes the same visible functions with five seeds and 200/200 epochs. NOT_RUN; this audits convergence, not benchmark fidelity. No cloud job is required for this CPU evaluation lesson.
- Browser, execution and copied-Pages statuses are recorded in separate delivery results. Live Colab and deployment NOT_CHECKED.

## Re-run

From repository root, use the installed course environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l073.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l073.py
.venv/bin/python labs/_build_l073.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l073.py
.venv/bin/python labs/_delivery_l073.py
```

Student notebook definitions execute in the current kernel and feed fresh training and summaries. The teacher execution compares complete prediction records against the author run. Students should not import solution definitions to bypass TODOs. No paper-parity tolerance is defined because the protocols are incomparable.
