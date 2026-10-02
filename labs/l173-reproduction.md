# L173 F1 Multi-task Masked-cell Pretraining

Approved 2026-10-02. Course experiment, not RT or KumoRFM reproduction. This package implements the complete approved target population and all six fresh fits. Read `evidence/l173/report.md` for measured results and per-task metrics.

## Frozen protocol

- Input: authenticated L171 F1 archive and nine Parquet members, L172 explicit schema. 97,606 original rows, 67 columns. Every observed eligible numerical/category feature cell in the six dated tables becomes a target: 370,024 targets, 21 column tasks. Nonfeature roles/types, missing targets and all three untimed tables are excluded and counted in `coverage.json`.
- Split: date before 2005 train (244,410), calendar 2005 validation (6,354), 2006 onward test (119,260). These are course autocomplete splits, not RelBench forecasting labels. Fit means/population SD and sorted category vocabularies only on training. UNKNOWN is reserved class0 and remains included in test metrics.
- Context: first four other observed eligible cells in sorted column order from the same row, then first four eligible cells from dated FK-parent rows, sorted FK/column order. Parent event date <= query date. Untimed parents excluded. No reverse links, raw IDs, text or timestamps as model features. Every target identity removed from every slot. Same-row post-outcome proxies remain a limitation of autocomplete.
- Model: column embedding + numerical projection OR category embedding, width32. Masked means of row and parent groups, concatenate target-column embedding (96 total). Shared ReLU MLP96→64→32. Per-column linear head, numerical output1 or categorical vocabulary-size+1. Padding excluded from pooling. No attention or learned cross-schema semantics.
- Loss: numerical Huber delta1 on train-normalized values; categorical cross-entropy. Cell arm averages examples. Task arm uses global training-count importance weights N/(21 N_t), then averages examples. No within-batch task renormalization.
- Seeds0/1/2, all training targets once per epoch, three complete epochs, batch1024, Adam lr0.001 with default betas/epsilon and no weight decay. CPU one thread, deterministic Torch algorithms. Both arms share initialization and NumPy permutation streams. Targets are masked individually, not randomly subsampled.
- Select earliest minimum validation macro task loss from three saved checkpoints for both arms. Test only selected checkpoint. No test-based hyperparameter selection. Save all18 checkpoints, full logits for715,560 test predictions, per-task losses and metrics, input/source hashes and seed/batch pairing receipts.
- Baseline: normalized numeric training mean0 and Laplace(+1)-smoothed training categorical frequencies. Full test population. Raw-unit numerical MAE and categorical accuracy are separate; macro task loss is a declared scalar aggregation, not a common physical unit.

## Evidence and independent checks

`_verify_l173.py` reconstructs every target and every capped context from raw tables without calling preparation/tokenizer functions. It independently fits normalization/vocabulary and checks split/identity/date invariants. Eighty hidden-payload interventions require identical model outputs. Three incorrect learner implementations are rejected. Float64 NumPy Huber/logsumexp calculations match saved task losses; all six selected checkpoints regenerate exact test logits, and all18 validation checkpoints are rescored. Every epoch must have the complete per-task training counts.

The solution notebook embeds all prepared inputs and raw evidence, inlines the actual model and trainer, and executes a separate fresh three-epoch cell/seed0 fit. This validation fit is additional to the six author experiment fits. Full report parity is checked in the matching author environment. Student TODOs remain blank. The optional six-fit notebook cell is off by default; author results are clearly labeled saved evidence.

## Commands (from repository root)

```bash
# Small behavioral checks, counted in the aggregate numerical budget:
.venv/bin/python labs/_budget_l173.py .venv/bin/python labs/_check_l173.py

# Independently reconstruct and verify the delivered experiment:
.venv/bin/python labs/_budget_l173.py .venv/bin/python labs/_verify_l173.py

# Fresh six-fit reproduction into a NEW directory (refuses overwrite):
.venv/bin/python labs/_budget_l173.py .venv/bin/python labs/_run_l173.py --output /tmp/l173-fresh-six
.venv/bin/python labs/_budget_l173.py .venv/bin/python labs/_verify_l173.py --evidence /tmp/l173-fresh-six

# Rebuild notebook/lesson/figures, then execute the portable solution:
.venv/bin/python labs/_figures_l173.py
.venv/bin/python labs/_build_l173.py
.venv/bin/python labs/_budget_l173.py .venv/bin/python labs/_execute_l173.py
.venv/bin/python labs/_delivery_l173.py
.venv/bin/python labs/_checkout_l173.py
```

The raw-to-tensor builder is `_prepare_l173.py`. It authenticates original inputs and materializes the frozen population. Rebuilding input artifacts is not necessary to train or verify; the independent verifier reconstructs them in memory. Never edit the manifest to conceal a mismatch. Numerical runs are charged to `evidence/l173/local-budget.json`, including failures; the watchdog kills the entire process group at the remaining limit. An interrupted active reservation requires reconciliation, not deletion. Do not reset the ledger to bypass the cap.

## Budget, sources and unrun scope

USD0 cloud/API; 3600 aggregate numerical seconds covers preparation, failures, six fits, independent verification and notebook validation. Source retrieval, figure rendering and delivery/browser checks are outside numerical execution. Current actual usage is in the budget ledger; CPU time is a runtime bound, not a cloud charge. If budget expires, missing runs remain INCOMPLETE; do not silently shrink data, epochs or seed count. No cloud fallback or deployment authorized.

Primary readings are pinned HTML snapshots with SHA256 in `sources/l173/source-ledger.json`:

- RT v1 https://arxiv.org/html/2510.06377v1 §3.3 (numeric Huber/boolean BCE and masked-cell mean) and §4.1 (paper protocol).
- KumoRFM-2 v1 https://arxiv.org/html/2604.12596v1 §3 (row, column, FK, cross-sample information axes).

Differences: course multiclass CE, local per-column heads, pooled small encoder, one database, autocomplete-only tasks, course temporal splits, fixed context caps, three epochs and task-balancing intervention. The paper's larger model, corpus, attention, boolean objective, training schedule and forecasting evaluation are not reproduced. RT reports ~2h on8A100 per pretraining run (~16 GPU-hours): approximately USD33.58–39.97 GPU-only at Modal A10040/80GB rates checked2026-10-02, before evaluation/retries/overhead. Pricing source: https://modal.com/pricing . No money spent.

Fresh selected course pretraining COMPLETE. Whole-paper reproduction NOT_RUN. Historical availability and cross-database transfer NOT_ESTABLISHED. Live Colab and deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE. No claim of architectural parity or broad superiority follows from these runs.
