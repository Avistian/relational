# L174 F1 Temporal Adaptation

Approved2026-10-02. Complete selected **course** experiment; twelve fresh adaptation fits. Whole-paper reproduction NOT_RUN. This is retrospective same-database, same-task autocomplete, not unseen-task forecasting or cross-database transfer.

## Frozen protocol and lineage

All21L173 targets with unchanged raw-to-cell mapping, original pre2005 normalization and category vocabulary, target erasure and4row/4dated-FK context slots. Context parents have event dates no later than their owner; absent arrival histories and same-row outcome proxies remain limitations. Numeric Huber delta1 and multiclass cross-entropy including UNKNOWN; no re-fitting on later periods. Authentication includes original raw inputs and selected L173 cell-arm checkpoint hashes. Source checkpoints were selected solely on2005validation, independently for seeds0/1/2; source arm fixed by the approved design.

2006adaptation:6,429targets;2007validation:6,074;2008+test:106,757. All21tasks occur in each split. Full source packet contains370,024target identities;No rows are silently sampled: excluded pre2006 targets are exhaustively counted in coverage.json. L173 already evaluated later periods, so this is not an untouched confirmatory test. Source checkpoints learned these tasks before2005; no unseen-task claim.

Four arms: freeze (only original heads train); full (all original parameters train); adapter (encoder fixed, original heads and new adapter train); scratch (random original model, all train). Original head weights retained except scratch. Adapter is h+U ReLU(Dh+b)+c after the32-wide shared MLP; D random, U/czero. Dimensions32→8→32.552new parameters; existing heads9,834; encoder18,592. Total/trainable: freeze28,426/9,834; full28,426/28,426; adapter28,978/10,386; scratch28,426/28,426.

Seeds0/1/2, ten complete epochs, batch1024,70updates/fit, Adam learning rate.001, default betas(.9,.999)/epsilon1e-8, no weight decay. Equal-task importance weight N/(21*N_task) from2006training counts only; never renormalize within a minibatch. All four arms share seed-specific training permutations. CPUone thread, deterministic Torch algorithms. Shared hyperparameters are a fixed-schedule comparison, not tuned-optimal or equal-compute arms. Record actual runtime separately.

Select earliest minimum2007validation macro loss among ten checkpoints; evaluate test only afterward. Save initial states,120epoch checkpoints, full1,281,084fitted-model predictions and427,028control predictions by cell ID. Cell IDs map uniquely to table/row/column/date under the authenticated original packet. Report all tasks' losses plus raw MAE or accuracy separately. Mean±sampleSD across seeds describes seed variation, not independent databases or a confidence interval.

Controls: all three unchanged source checkpoints and a2006training-only numeric-mean/Laplace(+1)categorical-frequency baseline. The scratch control uses no inherited weights, while keeping the common pre2005 preprocessing. Thus it tests weight initialization under the approved shared input representation, not complete independence from earlier data.

## Commands from repository root

```bash
# Complete NEW twelve-fit experiment (existing destinations refused):
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_run_l174.py --output /tmp/l174-fresh-twelve
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_verify_l174.py --evidence /tmp/l174-fresh-twelve

# Independently check delivered evidence and behavioral contracts:
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_verify_l174.py
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_check_l174.py
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_check_operator_l174.py

# Rebuild narrative and portable notebooks; execute a separate full adapter fit:
.venv/bin/python labs/_figures_l174.py
.venv/bin/python labs/_build_l174.py
.venv/bin/python labs/_budget_l174.py .venv/bin/python labs/_execute_l174.py
.venv/bin/python labs/_delivery_l174.py
.venv/bin/python labs/_checkout_l174.py
```

`_prepare_l174.py` authenticates the raw inputs and L173 artifact seal, copies the prepared population and selected checkpoints, and records complete provenance. The delivered packet is ready to use; preparation need not be repeated. `_verify_l174.py` calls the original independent raw-table/context reconstruction and uses its own float64 NumPy metric oracle. It replays all120validation checkpoints and every selected test checkpoint, checks paired orders and complete per-task epoch counts, and compares every frozen tensor at every epoch. Three wrong learner functions must fail. Exact saved logits must replay in the author environment; notebook fresh-fit parity excludes wall-clock timing.

## Sources, deviations and costs

Primary sources pinned by SHA256 in sources/l174/source-ledger.json: Houlsby et al.(2019), RTv1, Modal pricing snapshot2026-10-02. Houlsby supports fixed-backbone adapter transfer; our single post-MLP adapter, local heads, F1 corpus and schedule are not their BERT/GLUE experiment. RTv1Figure3/AppendixD supplies the published relational adaptation comparison; this course does not reproduce RTarchitecture, unseen-task setup, full dataset portfolio or learning curves. Published schedule:1.5h×8A100≈12GPU-hours per fine-tuning run, aboutUSD25.19–29.98GPU-only at snapshot rates, above theUSD10aggregate ceiling. RTpretraining and full-paper reproduction NOT_RUN. No claim of paper-score parity.

USD0new cloud/API;3,600aggregate numerical seconds across preparation, test failures, all12fits, verification, operator checks and fresh notebook execution. Planning data-count probes are recorded too. Ledger:evidence/l174/local-budget.json. Figure rendering/source retrieval/delivery checks excluded from numerical runtime. The subprocess watchdog kills descendants at the remaining cutoff; interrupted reservations require reconciliation. Do not reset the ledger, silently shorten the experiment, or use a paid fallback. Missing runs become INCOMPLETE.

Measured results: evidence/l174/runs/report.md and report.json. Existing L173 training is inherited, not freshly reproduced pretraining. LiveColab and deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE.
