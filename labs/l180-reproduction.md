# L180 · RT-v1 public encoder fine-tuning checkpoint

Approved2026-10-02. Complete selected saved-context/source/cost audit; fresh public-encoder fine-tuning NOT_RUN; practical exit INCOMPLETE. No new model loading, gradient updates, sampling, inference, pretraining or paid work. Whole-paper reproduction NOT_RUN. Learner PENDING_WRITTEN_DEFENSE.

## Target and immutable identities

Named published target: RT-v1 v1 §4.1 / AppendixD supervised fine-tuning, selected rel-f1/driver-dnf task. Source commit8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb; public initialization pretrain_rel-f1_driver-dnf.pt at stanford-star/rt-v1 revision299701dedae451f3dfa40717b831d9dc17c0e4e7; preprocessing revisione8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee. Filename/revision pinned; weight bytes NOT_DOWNLOADED/NOT_CHECKED. Database-held-out training is release provenance, historical lineage NOT_ESTABLISHED. Do not substitute already fine-tuned weights or the local L173 encoder.

Paper: approximately33k steps, globalbatch256, context1024, AdamW LR1e-4, WD0, reported1.5hours on8A100. Original source example: seed0, max_steps32769, batch32 per rank, BFSwidth256, 12blocks,width256,8heads,FF1024,384wideMiniLM vectors. All RT parameters are trainable; precomputed language vectors stay fixed. Boolean masked-target BCE. Exact model and trainer archived under sources/l180/upstream/rt and visible in both notebooks.

The example targets Amazon churn; changing to F1 requires explicit task/load path configuration. save_ckpt_dir defaults toNone. The trainer's comparator uses validation metric only; test is also evaluated/logged. Thus test visibility is documented but is not itself proof of test-selected weights. Source max_eval_steps40 requires actual complete-key coverage verification. Actual RT gradients, full training split and full validation/test traversal NOT_CHECKED. Source inspection does not assert that every post-gate detail is runnable or historically identical. No derivative F1 training implementation is claimed validated.

## Executed audit and saved evidence

The input manifest authenticates all3×702 saved L175contexts, every1024cell slot, metadata, raw results/drivers, label oracle and source example. It verifies the inherited oracle against raw tables, reconstructs the30dayDNF target for every query and compares all complete(driverId,cutoff)key sets. Independent scalar scanning rechecks all2,156,544slots. No fresh model predictions are generated.

Observed385future-dated cells in77contexts, all race schedules;0unmasked query targets;0unfinished-window labels;432050unknown-time slots. Declared event-time policy fails; unknown arrival histories mean outcome leakage/historical availability NOT_ESTABLISHED. Do not treat successful replay as a repaired sampler. The L178RelGNNgradient failure is not assigned toRT. L179is absent; no mastery or missing lesson is invented.

## Cost and stop

USD0newcloud/API. Aggregate1800local numerical seconds include preparation, failed checks, audit, verification and notebook executions. _budget_l180.py keeps every attempt and enforces remaining time; do not reset it. Figure rendering, source retrieval, HTML construction and browser/publication checks are outside the numerical cap. Concurrent numerical work is refused while a reservation is active.

Full-run scenarios:8×1.5×3600×.000583=USD25.185600(A10040GB); at.000694=USD29.980800(A10080GB). Rates checked2026-10-02; scenario follows reported runtime, not measured pilot. Both GPU-only figures exceed USD10beforeCPU/RAM/preparation/retries. No paid pilot/dispatch, model download or shortened run authorized. Original temporal failure independently blocks training. Other unverified execution prerequisites remain visible. The all-in upper bound is unknown; hypothetical admission additionally requires a finite total bound at least as large as the GPU component and within the cap. Future repairs/full execution need a new approved protocol and all-in budget.

## Repository commands

```bash
# Full audit from the embedded saved packet; no cloud/model execution:
.venv/bin/python labs/_budget_l180.py .venv/bin/python labs/_audit_l180.py
.venv/bin/python labs/_budget_l180.py .venv/bin/python labs/_verify_l180.py
# Reproduction admission; expected exit2, BLOCKED before dispatch:
.venv/bin/python labs/_budget_l180.py .venv/bin/python labs/_run_l180.py
# Deterministic artifacts, then execute portable solution from empty directory:
.venv/bin/python labs/_figures_l180.py
.venv/bin/python labs/_build_l180.py
.venv/bin/python labs/_budget_l180.py .venv/bin/python labs/_execute_l180.py
.venv/bin/python labs/_delivery_l180.py
.venv/bin/python labs/_seal_l180.py
.venv/bin/python labs/_checkout_l180.py
```

Dependencies for portable replay: NumPy,pandas,pyarrow,ml_dtypes; no network by default. Student temporal, cost and decision functions control the actual report. Wrong functions/corrupt pins must fail. Hypothetical gate changes never alter saved observations or authorize compute. Practical exit remains INCOMPLETE until a fresh fit is independently verified; written defense remains pending until reviewed.

Clean Git-index Pages checks establish package delivery only. LiveColab/deployment NOT_CHECKED; no push/deployment requested. Public training/historical identity/learner mastery are separate evidence lanes.
