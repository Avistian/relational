# L177 Compute Feasibility Ledger

Approved 2026-10-02. Full selected accounting replay, USD0 new cloud/API, 1800 aggregate local numerical seconds. No fresh model training/inference, checkpoint download or cloud dispatch. This systems lesson has no named paper-result reproduction target; RT's published compute is a source-based price scenario.

## Exact input universe

363 SHA256-pinned input files in evidence/l177/input-manifest.json. 327 matched inherited author artifact seals during preparation. All 300 L176 individual JSON receipts and the two phase receipts (6 pilot, 294 remaining); full grid of two tasks × three arms × five context sizes × ten seeds. Their reported 229050 predictions describe prior inference: no probabilities are re-evaluated here. All six L173 fit records (cell/task × seeds0–2, three epochs) and twelve L174 fit records (freeze/full/adapter/scratch × seeds0–2, ten epochs). Every local ledger attempt in L173–176, all three L175 cloud reservations, its available two worker receipts, failure logs, original timer/dispatch implementations and source snapshots.

No statistical subsampling, changed epochs or reduced seed grid. Every record must match its aggregate receipt, every identity and epoch grid must be complete. Missing accounting evidence remains missing; audit-1 has no worker receipt, L173 no individual fit timers, both CPU lessons no peak-memory measurements, all lessons no recorded researcher-time series. Input hashes establish byte identity, not historical clock truth, source lineage or an invoice.

## Accounting contracts

L173's six-fit body timer is enclosed by its command timer. L174's twelve per-fit timers include evaluation/serialization and are enclosed by its run-command timer. The full local ledgers also contain failed preparation, checks, verification and notebook reruns; do not add those enclosing and nested times.

L176's load_seconds repeats for every task/model/phase group. Deduplicate into12loads; never sum all300copies. Evaluation timers include preprocessing, support fitting, prediction, scoring and serialization as placed in _run_l176.py; they are not pure GPU-kernel latency. Two outer worker timers enclose loading and evaluation. Worker time×one GPU gives worker-body GPU-hours, not full billed GPU-hours. Residual worker time is not cloud startup. Peak GPU bytes are torch.cuda.max_memory_allocated(): allocated tensor peak, not host RAM, total VRAM, device capacity, energy or FLOPs.

Reconstruct L176 reservation as3USD overhead+(600+30+7200+30)seconds×.00028372USD/s=5.2300392USD. Retain its491.470907582worker seconds and.139440125899USD worker estimate as a separate view; do not add the estimate to the reservation. Invoice NOT_ITEMIZED. L175 retains all3×930second CPU allowances at.00006172USD/s, plus1USDbuild and2USDoverhead=3.1721988USD. Invoice NOT_CHECKED. The local ledger's USD0 does not erase past cloud reservations. Reservations are historical planning allowances, not provider-enforced billing guarantees.

Historical L176 forecast:294remaining×8.020519169slowest six-run pilot×3margin+27.568350270pilot-body=7101.666257328seconds. This conservatively included pilot time when checking the7200main timeout. Reconstruct the historical PROCEED decision without using the later observed491seconds to revise it. The3xmargin is a heuristic, not a statistical bound.

## Source-based scenarios

Provider base rates checked2026-10-02; snapshot hash in sources/l177/source-ledger.json. L4 .000222USD/s, physical CPU .0000131/core/s, memory .00000222/GiB/s. CPU and memory additions use2physical cores/16GiB. A10040GB .000583USD/s and80GB .000694USD/s. Exclude free credits and regional/premium multipliers; do not present these rates as a reconciled bill.

RT arXiv2510.06377v1 §4.1 reports approximately2hours pretraining and1.5hours fine-tuning per run on8A100s. Hypothetical40GB rental GPU-only amounts33.5808/25.1856USD and80GB39.9744/29.9808USD. A100 variant and achieved runtime on the hypothetical rental are not established; no one-GPU linear-scaling claim. Even the cheaper scenario exceeds10USD before other costs. These are incomplete cost scenarios sufficient for rejection, never for admission.

RDBLearn2602.18495v1 grounds the conceptual featurization+ICL route only. The L176 DFS+TabICL comparator does not reproduce RDBLearn. PyTorch v2.5.1 CUDA memory source grounds the allocator-boundary explanation. A docs snapshot403failure is retained; the tagged source was successfully pinned instead.

## Session scenarios

One hour=3600s; illustrative extended session=10800s. Serial scheduling adds wall time and active work; no overlap or perfect parallel scaling assumed. L176 forecast/observed-worker scenarios add an explicitly assumed120s startup. Active work1200or3600s and required GPU memory16or32GiB are learner assumptions, not observations. The recorded12.052GiB tensor peak rules out an8GiB device for the unchanged workload but does not establish that16GiB fits. Unknown measurements cannot yield FEASIBLE_SCENARIO; any known scientific/budget/memory/time violation can reject first. Scientific stop has precedence. The calculator is read-only and is not a paid-run operator.

## Execute the full audit

From the repository root, use existing Python environment:

```bash
.venv/bin/python labs/_budget_l177.py .venv/bin/python labs/_check_l177.py
.venv/bin/python labs/_budget_l177.py .venv/bin/python labs/_audit_l177.py
.venv/bin/python labs/_budget_l177.py .venv/bin/python labs/_verify_l177.py
.venv/bin/python labs/_figures_l177.py
.venv/bin/python labs/_build_l177.py
.venv/bin/python labs/_budget_l177.py .venv/bin/python labs/_execute_l177.py
.venv/bin/python labs/_delivery_l177.py
```

Default notebooks embed all363inputs and exact audit functions. Only Python standard library needed for replay. Student TODOs directly control the full audit. The executed solution must reproduce the repository report exactly from an empty directory. Source code for original training/inference timers is included for inspection only. Fresh model operators remain in prior lessons and require separately scoped budgets; this lesson does not dispatch them.

Independent verification uses SQL timing/group queries, Decimal arithmetic, full18fit coverage,200random reservation cases,200forecast cases, three rejected wrong learner implementations and four rejected corrupt evidence packets. Browser tests cover desktop/mobile scenarios, keyboard/reset, no-JS/print, portable diagrams, manifest galleries and local links. Source parity and deterministic generation are separate checks. Clean Git-index Pages construction does not constitute deployment or live Colab.

## Evidence boundaries

COMPLETE_SELECTED_ACCOUNTING_REPLAY. Fresh training/inference and whole-paper reproduction NOT_RUN. Researcher time NOT_MEASURED; cross-hardware efficiency/historical lineage NOT_ESTABLISHED. Invoice reconciliation/liveColab/deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE. No mastery record or exit gate is advanced by author preparation. No push/deployment requested.
