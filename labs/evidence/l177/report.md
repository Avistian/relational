# L177 Compute Feasibility Ledger

**Worker time:** the elapsed interval measured inside a process. **GPU-hours:** sum of GPU count × elapsed hours across jobs. **Researcher time:** active preparation, debugging and review; not inferable from model timers.

**Nested clocks:** use the enclosing timer for its total. Inner timers explain its contents. Deduplicate repeated metadata by the event identity. L176 has 300 evaluation records but 12 distinct task/model/phase load events.

**Three money columns:** worker estimate = measured seconds × stated resource rate; reservation = every attempt's timeout/lifecycle allowance × rate + overhead; invoice = provider-billed amount. Never add a subset estimate to an enclosing reservation. Never release a failed attempt's reservation without reconciliation.

**Forecast:** remaining count × slowest relevant pilot × declared margin + fixed work. This is a heuristic, not a confidence bound. Freeze the full grid before timing; do not drop seeds or change hardware silently.

**Memory:** host RAM, tensor allocations, reserved GPU allocator memory and total device usage are different quantities. L176's 12.052 GiB allocation peak does not prove the total required capacity.

**Admission:** reject known scientific, cost, memory or time violations. If an essential measurement is missing, return INCOMPLETE_MEASUREMENT. Serial-session scenarios conservatively add machine and active time. FEASIBLE_SCENARIO means assumptions fit, not that a future run is guaranteed.

| Evidence | Work completed | Recorded time | What it supports |
|---|---|---|---|
| L173 · local CPU | Six small-model pretraining fits | 51.837 s group; 188.416 s all local commands | Course masked-cell objective; no per-fit timer |
| L174 · local CPU | Twelve adaptation/control fits | 19.566 s sum of fits; 107.588 s all local commands | Four adaptation policies on the course tasks |
| L176 · L4 GPU | All 300 checkpoint evaluations | 491.471 s worker bodies | Two tasks, three models, five context sizes, ten seeds |
| L175 · cloud CPU | Three reserved audit attempts; stopped before GPU inference | 7.641 s in two receipts; first attempt missing | Failed/stopped work still consumes reservations |


**Worked money:** L176 worker estimate $0.139440; reservation $5.230039; invoice NOT_ITEMIZED. L175 reserves $3.172199 including three attempts, overhead and build, but GPU inference is NOT_RUN after a temporal failure.

**Source-based scenarios:** RT v1 reports about 16 pretraining or 12 fine-tuning GPU-hours per run. At the checked A100 40 GB base rate, GPUs alone cost about $33.58/$25.19. This is a hypothetical rental calculation; it does not reproduce the model or establish its bill.

**Executed:** complete selected accounting replay: **300 inference receipts, 18 fit records, three cloud attempt reservations**. No new model run or cloud/API spend. Fresh training, fresh inference and whole-paper reproduction are `NOT_RUN`; learner status is `PENDING_WRITTEN_DEFENSE`. Invoice reconciliation and live Colab are `NOT_CHECKED`.

[Lesson](../lessons/0177-compute-budget-realism.html) · [Notebook](../labs/0177-compute-budget-realism.ipynb) · [Full report](../labs/evidence/l177/report.json) · [Protocol](../labs/l177-reproduction.md) · [Modal pricing](https://modal.com/pricing) · [RT v1 §4.1](https://arxiv.org/html/2510.06377v1#S4.SS1) · [PyTorch memory source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py).

