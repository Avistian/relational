**Your win:** write a feasibility ledger that can reject an unaffordable experiment before it starts—and explain exactly what an affordable run would establish. Allow 15 minutes for the lesson, then use the rest of your session for the notebook.

In [lesson 176](0176-few-shot-icl-evaluation.html), you measured what more labeled context buys. The missing question is whether you can afford the **whole protocol**: every model, seed and context, plus preparation, verification and failed attempts. That question connects your [mission](../MISSION.md) to a feasible Year 6 research project.

**Prerequisites:** distinguish pretraining from adaptation ([173](0173-multi-task-pretraining.html), [174](0174-fine-tuning-protocol.html)); recall why the cheap RT audit stopped in [175](0175-zero-shot-evaluation.html). You only need multiplication, units and the idea of a timer inside another timer.

[[WARMUP]]

## 1 · What work are you buying?

Pretraining changes shared weights across many tasks. Fine-tuning changes parameters for a selected target setting. In-context prediction uses labeled support with fixed pretrained weights, but still pays for relational features, preprocessing, loading and prediction. [RDBLearn §3](https://arxiv.org/html/2602.18495v1#S3) illustrates the featurization-plus-ICL route; the L176 TabICL comparator is a different pipeline and does not reproduce RDBLearn.

[[FIG:routes]]

A released checkpoint lets you reuse someone else's pretraining investment. It does not make that investment zero. Likewise, our small CPU pretraining exercise is affordable because of its explicit model and data scope; it does not establish the cost of pretraining a foundation model. **“Pretraining ≫ fine-tuning ≫ ICL” is a planning intuition, not a universal measured ordering.** Compare complete protocols on the same task before claiming an efficiency advantage.

## 2 · Put boundaries around every number

[[RESULTS]]

These are different workloads and machines. The table describes achieved capabilities; its seconds are **not a speed leaderboard**. L173 lacks individual fit timers. L174 times each fit including evaluation and serialization. The local-command totals also include preparation, audits, failures and notebook validation; their enclosing timers must not be added again to the fit totals. See the [complete ledger](../labs/evidence/l177/report.json) and [timer source](../labs/_run_l176.py).

[[PREDICT]]

For L176, the worker body took **491.471 seconds** on one L4. Within it, the 300 evaluation timers total **474.434 seconds**. The model-load field is repeated in each receipt: adding all copies gives **23.896 seconds**, but there were only **12 distinct task/model/phase loads**, totaling **0.939 seconds**. The remaining **16.098 seconds** is residual worker work. It is not measured cloud startup time.

[[FIG:accounting]]

The posted resource-rate calculation is:

```text
L4 + 2 physical CPU cores + 16 GiB host RAM
= 0.000222 + 2 × 0.0000131 + 16 × 0.00000222
= $0.00028372 per second

Worker estimate = 491.470908 × rate = $0.139440
Reservation = $3 overhead + (600 + 30 + 7,200 + 30) × rate
            = $5.230039
```

The reservation holds money for timeouts and overhead, including unsuccessful attempts. Do not add the worker estimate to it: that estimate describes work already covered by the reservation. An invoice is a third quantity. L176's remains **NOT_ITEMIZED**. [Modal pricing](https://modal.com/pricing) also bills loading and idle container time, and meters CPU/RAM at the greater of requested or used resources. These are historical planning reservations, not provider-enforced invoice ceilings.

**Memory is another boundary.** The largest L176 PyTorch allocation was **12.052 GiB**. That is allocated tensor memory on the device, not total GPU usage or 16 GiB host RAM. Cached allocations, libraries and other device use can differ. A successful run on the configured L4 is evidence about that run; it does not prove an arbitrary 12 GiB device will fit. See [the pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py).

[[FIG:context]]

## 3 · Forecast before you know the result

The historical L176 pilot used six runs at k=1,024. Its slowest evaluation took **8.020519 seconds**. Before dispatching the other 294 runs, the declared heuristic was:

```text
294 × slowest pilot × 3 + pilot worker duration
= 294 × 8.020519 × 3 + 27.568350
= 7,101.666 seconds ≈ 118.4 minutes
```

This fit the 7,200-second main timeout and the $8 planned spending stop, leaving $2 of the $10 cap protected. The threefold margin is a heuristic, not a confidence interval. The pilot cannot guarantee runtime at unseen context sizes, on another GPU or with new feature construction. Retain the original forecast even when execution turns out much faster.

[[CODE]]

**Parallelism changes two clocks.** Four one-hour jobs on four GPUs can take roughly one elapsed hour while consuming four GPU-hours. They may also require more memory and startup costs. GPU-hours = the sum of GPU count × each job's elapsed hours. Do not use elapsed time alone as a price estimate, or assume that multi-GPU runtime scales linearly onto one GPU.

## 4 · One hour or an extended session?

Human preparation time was not recorded for these experiments. A machine finishing in eight minutes does not establish that a researcher can reproduce the project in eight minutes. Treat one hour as 3,600 seconds and an illustrative extended session as three hours; separate observed evidence from assumptions about your own time.

[[EXPLORER]]

**Try this:** select the L176 forecast, three hours, 20 minutes of active work, and a **hypothetical 16 GiB requirement** on a 24 GiB GPU. The serial scenario fits at about 140.4 minutes, including an assumed two minutes of additional startup. Switch to one hour: it fails the session gate. Restore unknown memory or active work: it cannot earn a complete admission. The calculator does not dispatch a job.

For a baseline session, work through the complete saved ledger, implement one contract, and write your decision. With more time, complete the notebook and investigate a missing measurement. A fresh run still needs its own complete protocol and reservations. L178 will need a matched-task comparison, including feature construction, tuning, evaluation and retries for each paradigm.

## 5 · Some plans must stop

[RT v1 §4.1](https://arxiv.org/html/2510.06377v1#S4.SS1) reports approximately two hours of pretraining or 1.5 hours of fine-tuning on eight A100s per run. That is **16 or 12 GPU-hours**. At the checked A100 40 GB base rate, a hypothetical rental costs **$33.58 or $25.19 for GPUs alone**; the 80 GB scenario is $39.97 or $29.98. These are source-based price scenarios, not measured bills or a promise that either variant reproduces the runtime. Even the cheaper scenario exceeds our $10 cap before CPU, memory, preparation, seeds or retries. Full runs remain **NOT_RUN**.

Money is not the only stop condition. L175 reserved **$3.172199** across its three CPU audit attempts plus overhead/build allowances; one attempt has no worker-time receipt. Its temporal gate failed, so GPU inference remained **NOT_RUN**. A larger budget or longer session cannot repair that scientific failure.

## Lab · Rebuild the complete feasibility ledger

[Student notebook](../labs/0177-compute-budget-realism.ipynb) · [Executed solution](../labs/html/0177-compute-budget-realism.html) · [Reproduction protocol](../labs/l177-reproduction.md) · [Quick reference](../reference/compute-budget-realism.html)

Implement three live functions: retain every unique attempt reservation; forecast a complete remaining grid; and admit a scenario only when its independent gates pass. They drive the full audit of **300 inference receipts, 18 fit records, and three cloud attempt reservations**. The self-contained packet contains the original accounting records and timer source. No cloud account or checkpoint download is needed.

**Exit:** defend a one-hour and a three-hour plan. Include scope, units, preparation, all seeds/retries, memory evidence, explicit human-time assumptions, protected margin and a scientific stop. Explain why $0.139440, $5.230039 and the missing invoice answer different questions. Revisit your decision after your next timed practice session.

[[TEACHBACK]]

[[STATUS]]

**Read next:** [Modal's resource pricing and billing FAQ](https://modal.com/pricing), then trace the timers in the supplied notebook. Ask the teacher about any unclear unit, boundary or budget decision; bring your written ledger for feedback.

**Next lesson:** [178 · Fair model comparison](0178-fair-model-comparison.html) asks whether equally budgeted methods actually receive comparable tasks and information.
