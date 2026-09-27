# Lesson 128: task taxonomy and historical-label DNF reproduction

## Target and executed result

RelBench arXiv:2407.20060v1 Table6, F1 driver-dnf RDL. Five fresh seeds0–4,
ten full epochs each, all11411 training queries per epoch. Validation
AUROC71.87810±1.29392 percentage points; test71.67081±1.17596. Paper targets
71.36±1.54 /72.62±.27. Both means CLOSE under the predeclared2pp descriptive
tolerance. Sample seed SD is not a confidence interval. Whole-paper and
historical identity NOT_ESTABLISHED.

## Critical data reconstruction

The first pilot failed before training: relbench1.1.0 expected task zip
58553e0ecebff60e9f8c12202ae2d1109b206b6f8a1ec0589af2540ac2982178, but the
current endpoint serves bd562529a3c0016363d5cae247979fd3712d77ba948574f135dab88c036d3e2d.
Do not replace the registry checksum and silently claim historical data.

Upstream c348273a8e66 flipped the task labels. Its parent
 e416c3d208cb8a6d8bd31a68c46ca9090ecb334a contains the historical CASE expression:
positive means no result with statusId !=1 in the future window. The historical
zip hash in that parent was948df149bc36537cb14d7624c46db886e75d9c793a48c35fde49ac9f41220655;
those bytes are unavailable here. We pin the current archive, transform
1-current_label, and verify every query key and label against unmodified
pre-flip SQL plus independent SQLite raw-event aggregation. Counts1365/125/207
match paper Table13. Original row order and byte identity remain unestablished.
This is a selected experiment with reconstructed historical labels, not exact
historical-archive replay. The current task's positive semantics differ.

Released eligibility has no upper bound in its prior-year subquery, and task
membership conditions on future participation. A separate past-only365-day audit
finds1022/27/26 queries without observed prior-year results. We preserve released
membership rather than silently correcting the population. Keep sub-day timestamps:
dropping race times to calendar days changes strict lower-window membership.

## Visible implementation and protocol

- relkit/taxonomy_l128.py: live task contract, pairwise tied AUROC and macro MAP.
- relkit/classification_l128.py: full BCE-with-logits trainer, sigmoid evaluation,
  first maximum validation AUROC checkpoint, independent final scoring.
- relkit/rdl_l117.py: visible full table encoder/time encoder/GraphSAGE/model/graph.
- relkit/historical_task_l128.py: hash-checked historical task reconstruction.
- relkit/batch_audit_l123.py: every-batch root-cutoff/edge/query identity checks.
- sources/l128/: classification/recommendation API, metrics, historical SQL,
  current archive, flip commit and provenance manifest. sources/l117/: full
  shared upstream model/graph/primitive sources and licenses.

Upstream model/trainer commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639.
Runtime: modal/l128_repro.py and requirements-l117-runtime.txt. T4; torch2.5.1,
pyg-lib0.4.0 CUDA12.4; seed42 before preprocessing; declared train seeds0–4.
Model width128, two sum-GraphSAGE layers, per-table four-block Frame ResNets,
relative-time encoding, batch512, uniform fanouts[128,64], Adam.005, ten epochs.
GloVe model revision pinned in sources/l117/text_model.json.

Other deviations: source fanouts[128,64] versus paper table128; feature statistics
fit through test database cap rather than train-only; missing actual arrival,
static creation and mutable-feature histories; historical seed/environment
identities unknown. Final validation scores can differ from selection scores
because evaluation resamples neighborhoods. Test never selects checkpoints.

## Commands

From repository root, with existing `.venv`:

```bash
.venv/bin/python labs/_check_l128.py
.venv/bin/python labs/_mutation_l128.py
.venv/bin/python labs/_source_check_l128.py
.venv/bin/python labs/_data_l128.py
.venv/bin/python labs/_historical_l128.py
.venv/bin/python labs/_audit_l128.py
.venv/bin/python labs/_analyze_l128.py
.venv/bin/python labs/_figures_l128.py
.venv/bin/python labs/_build_l128.py
.venv/bin/python labs/_execute_l128.py
.venv/bin/python labs/_delivery_l128.py
.venv/bin/python labs/_verify_l128.py
```

Completed author launch sequence (budget ledger prevents accidental repeat):

```bash
.venv/bin/modal run modal/l128_repro.py --mode pilot
.venv/bin/python labs/_collect_l128.py --mode pilot
.venv/bin/python labs/_pilot_check_l128.py
.venv/bin/modal run modal/l128_repro.py --mode paper
.venv/bin/python labs/_collect_l128.py --mode paper
```

The first command initially failed on the old registry hash. Its one worker
reservation remains recorded as failed-pilot-download. After source/label audit,
a second pilot used the explicit recovery function. No training result was
substituted for that failed attempt. No automatic retries.

For fresh training outside the consumed author budget, use the exact Modal image
environment and a new output directory (local command has no monetary guard):

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l128.py --seed "$seed" --epochs 10 \
    --output "labs/results/l128/my-new-run/seed-$seed"
done
```

The notebook embeds the complete model/trainer, all task-table rows and all6340
final predictions. Default run independently replays author evidence. Its
RUN_FULL_REPRODUCTION=False gate can run five new full fits in the pinned GPU
environment; it has no dollar cap. Live Colab NOT_CHECKED. Archived author weights
are local in results/l128; fresh training regenerates them. The analysis requires
those weights for hash validation. Publication staging excludes checkpoint files.

## Compute accounting

Aggregate authorized capUSD10 includes seeds, pilot, retries and validation.
Current verified rate: T4.000164 +2CPU×.0000131 +16GiB×.00000222 =USD.00022572/s.
Eight one-hour worker slots bound resources atUSD6.500736; USD3.499264 overhead
reserve. Seven slots consumed (one failed download, one successful pilot, five
fits), one unused. Successful function-body resource estimateUSD.07802214;
failed pilot conservatively bounded by its entireUSD.812592 slot. Setup/storage/
unitemized charges are not an invoice total. No further cloud work planned.
https://modal.com/pricing
Pilot https://modal.com/apps/pszar92/main/ap-k3zmbl2QIeKJiH98JTN1dD
Full https://modal.com/apps/pszar92/main/ap-V3BiZojBilUI9QxGKO0gCa

## Verification and boundaries

All6340 final probabilities independently rescored; original-model maximum logit
difference9.536743e-7. All605190 sampled query occurrences audited. All12679
historical task keys/labels match original SQL. Six semantic mutants rejected.
Execution, browser and copied-Pages checks have separate JSON reports; consult
those actual outcomes. No trained recommendation or autoregressive result is
claimed. Whole-paper identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.
Live Colab/deployment NOT_CHECKED. No publication requested.
