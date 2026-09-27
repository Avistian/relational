# Lesson 127: RelBench v1 selected-experiment reproduction

## Target and executed outcome

Robinson et al., arXiv:2407.20060v1, Table7 F1 `driver-position` RDL.
**COMPLETE selected released-protocol replay:** five fresh full-data ten-epoch
runs, seeds0–4, following a separate one-epoch seed100 pilot. Validation MAE
**3.178173712 ± .022775942**; test **3.967165273 ± .093905638** (mean ± sample SD).
Paper targets3.193±.024 /4.022±.119. Both means are **CLOSE** under the predeclared
.2 MAE descriptive tolerance. This is not an equivalence test.

All6,295final predictions independently rescored; original-model same-batch
replay max raw-output discrepancy2.861023e-6. Every actual sampled batch audited:
403,895query occurrences across training, epoch validation and final evaluation.
Original node identities/times, root cutoff and original edge/query isolation pass.

Whole-paper and historical identity **NOT_ESTABLISHED**. Other29 tasks **NOT_RUN**
in this lesson. The manual-feature user study is not rerun. Author execution does
not establish learner mastery: **PENDING_WRITTEN_DEFENSE**.

## Protocol and deviation ledger

| Axis | Executed contract / boundary |
|---|---|
| Paper | July2024 arXiv:2407.20060v1, Table7; pinned HTML bytes |
| Source | Released commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; actual paper training commit unknown |
| DB | Full rel-f1 through2010-01-01, nine tables,74,063rows,338,842directed edges |
| DB archive SHA256 | ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task archive SHA256 | 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Split | 7,453train /499validation /760test queries, released IDs/times/labels |
| Task cohort | Released future-participation-conditioned driver-position; L124's past-only intervention is not applied |
| Features | Released stype inference seed42, pinned GloVe sentence encoder, separate four-block Frame ResNets to128 |
| Architecture | Relative-time encoding, two typed sum-GraphSAGE layers, scalar seed head |
| Training | Adam .005, batch512, mean L1, ten complete epochs, fresh seeds0–4 |
| Sampling | Uniform [128,64] fanouts, root-specific timestamp cutoff; paper table says128 |
| Preprocessing | Released statistics fit to DB through test cutoff; not train-only |
| Selection | First strict improvement in epoch validation MAE; test never chooses checkpoint |
| Inference | Training-label2nd/98th percentile clipping; all final validation/test rows |
| Validation resampling | Final selected-model validation score may differ from selection trace due to neighborhood RNG |
| Temporal limits | Event times observed; ingestion, static creation and mutable-feature histories absent |
| Historical runtime/seeds | Declared current pinned runtime and seeds; historical identities not recovered |
| Scope | One complete selected released-protocol experiment; not every paper result |

Training runtime: Python3.11, Torch2.5.1+cu124, pyg-lib0.4.0+pt25cu124, and exact
requirements in `requirements-l117-runtime.txt`. Pinned text-model revision in
`sources/l117/text_model.json`. Actual worker packages are in each `audit.json`.
Current CPU source checks use a separate newer local runtime, recorded in their
report. Source bytes and AST/output/gradient/update parity are independent checks.

## Exact commands

From repository root in the existing author environment:

```bash
.venv/bin/python labs/_check_l127.py
.venv/bin/python labs/_mutation_l127.py
.venv/bin/python labs/_source_check_l127.py
.venv/bin/modal run --detach modal/l127_repro.py --mode pilot
.venv/bin/python labs/_collect_l127.py --mode pilot
.venv/bin/python labs/_pilot_check_l127.py
.venv/bin/modal run --detach modal/l127_repro.py --mode paper
.venv/bin/python labs/_collect_l127.py --mode paper
.venv/bin/python labs/_analyze_l127.py
.venv/bin/python labs/_audit_l127.py
.venv/bin/python labs/_figures_l127.py
.venv/bin/python labs/_build_l127.py
.venv/bin/python labs/_execute_l127.py
.venv/bin/python labs/_delivery_l127.py
.venv/bin/python labs/_verify_l127.py
```

These record the completed initial author run. Paid dispatch refuses an already
reserved mode and completed outputs. A new independent remote experiment requires
a new volume and source-hashed budget ledger; never erase prior reservations to
rerun it. No automatic retries. The runner downloads checksum-pinned data and the
pinned text model, so training does not require the author's cached weights.

For a separate fresh local run, provision the exact GPU environment specified in
`modal/l127_repro.py`, including native pyg-lib, then use a NEW output directory:

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l127.py --seed "$seed" --epochs 10 \
    --output "labs/results/l127/my-new-run/seed-$seed"
done
```

Local execution has no monetary guard. Default notebook execution is a replay of
complete embedded author predictions/epoch records plus reconstruction of a real
query's graph. It does not train. The explicit five-seed training gate requires the
pinned GPU environment and is OFF by default. Live Colab **NOT_CHECKED**.

Optional source refresh `labs/_prepare_l127.py` fetches paper HTML and hashes local
load-bearing sources. HTML rendering bytes may change even for a versioned paper;
retain the original recorded artifact if investigating provenance.

## Compute accounting

Approved aggregate capUSD10, including pilot, fits, validation and retries.
2026-09-27 verified Modal rates: T4 .000164 +2CPU×.0000131 +16GiB×.00000222 =
USD.00022572/s. Eight one-hour reservations cap worker resources atUSD6.500736;
USD3.499264reserved for overhead/storage/setup. Six reservations used, no retries.
Pilot conservative projectionUSD.23734672. Measured pilot+five-fit function-body
resource estimate **USD.06339861**; setup/storage/unitemized billing excluded.
This is not an invoice total. Validation ran locally.

Pilot: https://modal.com/apps/pszar92/main/ap-SzVnGvCG33wXekeXadrD0T
Full fits: https://modal.com/apps/pszar92/main/ap-uwgYTBhJnzTLv6ymKbtbFV
Pricing: https://modal.com/pricing

## Artifacts and verification boundaries

- `evidence/l127/paper/seed-*/`: complete traces, final predictions, runtime/source/data audits, completion identities and per-batch audit totals.
- `results/l127/paper/seed-*/selected.pt`: local selected weights, intentionally not copied into Pages. Fresh training regenerates weights.
- `evidence/l127/summary.json`: full selected-experiment aggregation and deviations.
- `evidence/l127/trace.json`: real driver10 query at2004-07-05; 99nodes/245edges, typed shapes. Reconstructed from the full inherited L123 graph, not a saved training batch or neural activation dump.
- `_sources_l127.json`, `_upstream_l127.json`: source hashes, pinned paper and freshly verified original source URLs.
- `relkit/benchmark_l127.py`: learner contract/selection/aggregation functions; `relkit/rdl_l117.py`: visible full released model/trainer; `relkit/batch_audit_l123.py`: visible batch audit.
- `_check_l127_results.json`, `_mutation_l127_results.json`: behavioral checks and six rejected experiment-accounting faults.
- `_source_check_l127_results.json`: source AST and current-CPU outputs, gradients and Adam update parity.
- `_audit_l127_results.json`: independent archive query identities and SQL graph census.
- `_execution_l127_results.json`, `_delivery_l127_results.json`, `_verify_l127_results.json`: actual notebook, browser/copied-Pages and consistency results.

Notebook reconstruction embeds the complete graph with hashes; raw database and
full GPU training are available through the fresh-run lane. Student notebooks have
three live TODO functions; the solution's same code is executed in an empty working
directory. Full reproduction, replay, structural trace and synthetic fault checks
are explicitly distinguished. Local delivery does not establish deployment or live
Colab. No publication requested; both **NOT_CHECKED**.
