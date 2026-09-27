# Lesson130 · Q1 end-to-end RDL checkpoint

## Selected experiment and current evidence

Fresh complete released-protocol replay of Robinson et al., arXiv2407.20060v1,
Table7, rel-f1/driver-position basic RDL. Five new seeds0–4, ten complete epochs
per seed, after a separate one-epoch seed100 pilot. Validation MAE
3.164115849 ±0.038286778; test4.070920721 ±0.075000232 (mean ±sample seed SD).
Paper means3.193/4.022: both CLOSE under predeclared absolute mean tolerance0.2.
This is descriptive closeness, not an equivalence test.

All6,295 final predictions independently scored and compared to the original
model. All403,895 sampled training/validation/test query occurrences audited.
Independent SQL graph census and released archive IDs/timestamps/labels checked.
Checkpoint functions execute inside each fresh worker, including scoring
reversed prediction records by exact keys. Evidence: evidence/l130/summary.json,
checkpoint-audit.json in every seed, _audit_l130_results.json.

L129 manual-FE comparison is explicitly REUSED evidence. All6,295 identical
query keys and targets rechecked; test mean3.948916837 versus fresh RDL4.070920721.
FE minus RDL =−0.122003884. Different preprocessing/search/computation budgets;
no claim of universal superiority or repeated human study. Saved artifact hashes
and metric recomputation: evidence/l130/comparison.json.

## Protocol and deviations

| Axis | Frozen contract or boundary |
|---|---|
| Paper | arXiv2407.20060v1 Table7, HTML bytes pinned in sources/l130/paper.html |
| Source | Released9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; exact historical training commit unknown |
| Database | Full9-table rel-f1 through2010-01-01,74,063rows,338,842directed edges |
| DB SHA256 | ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task SHA256 | 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Queries | 7,453train /499validation /760test; (driverId,nanosecond cutoff) keys |
| Cohort | Released future-participation condition; not an operational past-only cohort |
| Row encoders | Four-block per-table Frame ResNets,128channels, pinned GloVe model/revision |
| Graph computation | Relative-time additions, two typed sum-GraphSAGE layers, scalar root head |
| Training | Adam .005, batch512, L1, ten full epochs, seeds0–4 |
| Sampling | Uniform[128,64], own root cutoff at every hop; paper table says128 |
| Preprocessing | Type proposal seed42; statistics through database test cap, not train-only |
| Selection | First strict validation MAE improvement; test never chooses checkpoint |
| Scoring | Train-label2nd/98th percentile clipping; official evaluator plus keyed independent MAE |
| Validation | Final resampling can differ from checkpoint-selection MAE |
| Temporal limit | Missing arrival/static-creation/mutable-feature histories are not proved safe |
| Runtime | Python3.11/Torch2.5.1+cu124, PyG2.6.1, Frame0.2.3, RelBench1.1.0; all pins in requirements-l117-runtime.txt and modal/l130_repro.py |
| Historical identity | NOT_ESTABLISHED; original package/seed/random-state identity unavailable |
| Whole paper | NOT_ESTABLISHED; other29tasks NOT_RUN in this lesson |
| Learner | PENDING_WRITTEN_DEFENSE; use l130-defense.md |

Original source, primitive source and licenses remain visible under sources/l117.
_sources_l130.json pins all inherited load-bearing files. CPU AST/output/gradient/
Adam-update parity is distinct from full-data original-model GPU replay.

## Exact author commands

From the repository root, using the established author environment:

```bash
.venv/bin/python labs/_check_l130.py
.venv/bin/python labs/_mutation_l130.py
.venv/bin/python labs/_source_check_l130.py
.venv/bin/modal run --detach modal/l130_repro.py --mode pilot
.venv/bin/python labs/_collect_l130.py --mode pilot
.venv/bin/python labs/_pilot_check_l130.py
.venv/bin/modal run --detach modal/l130_repro.py --mode paper
.venv/bin/python labs/_collect_l130.py --mode paper
.venv/bin/python labs/_analyze_l130.py
.venv/bin/python labs/_audit_l130.py
.venv/bin/python labs/_compare_l130.py
.venv/bin/python labs/_figures_l130.py
.venv/bin/python labs/_build_l130.py
.venv/bin/python labs/_execute_l130.py
.venv/bin/python labs/_delivery_l130.py
.venv/bin/python labs/_verify_l130.py
```

Paid dispatch is single-use: it refuses already reserved modes, mutated source
and completed outputs. To repeat independently use a new named volume and budget
ledger; do not erase reservations or overwrite this evidence. Downloads verify
archive hashes and a pinned text-model revision; no author's trained weights are
required to run the experiment. _prepare_l130.py refreshes source/paper provenance
and is not needed merely to replay existing evidence.

For fresh local training, install the exact documented GPU/native pyg-lib runtime,
choose a NEW output directory, then run:

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l130.py --seed "$seed" --epochs 10 \
    --output "labs/results/l130/my-new-run/seed-$seed"
done
```

The local operator has no monetary guard. Default notebook execution uses three
live checkpoint functions on complete embedded author evidence, rechecks the
manual-FE comparison, and reconstructs the real99-node/245-edge query context.
It does not train by default. The explicit RUN_FULL_REPRODUCTION gate trains all
five full runs from downloads, uses the visible model/trainer and the learner's
functions, and writes l130-full/packet.json. It refuses an existing output folder.
The gate requires the pinned GPU runtime; installing missing basic packages alone
is insufficient. Live Colab remains NOT_CHECKED.

## Budget and delivery

USD10 aggregate across pilot, all seeds, validation and retries. T4 +2physical
cores +16GiB at .00022572USD/second. Eight bounded one-hour reservations cost at
most6.500736USD; overhead reserve3.499264USD. No automatic retries. Primary
pilot+five-fit measured worker estimate0.063100029USD excludes unitemized overhead.
Any additional notebook-gate validation is accounted separately in the same
_budget_l130.json and is never mixed into the primary five-seed aggregate.

Default notebook, source, browser/mobile/keyboard/no-JS/print, deterministic-build
and copied-Pages checks have separate reports. Report status is authoritative.
No publication requested; deployment NOT_CHECKED. Fresh notebook-GPU validation
is not a test of Google's Colab frontend.


## Completed portable full-training validation

All22 portable code cells executed with RUN_FULL_REPRODUCTION=True in an isolated
pinned T4 namespace. Five additional ten-epoch fits completed; all6,295 predictions
independently rescored, selected epochs checked and task keys/targets matched to
primary evidence. Validation3.171492452±.032179820; test4.083288832±.162746086MAE,
both descriptive CLOSE. These validation fits do not enter the primary aggregate.
Exact bitwise repeatability is NOT_ESTABLISHED; Google Colab frontend NOT_CHECKED.

The first validation launcher failed before notebook execution because its image
construction imported a sibling module unavailable remotely. That detached app
was explicitly stopped. Failed source is preserved in
sources/l130/notebook-check-attempt1.py; the corrected operator is self-contained.
Its reservation remains charged conservatively at the one-hour worker ceiling.

```bash
.venv/bin/modal run --detach modal/l130_notebook_check.py
.venv/bin/python labs/_collect_notebook_l130.py
```

These are the completed single-use recovery commands; all8 reservations are now
consumed. No further cloud work is authorized by this ledger. Primary worker
estimateUSD.063100029 +successful notebook validationUSD.041471124 =USD.104571153.
Failed setup allowanceUSD.812592 gives a worker estimate/boundUSD.917163153,
plus the unchangedUSD3.499264 overhead reserve, below theUSD10 cap. Actual total
billing is NOT_ITEMIZED. Check _budget_l130.json and _notebook_gate_l130_results.json.
