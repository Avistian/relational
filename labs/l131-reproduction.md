# Lesson131 · measured GNN + tabular encoder stack

## Named experiment and result

RelBench arXiv2407.20060v1 Table7 basic RDL, rel-f1/driver-position.
Five fresh full-data ten-epoch runs, seeds0–4. Validation MAE3.187859582 ±.020248973;
test4.119595342 ±.228564635 (mean ±sample seed SD). Published means3.193/4.022;
both descriptive CLOSE under predeclared absolute mean tolerance.2. Not an equivalence test.
All6295 final predictions independently rescored and replayed through the original model;
403895 sampled query occurrences audited. SQL graph counts and official task archive IDs,
timestamps and labels independently verified. See evidence/l131/summary.json and
_audit_l131_results.json. Selected experiment COMPLETE; historical/whole-paper parity
NOT_ESTABLISHED, other29tasks NOT_RUN here.

## Protocol and source ledger

| Axis | Contract / deviation |
|---|---|
| Paper | arXiv2407.20060v1 Table7, pinned HTML in sources/l131/paper.html |
| Blueprint | Fey2024 §5, https://proceedings.mlr.press/v235/fey24a.html |
| Released source | RelBench9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; historical training commit unknown |
| DB archive SHA256 | ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task archive SHA256 | 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Full data | Nine tables,74063rows,338842directed edges; database through2010-01-01 |
| Queries | 7453train /499validation /760test; entity + nanosecond cutoff |
| Cohort | Released future-participation conditioning; not an operational past-only cohort |
| Features | Keys removed; type inference seed42; database statistics through test cap, not train-only |
| Text | Pinned sentence-transformers GloVe model/revision in sources/l117/text_model.json |
| Model | Four-block per-table Frame ResNets,128channels, relative time, two typed sum-GraphSAGE layers, scalar root head |
| Sampling | Uniform temporal [128,64]; paper hyperparameter table says128 |
| Training | Adam.005, batch512, mean L1, ten full epochs, seeds0–4 |
| Selection | First strict validation MAE improvement; frozen final test evaluation |
| Scoring | Train-label2nd/98th percentile clipping, official evaluator + independent keyed MAE |
| Validation | Final resampling differs from checkpoint-selection sampling |
| Runtime | Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame0.2.3,RelBench1.1.0; full requirements-l117-runtime.txt |
| Unobserved | Arrival histories, mutable-feature histories, static-row creation times |
| Historical identity | Original seeds, runtime and exact training source NOT_ESTABLISHED |

All inherited source files and licenses are visible in sources/l117; load-bearing source
hashes in _sources_l131.json and _budget_l131.json. No primary model/trainer algorithm
was changed to obtain a closer score.

## Instrumentation and the gradient finding

Each seed records its actual first training minibatch before an optimizer update.
Three isolated model copies use the same batch/state/RNG: original forward, explicitly
instrumented forward, and encoded-row detach intervention. Recorded stages include row
vectors, query-relative days, time vectors, addition, relation sums, normalization, ReLU,
root readout and raw prediction. Shapes and summary coordinates are real activations,
not exhaustive structural counts from an earlier lesson. These are initialized-model
traces, not trained feature interpretations.

Compare outputs/loss, gradient presence, exact NaN/Inf masks and finite parameter gradients.
GPU scatter reduction allows numerical tolerance; CPU fixture also checks the original
source output/gradient/Adam update. Detached encoders retain forward values but lose their
parameter gradients. State/RNG preservation is verified by the standalone neural fixture.

The first pilot failed because a strict gradient check rejected NaNs. The recovery checks
established matching nonfinite gradient masks in original and instrumented paths, specifically
results numerical-encoder weights. Gradient norms exclude nonfinite entries and counts are
reported beside them. A separate minimal autograd diagnostic demonstrates that replacing
encoded NaN outputs with zero does not guarantee finite upstream gradients. The released
algorithm remains unchanged: parity is not a claim of healthy optimization. Failure source,
budget snapshot and log are retained in sources/l131/failed-pilot/.

## Exact commands

From repository root in the author environment:

```bash
.venv/bin/python labs/_check_l131.py
.venv/bin/python labs/_mutation_l131.py
.venv/bin/python labs/_source_check_l131.py
.venv/bin/python labs/_missing_gradient_l131.py
.venv/bin/modal run --detach modal/l131_repro.py --mode pilot
.venv/bin/python labs/_collect_l131.py --mode pilot
.venv/bin/python labs/_pilot_check_l131.py
.venv/bin/modal run --detach modal/l131_repro.py --mode paper
.venv/bin/python labs/_collect_l131.py --mode paper
.venv/bin/python labs/_analyze_l131.py
.venv/bin/python labs/_audit_l131.py
.venv/bin/python labs/_trace_audit_l131.py
.venv/bin/python labs/_figures_l131.py
.venv/bin/python labs/_build_l131.py
.venv/bin/python labs/_execute_l131.py
.venv/bin/modal run --detach modal/l131_notebook_check.py
.venv/bin/python labs/_collect_notebook_l131.py
.venv/bin/python labs/_delivery_l131.py
.venv/bin/python labs/_verify_l131.py
```

Completed paid reservations cannot be reused or overwritten. Reproduce independently with
a new volume and budget ledger; preserve original reservations and outputs. The first
failed pilot remains charged; its recovery was an explicit one-time action. No application-level
retries; infrastructure preemption may restart the function, where the overwrite guard stops it. The prepare script records primary paper/source hashes; it is not a training run.

For fresh local training with the pinned GPU/native-pyg-lib runtime:

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l131.py --seed "$seed" --epochs 10 \
    --output "labs/results/l131/my-new-run/seed-$seed"
done
```

Use fresh output directories. The local operator and notebook gate do not enforce a dollar
cap. The notebook embeds complete visible model/trainer/audit code and author trace/prediction
payloads; default execution needs no repository imports. Three learner functions remain live
in replay, the neural fixture and full trace path. RUN_FULL_REPRODUCTION enables all five
fresh fits and saves l131-full/packet.json plus stack-traces.json. The default run is author
artifact replay plus a small independently executed neural fixture, not full training.
A separate isolated pinned-GPU notebook check exercises the full gate; its report is
_notebook_gpu_l131_results.json: PASS, all24code cells and five full fresh fits.
All6295 notebook predictions and five trace reports were independently checked after
download. Notebook-validation means3.198490216/4.014245388MAE remain separate from
the primary experiment. This does not test Google's Colab frontend.

## Aggregate budget

USD10 across all seeds, failed attempts, validation and overhead. Rates checked2026-09-27:
T4.000164 +2physical CPU cores*.0000131 +16GiB*.00000222 =.00022572USD/second.
Source: https://modal.com/pricing. The original eight-reservation plan was revised within the same USD10 cap after notebook
preemption: ten reservations, at most3600seconds each, yield maximum worker
cost8.125920USD plus1.874080USD overhead reserve. Reservations are failed pilot1
+successful recovery pilot1 +primary fits5 +preempted notebook1 +guard-stopped
infrastructure restart1 +fresh notebook recovery1. Preemption interrupted the second
notebook training seed; its existing artifacts remain in the original volume and its
operator/log are archived under sources/l131/preempted-notebook/. The recovery uses a
fresh volume; partial seeds are never merged into the primary five-seed result.
Successful pilot and five primary fits measured worker estimate.061831445USD. The failed
pilot has a conservative full-reservation ceiling.812592USD; actual billing is NOT_ITEMIZED.
Successful notebook recovery worker estimate0.041215048USD;
all successful workers total0.103046493USD. The initial pilot,
preempted notebook and guard-stopped restart have combined conservative ceiling
2.437776USD. Successful worker estimates plus those
failed/preempted bounds total2.540822493USD,
before the1.874080USD overhead reserve. Billing remains NOT_ITEMIZED. Do not treat the primary-fit
estimate as the entire invoice. All dispatches are bounded and the full pilot gates five-fit
projection before launch.

## Delivery and mastery

Execution, source, mutation, browser/mobile/keyboard/noJS/print, deterministic rebuild and
copied-Pages checks have separate authoritative reports. Live Colab/deployment NOT_CHECKED.
No publication requested. Author execution leaves learner PENDING_WRITTEN_DEFENSE; submit
the lesson's annotated trace and five written answers.
