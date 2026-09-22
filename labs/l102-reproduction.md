# L102 — complete selected TGN-attn Wikipedia replay

## Target and frozen tolerance

Rossi et al., arXiv:2006.10637v3, Table 2, TGN-attn Wikipedia AP: **98.46 ± 0.1%** transductive and **97.81 ± 0.1%** inductive. Ten complete runs, complete released dataset and both evaluation branches. Numerical CLOSE means each ten-run mean is within **0.5 percentage point**, fixed before results. This is a descriptive course criterion, not an equivalence test. Historical identity is INCOMPARABLE; full-paper parity NOT_ESTABLISHED.

## Run

```bash
# From the repository root. Use a fresh output directory if implementation/runtime changes.
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l102.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l102.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l102.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l102.py --preset smoke --seeds 0,1,2
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l102.py --preset paper --seeds 0,1,2,3,4,5,6,7,8,9

# Same model and trainer on a separately pinned GPU runtime; persistent Modal volume.
.venv/bin/modal run --detach modal/l102_paper_repro.py::main --preset paper
# After seed 0 completes, replay its full evaluation with the unmodified released model.
.venv/bin/modal run modal/l102_paper_repro.py::source_check

# Download the completed author run artifacts, including checkpoints (about 0.5 GiB).
.venv/bin/modal volume get l102-tgn-evidence /4a1d877bcbc37106561f8eafc2ea1e05d4b4863c3bf7ff623d2bc7bbf9f5368d/paper labs/results/l102/gpu --force

# Author packaging after collecting the completed GPU run directory:
.venv/bin/python labs/_collect_l102.py labs/results/l102/gpu
.venv/bin/python labs/_plot_l102_results.py
.venv/bin/python labs/_figures_l102.py
.venv/bin/python labs/_build_l102.py
.venv/bin/python labs/_execute_l102.py
.venv/bin/python labs/_delivery_l102.py
```

**Artifact layout:** the author run accelerated six remaining seeds into separate groups; `_run_selection_l102.json` fixes their source folders before results. `_collect_l102.py` defaults to that layout. For a fresh run using the default four GPU groups, pass `--layout grouped` when collecting. This changes scheduling and storage only, not any seed’s protocol.

The solution notebook contains the complete model, data processing and trainer inline. Default execution runs three live task checks and a fresh short real-data training experiment. `RUN_PAPER_REPRO = True` runs all ten full-data initializations; the default remains False because it is a long-running experiment. No hidden course/model import is needed.

## Protocol audit

| Dimension | Frozen choice |
|---|---|
| Paper | Rossi et al. v3 (9 Oct 2020), Table 2, two Wikipedia TGN-attn AP cells |
| Source | `twitter-research/tgn` publication-era commit `e38cdf85998c6ca077167610dc4e769a688efa95` |
| Original code | Archived, unmodified, with Apache-2.0 license in `sources/l102/` |
| Raw data | `https://snap.stanford.edu/jodie/wikipedia.csv` |
| Raw SHA-256 | `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09` |
| Preprocessing | Preserve stream order, offset page IDs beyond user IDs, reserve zero padding; 172 edge features; all-zero 172-wide node features |
| Data | 157474 interactions, 9227 nodes; independent release preprocessing and split checks |
| Split | 70%/85% timestamp quantiles, Python sampling seed 2020, 922 held nodes selected from post-train-cutoff nodes |
| Complete counts | Train 81029, val/test 23621 each, new-val 12016, new-test 11715 |
| Architecture | One graph-attention layer, two heads, latest ten strict-past interactions, memory width 172, cosine time width 172 |
| Message | Identity concatenation of two memories, edge and time = 688; last message per node; GRU updater |
| Initialization | Release-equivalent constructors including cosine frequencies and Xavier-normal merge weights; no pretrained checkpoint |
| Objective | Mean positive BCE + mean negative BCE; one uniformly sampled destination per event |
| Negative RNG | Train global per-run NumPy RNG; validation seed 0; all-test seed 2; new-test seed 3. Consume unused source draws as release does |
| Negative collisions | No rejection of existing/positive pairs, matching release |
| New-node evaluation | At least one endpoint unseen in training; negative destination pool derived from that subset, not guaranteed all unseen |
| Validation | Full-node sampler for both branches, as the released trainer actually calls it |
| Training | Chronological batch size 200, Adam lr 0.0001, dropout 0.1, maximum 50 epochs, detach after each batch |
| Selection | All-event validation AP; patience 5, relative improvement tolerance 1e-10; if maximum reached, release uses final epoch |
| Metric | Equal average of per-batch AP and AUC; includes shorter final batch at equal weight; not pooled AP |
| Pending encoding | Stored concatenated messages already contain a computed time encoding; detach cuts that old graph. The time encoder learns through the attention path |
| Epoch reset | Zero all memory/clocks/pending messages at the start of each epoch |
| Branch reset | Validation/new-validation start at the same train snapshot; test/new-test start at the same restored validation snapshot |
| Full graph finder | May index the entire stream but returns only edges strictly earlier than each query; training uses training-only adjacency; evaluation restores earlier withheld-node history as context |
| Time statistics | Release calculates full-data inter-event statistics, but one-layer graph-attention embedding does not consume them; omitted inert computation |
| Seeds | Explicit independent model/NumPy seeds 0–9; a declared deviation from the original single seed followed by ten continuous RNG runs |
| Uncertainty | Sample standard deviation across ten model initializations on one fixed data split; not independent datasets |
| CPU environment | Python 3.12.3, torch 2.13.0+cpu, numpy 2.5.0, pandas 3.0.3, sklearn 1.9.0; one thread |
| GPU environment | Modal T4; Python 3.12 image; torch 2.8.0, numpy 2.2.6, pandas 2.3.2, sklearn 1.7.1; actual versions saved per group |

## Preserved source behavior versus robust state handling

The original `state_dict` omits queued messages. On early stop it restores selected-epoch model weights, persistent memory and clocks, but retains stopping-epoch pending messages. The named replay deliberately preserves that behavior. Saved course checkpoints additionally include a full temporal snapshot, so replaying the recorded evaluation remains possible. Fixing the historical training-selection policy belongs in a separately labeled experiment.

For operational checkpointing, save weights, memory, clocks, pending messages, event cursor and RNG/optimizer state consistently. The current per-seed artifacts support evaluation replay, not mid-epoch optimization resume. Completed per-seed CPU runs may be reused only when `identity.json` matches code, data, runtime and configuration. Partial epochs are not resumable and never count as completed evidence.

Epoch indices in saved traces and `selected_epoch` are zero-based; `epochs_completed` is a count. The run selector is frozen independently of test scores in `_run_selection_l102.json`.

## Execution and evidence boundaries

**Completed:** ten T4 runs, 293 epochs in total. All-event AP **98.5123% ± 0.0679pp sample SD** versus paper 98.46%; new-node AP **97.8295% ± 0.0861pp** versus paper 97.81%. Absolute gaps are 0.0523pp and 0.0195pp: both **CLOSE** under the frozen 0.5pp tolerance. Independent reconstruction covers 353360 real test-event evaluations (plus their paired negatives), with maximum metric error 2.22e-16. Unmodified released-source inference on seed 0 reproduces both AP values exactly, with probability differences below 3e-6.

The exact completion status and measured results are in `_paper_l102_results.json`. CPU pilot traces are incomplete and excluded from paper means. The primary complete-run environment is recorded alongside the results; do not mix CPU and GPU runs.

- `_source_check_l102_results.json`: CPU comparison of probabilities, gradients, persistent memory and pending messages against original source.
- Per-GPU-group `source_parity.json`: the same check in the actual GPU runtime.
- `_state_check_l102_results.json`: warm-state current/future-feature exclusion, complete checkpoint counterexample and backward-time rejection.
- `_resume_check_l102_results.json`: unchanged completed-run reuse and rejection of a changed message helper.
- `_audit_l102_results.json`: complete independent preprocessing/split reconstruction and temporal/state oracles.
- `_experiment_l102_results.json`: separate three-seed short teaching runs (600 training events, 200 per evaluation branch, 2 epochs; complete strict-past neighbor index retained), INCOMPARABLE to Table 2.
- `_execution_l102_results.json`: standalone notebook task execution and exact seed-0 short-run parity; full gate off.
- Per-seed predictions/checkpoints/traces: durable experiment evidence, with file hashes in the final manifest. Predictions support independent AP reconstruction; large checkpoints remain in the persistent volume/local results cache.
- `_delivery_l102_results.json`: browser, notebook packaging and copied Pages checks, separate from scientific evidence.

The selected experiment excludes Reddit, Twitter, node classification, all competitor training and the ablation suite. Original runtime (torch 1.6/pandas 1.1/sklearn 0.23.1), original random streams and historical data identity are not recreated. Numerical CLOSE is not full-paper reproduction. Live Colab and deployment remain NOT_CHECKED until separately exercised. Learner status is PENDING_WRITTEN_DEFENSE.
