# L098 — full heterogeneous mini-batching course experiment

## Target and boundary

The roadmap specifies a RelBench-shaped toy graph and real HeteroData/NeighborLoader training. It names no published score. The complete declared experiment is **96 correctness configurations + native temporal isolation + nine full training fits**. All ran successfully. RelBench benchmark training is **NOT_RUN**; full-paper parity is **NOT_ESTABLISHED**. No published table has been substituted with a toy score.

## Run exactly

From the repository root, using the tested course environment:

```bash
.venv/bin/python labs/_verify_l098.py
.venv/bin/python labs/_run_l098.py
.venv/bin/python labs/_figures_l098.py
.venv/bin/python labs/_build_l098.py
.venv/bin/python labs/_execute_l098.py
.venv/bin/python labs/_build_l098.py
.venv/bin/python labs/_delivery_l098.py
```

The solution notebook embeds every graph, loader, operator, model, oracle and training function. It can execute in an empty working directory with the required native packages installed. Student TODOs replace three functions actually used downstream. A bootstrap cell reports versions and fails clearly if the native sampler is missing; it never falls back to a Python imitation.

## Environment and native extension

The executed host is Linux aarch64, Python 3.12, torch 2.13.0+cpu, torch-geometric 2.8.0.post1 and pyg-lib 0.10.0. `requirements-l098-observed.txt` records relevant versions; `requirements-l098-lock.txt` records the installed environment, excluding the separately pinned native extension. The exact installed Python wrappers are archived and hashed. API explanations also cite PyG 2.6.1; that older version is not misrepresented as the locally executed version.

For a new CPU environment, install the pinned runtime plus a C++ compiler, git, cmake and ninja; then compile against that environment's torch:

```bash
python -m pip install -r labs/requirements-l098-runtime.txt
MAX_JOBS=2 python -m pip install --no-build-isolation 'git+https://github.com/pyg-team/pyg-lib.git@edc9e2a88d1c5d0953b5f69c98b8365597c6b699'
```

Torch wheels and compiled extensions must match Python/platform/torch ABI. The native source revision and submodules were established in the existing course environment; this lesson **did not independently rebuild that environment**. Package availability on another platform and live Colab execution are **NOT_CHECKED**. An incompatible extension is an installation failure, not permission to skip the native tests. The standalone notebook was executed afresh from an empty directory on the author kernel; all 96 audits and nine fits matched exactly. That is code portability within the tested environment, not cross-platform proof.

## Data/model/protocol ledger

| Component | Declared protocol |
|---|---|
| Data | Complete generated 24-customer / 144-order / 12-product graph; no downloads |
| Random generation | Torch Generator seeded per graph; four normal features per type, product IDs sampled uniformly |
| Relations | Buyer and item FKs plus separately named reverse stores; 576 edges before cutoff |
| Availability | Events day 1..6; static nodes time 0; cutoff 5 removes both directions of future event edges; 480 edges remain |
| Labels | Sign of mean product feature 0 among a customer's first five orders; synthetic historical target |
| Features | Generated before labels; no target channel, normalization, dropout or BatchNorm |
| Model | Two typed mean layers 4→16→16, per-type root and per-relation message transforms, ReLU, customer linear logit |
| Native sampling | Two hops, per-relation fanout, no replacement, directional edges, num_workers=0, fixed seed order |
| Correctness | Graph seeds 0..31 × batch sizes 1/4/24; all 24 customers; all neighbors |
| Oracle | Independently built dense destination×source matrices; output and parameter-gradient checks at frozen tolerance 2e-6 |
| Gradient accounting | Batch mean multiplied by B/24; no optimizer step between audit batches; mutually unused parameters checked |
| Temporal | Customer 0 queried at times 2 and 5; disjoint native components; independent cutoff-snapshot logits; future-feature perturbation |
| Training | One graph seed 98; train customers 0..15, validation 16..19, test 20..23; seeds 0,1,2 |
| Arms | All neighbors, fanout 2, fanout 1; each runs 40 full epochs, four batches/epoch |
| Optimization | Adam lr .02, default betas/eps, zero weight decay; BCEWithLogitsLoss mean; width 16 |
| Selection | Earliest strictly lowest full-neighbor validation BCE; test only after checkpoint selection |
| Evaluation | Full-neighbor test BCE and accuracy; separately one sampled-logit diagnostic over all 24 seeds |
| Saved evidence | Full 40-epoch train/validation traces, selected epoch, trained state_dict as JSON arrays, test logits/labels and workload proxies |
| Uncertainty | Three initializations on one generated graph; four test targets; no dataset-level inference |

## Sources and deviations

`_sources_l098.json` pins RelBench commit 584a03d518b2b655580ea8e1cfbbb26bec0a2841, native pyg-lib source edc9e2a88d1c5d0953b5f69c98b8365597c6b699, API reference bytes and actual installed wrappers. The model is a transparent course encoder, not the full RelBench model. There is no database download, real temporal task definition, feature encoder pipeline, dataset split reconstruction, hyperparameter benchmark search or published score comparison. Those missing components explain **NOT_RUN**, not a claim of benchmark parity.

## Evidence interpretation

`_experiment_l098_results.json` contains all records, source hash and measured environment. `_verify_l098_results.json` reports behavioral tests; `_execution_l098_results.json` reports fresh notebook execution; `_delivery_l098_results.json` reports browser/copy-staging tests. Full-neighbor output parity does not establish sampled-logit unbiasedness. Node/edge counts do not establish a peak-memory or speed benchmark. Full-gradient accumulation is not a claim that sequential SGD updates follow the same trajectory.

No learner mastery inferred: **PENDING_WRITTEN_DEFENSE**. Deployment and live Colab remain separate, **NOT_CHECKED** states.
