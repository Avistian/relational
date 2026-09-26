# L108 — temporal sampling: reproduction contract

Approved scope: full Wikipedia evaluation of ten existing TGAT checkpoints, plus strict-past sampling interventions and a full-data CPU sampler benchmark. Aggregate cloud budget: USD10. This is an efficiency/evaluation lesson. L108 does not train new model weights.

## Reproducible commands

From the relational checkout, use Python 3.12. The lesson notebook uses the pinned local teaching runtime in `requirements-l108-authoring.txt`. GPU evaluation uses the separate `requirements-l108-runtime.txt`; Modal builds it automatically.

```bash
.venv/bin/python labs/_check_l108.py
.venv/bin/python labs/_audit_l108.py
.venv/bin/python labs/_benchmark_l108.py
# Local small pilot (not full evaluation evidence):
.venv/bin/python labs/_run_l108.py --seed 0 --pilot --max-seconds 240
# Approved full evaluation lane; run a GPU pilot before committing the budget:
.venv/bin/modal run modal/l108_replay.py --pilot
.venv/bin/modal run --detach modal/l108_replay.py
.venv/bin/python labs/_collect_l108.py --download
.venv/bin/python labs/_provenance_l108.py
.venv/bin/python labs/_figures_l108.py
.venv/bin/python labs/_build_l108.py
.venv/bin/python labs/_execute_l108.py
.venv/bin/python labs/_mutation_l108.py
# After the local pilot, exercise unchanged reuse and changed-helper rejection:
.venv/bin/python labs/_resume_l108.py
.venv/bin/python labs/_delivery_l108.py
```

A full local evaluation is also available via `_run_l108.py --device cpu --seed N --max-seconds 2850`, but may exceed the cutoff. Completed CPU and CUDA results have different identities; neither silently replaces the other. A timeout produces no completed result. Ten seeds are fixed before evaluation; partial results cannot be reported as the full lane.

### Input retrieval and fresh training

The raw public data is `https://snap.stanford.edu/jodie/wikipedia.csv`, SHA256 `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09`. The first download is about 560 MB. `load_wikipedia` authenticates raw bytes and builds `processed.npz`; the L108 runner additionally authenticates every processed array against `_inputs_l108.json`.

The ten selected checkpoints and original prediction archives are already local under `labs/results/l103/gpu/seed-N/`. An authorized operator can retrieve the same immutable inputs with:

```bash
.venv/bin/python labs/_download_l103.py
```

The named Modal input volume is `l103-tgat-evidence`, path `6a9dc0868fd047a68b3fbc547eb250898fb741a0792a8610b45c8dfdea48ae4f/paper` **as derived programmatically from `_inputs_l108.json`**, not a manually typed path. Use the downloader; the manifest is authoritative. Checkpoints are not embedded in the notebook or hosted as a public download. The local operator's Modal credentials are needed to retrieve that private volume. This access requirement is an explicit portability boundary.

For independence from those archived weights, the complete model/trainer is visible in the notebook and `relkit/tgat_l103.py`. The [L103 protocol](l103-reproduction.md) gives the full ten-seed training command:

```bash
.venv/bin/modal run --detach modal/l103_paper_repro.py::main --preset paper
```

That command belongs to the upstream training lane and may reuse completed L103 runs. To perform genuinely fresh training, use a new evidence destination and freeze the new checkpoints and prediction archives as a distinct cohort before evaluating them. Do not overwrite `_inputs_l108.json` or relabel new weights as the archived L108 cohort. Fresh training is outside the approved L108 run and budget; no new training cost is hidden in this replay.

## Protocol and deviations

| Component | Named released replay | L108 inference interventions |
|---|---|---|
| Paper/source | Xu et al. ICLR 2020, Tables 1/2 Wikipedia; source commit `9293d10d1943c4bd4a186337cf38ba98e4c8bb99` | Same frozen trained model; independent sampler |
| Data/split | 157,474 events, 172 edge features, same quantile split and held-node selection as L103 | Identical full history and queries |
| Architecture | Two TGAT attention layers, two heads, 172-wide features, cosine time encoder, shared pair MLP | Unchanged |
| Objective/optimizer/selection | Previously trained L103 BCE, Adam 1e-4, batch 200, max 50 epochs, release early-stopping/checkpoint convention; ten seeds | No optimizer, fitting, tuning or selection |
| Sampling | Source uniform with replacement, fanout 20, edge-ID sort and omitted boundary element preserved | Timestamp/event-ID sort, strict [cutoff-window,cutoff), uniform20/uniform5/recent20/day20 |
| Recursive time | Child inherits traversed edge timestamp | Identical recursion; checked by trace |
| Padding | ID zero, finite -1e10 attention mask, all-padding behavior preserved | Same model padding behavior |
| Questions | 23,620 all-event and 11,714 new-node positives per seed; release final-event omission preserved | Exact same event IDs, negatives and batches |
| Negatives/RNG | Restore checkpoint NumPy state; preserve original draws, including unused negative sources | Frozen candidates, reset seed 108+seed for each population/condition |
| Random coupling | Exact source stream | Same seed is not same variates across changed recursive fanout |
| Metric | Unweighted mean of per-batch AP; compare means against 95.34% / 93.99%, 0.5 pp course tolerance | Pooled AP per population per seed; paired differences from uniform20 |
| Timing | Not a source hardware-performance reproduction | Warmup, rotated arm order, GPU synchronization, all scoring/sampling/transfers, one timing pass per seed |
| Memory | Checkpoint provenance | Index-array bytes plus separate CUDA peak allocated tensor bytes; not process RSS |
| Availability | Event time only; assume a=t | Same limitation; no invented ingestion histories |
| Environment | Python 3.12, Torch 2.8.0, NumPy 2.2.6, pandas 2.3.2, sklearn 1.7.1 | Identical GPU environment |
| Historical differences | Original Torch 1.1 runtime and historical RNG/search unavailable; source release bugs/parameterization differ from paper | Additional intentional context changes |

The source's “all” population includes new nodes; it is not a disjoint old-node subset. Both populations overlap. New-node negatives need not themselves involve new nodes. Negative collisions are preserved. Source behavior and literal paper protocol are distinguished in the linked L103 ledger.

Uniform selection repeats eligible records; recent selection pads scarce histories. Comparing recent against uniform therefore changes both recency and replacement/padding behavior. The window changes context but retains all stored rows; it does not reduce persistent index memory. The 421-versus-31 neighbor-tree count omits TGAT source recursion and three endpoint branches; it is not a measured speed ratio.

On Wikipedia, all timestamps are integer values exactly representable as float32. The TGAT adapter returns float32 child timestamps to match the model. Reusing this adapter with large absolute Unix times or fractional high-precision timestamps requires a precision audit or a redesigned time representation.

## Evidence and restart rules

`_inputs_l108.json` pins raw/processed/split bytes, every checkpoint and archived evaluation file. `_sources_l108.json` names the source claims. `_run_l108.py` fingerprints the full sampler, TGAT, replay helper, runner and input manifest. A completed result is reused only if recorded Python/Torch/NumPy/platform, input and code identities match and its prediction archive hash verifies. Changing a helper changes the fingerprint; no incomplete checkpoint evaluation is counted. These are conservative source-byte identities, not interpreter `marshal` hashes.

`labs/evidence/l108/full/seed-N/{result.json,predictions.npz}` stores every source and intervention prediction, positive event ID, negative destination and batch ID. `_collect_l108.py` independently reconstructs pooled and batch AP and all paired differences, verifies full counts and zero recorded temporal violations, and records hashes in `_analysis_l108_results.json`. Source replay checks each probability against L103 archives; L103's independent original-model seed-0 replay is inherited provenance, not a newly run original-source evaluation in L108.

`_benchmark_l108_results.json` is the separate local CPU benchmark. It checks all queries against the readable reference with the same variates, alternates measurement order, and reports three runs. `_audit_l108_results.json` checks complete model scalar/vector output equality, recursive cutoffs, and changed-helper identity fingerprints. `_resume_l108_results.json` exercises actual unchanged completed-pilot reuse and rejection after changing the boundary helper in an isolated copy. `_execution_l108_results.json` records the actual standalone notebook execution. Browser/Pages checks live in `_delivery_l108_results.json`.

## Budget and claim limits

Current prices were checked at `https://modal.com/pricing` on 2026-09-26: T4 USD0.000164/s; conservatively two CPU cores USD0.0000262/s; 8 GiB USD0.00001776/s. At USD0.00020796/s, eleven 3000-second caps cost at most USD6.86268 in requested resources, leaving USD3.13732 for startup/build/storage overhead. One pilot plus ten full seeds; no automatic retries. The inner deadline is 240 seconds for pilot evaluation and 2850 seconds for each full run, with a 3000-second remote hard timeout. See `_budget_l108.json` for actual recorded elapsed-time estimates; these are not provider invoices.

Complete selected evaluation does not establish full-paper reproduction. Fresh L108 training, Reddit, industrial data, node classification and retrained sampler comparisons are NOT_RUN. Historical identity is INCOMPARABLE; full-paper parity NOT_ESTABLISHED. Live Colab and deployment are NOT_CHECKED. Learner mastery remains PENDING_WRITTEN_DEFENSE.
