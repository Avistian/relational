# L093 — HGT reproduction contract

Named target: Hu et al. WWW2020, Table2 CS Paper–Field L2, HGT(+Heter,+RTE). Published NDCG .403±.041, MRR .439±.078 over five model trainings. No numerical parity gate is claimed because protocol identity is unresolved. Full-paper parity: **NOT_ESTABLISHED**. Historical comparison: **INCOMPARABLE**.

## Three distinct execution scopes

1. **NN teaching:** width32, two layers, eight heads, dropout.2,10 epochs×4 batches×1 update,32 seed papers, sampling depth2/width16,10 test batches. Three seeds0–2; HGT, relation-mean R-GCN and HGT without RTE. Same sampling streams per seed across arms. Nine fits completed. One graph is not a multi-dataset benchmark.
2. **CS paper-setting reconstruction:** width256, three layers, eight heads,200 epochs×32 batches×2 repeats,256 seed papers, sampling depth6/width128,10 sampled test batches, five fresh seeds0–4. Validation KL selects. Full modern release operators remain; this is not the printed paper architecture or exact historical source. NOT_RUN.
3. **CS modern-release replay:** same complete schedule but width400/four layers and validation NDCG selection. Modern serial port, declared sampling/RNG changes. NOT_RUN.

The word `paper` is a configuration name for scope2, not a fidelity verdict. All presets expose the same load-bearing model and trainer in the notebooks. The full paper includes three graphs, four tasks, baselines, ablations and case studies; reproducing even one CS row would not complete that suite.

## Sources and chronology

- Paper: https://arxiv.org/html/2003.01332v1
- Main audited release: https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG (2023 revision). Archived intact under `sources/hgt-l093/OAG`; MIT license retained.
- Publication-era source: https://github.com/acbull/pyHGT/tree/fd4a244db8efc72410537f3effec3b0c432892f7 (2020-04-16). Archived under `sources/hgt-l093/publication-2020`. It has a triplet-indexed prior, no LayerNorm, dropout after the residual mixture, and a different2d temporal table with a different frequency expression. It is **not identical** to the modern audited operator. Exact table-producing source revision is unknown.
- `_sources_l093.json` has SHA256 for every archived file and both downloaded graph files.

Both source eras default to width400/four layers and validation NDCG, whereas the paper states width256/three layers and lowest validation loss. We do not silently choose one and call the discrepancy resolved.

## Acquired data

Authors' folder: https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq

| Data | Bytes | SHA256 |
|---|---:|---|
| NN |656262985|`7e0e7b997c427af479148f1d9af419878648b0624b4d3f9f1790bc3625cd7e61`|
| CS |8606617611|`bf054d93f45d385691f89902e7a021c53328c2ffefe50eb4d91c9dc51d05877f`|

The linked CS file's folder metadata says modified2020-11-28, after publication. Identity to paper Table1's CS graph and original feature bytes is NOT_ESTABLISHED. Files remain local caches; the site does not publish multi-GB data. The fetcher downloads the authors' files and checks known hashes. Loaders fail before deserializing if hashes differ.

NN contains paper18911, author32307, venue2008, affiliation3445, field9540. L2 candidates4782; train8489 (<2015), validation3439 (2015–2016), test6983 (>2016). These are measured counts, not the paper's CS statistics.

## Protocol/deviation ledger

| Element | Implemented and checked | Boundary |
|---|---|---|
| Q/K/V | node-type linear projections, separate relation attention/message matrices | Modern release oracle, not historical backend |
| Scale/prior | sqrt(head width), relation/head prior | Printed sqrt(total width);2020 prior is triplet-indexed |
| Softmax | all incoming edges per receiver/head | Independent numerical trace and upstream outputs/gradients checked |
| Output | GELU, type projection, transformed-branch dropout, learned gate, LayerNorm |2020 source has different dropout placement and no norm; paper writes additive residual |
| RTE |240 rows, gap+120, scaled paired sinusoids, learned projection and trainable embedding | Actual modern release behavior; printed fixed basis and historical2d table differ |
| Input features | supplied400-vector + text embedding + log10(citation+.01), type adapter/tanh/dropout | Raw XLNet/metapath2vec preprocessing NOT_RUN; point-in-time feature provenance NOT_ESTABLISHED |
| Loader | maps old Graph and pandas index classes; discards obsolete serialized lambda code for populated-map factories | Stored values retained; original Python3.7 decode identity NOT_CHECKED |
| Sampling | capped relation expansion, squared per-type budgets, inherited times, induced subgraph | Serial execution; original multiprocessing streams not recreated |
| Time | expansion respects cutoff; induced edges reconstructed as source does | Not a causal edge/feature audit; no unsupported point-in-time guarantee |
| Target links | both L2 relation directions masked for seed papers | Other context labels remain observed under released protocol |
| Splits | code's non-overlapping year conditions, train/valid shuffle seed43 | Printed2016 overlap resolved by code; future-only forecast not inferred |
| Objective | normalized field indicators, batch-mean KL divergence | Not binary cross entropy or ordinary single-label CE |
| AdamW | lr.001, weight decay.01, beta(.9,.999), eps1e-8, gradient norm clip.25 | Modern optimizer numerics differ from historical runtime |
| Scheduler | cosine T_max1000, eta_min1e-6, explicit counter starts1500 | Unusual released schedule retained |
| Selection | modern-release/teaching maximize validation NDCG; paper preset minimizes validation KL | Test never selects a checkpoint |
| Evaluation | selected checkpoint; ten sampled test batches; all candidate fields | Original also reports last model before best; its extra RNG consumption not reproduced |
| DCG | release method0: rank1/2 denominator1 | Not standard log2(rank+1) convention |
| Randomness | model seeds and local NumPy sample streams recorded | Historical initializations/seeds unavailable |
| R-GCN | independently written relation-mean comparator with shared task/streams | Not the missing original Table2 R-GCN baseline; budgets equal, parameter counts differ |

No resumed partial training is claimed; the CLI starts fresh and saves completed run records. Sampling seeds, query IDs, relevant ranks, target field IDs, validation trace, selected epoch and source/data hashes make those records auditable. A seed is a run replicate, not another dataset. Test batches can overlap.

## Exact commands from course root

```bash
.venv/bin/python labs/_fetch_l093.py --dataset NN
.venv/bin/python labs/_verify_l093.py
.venv/bin/python labs/_run_l093.py --preset smoke --output labs/_smoke_l093_results.json
.venv/bin/python labs/_run_l093.py --preset teaching --seeds 0 1 2 --arms hgt rgcn hgt_no_rte --output labs/_teaching_l093_results.json
.venv/bin/python labs/_baseline_l093.py
.venv/bin/python labs/_audit_l093.py
.venv/bin/python labs/_probe_l093.py
.venv/bin/python labs/_fetch_l093.py --dataset CS
.venv/bin/python labs/_probe_l093.py --cs-load
.venv/bin/python labs/_figures_l093.py
.venv/bin/python labs/_build_l093.py
.venv/bin/python labs/_execute_l093.py --teaching
.venv/bin/python labs/_delivery_l093.py
```

Full named target on a sufficiently provisioned host (unrun):

```bash
.venv/bin/python labs/_run_l093.py --preset paper --data labs/data/l093/graph_CS.pk --sha256 bf054d93f45d385691f89902e7a021c53328c2ffefe50eb4d91c9dc51d05877f --device cuda --seeds 0 1 2 3 4 --output labs/_paper_l093_results.json
# Substitute --preset release and a separate output for the modern release-default replay.
.venv/bin/modal run --detach modal/l093_paper_repro.py --preset paper
```

The Modal entrypoint downloads and validates CS bytes, uses the same canonical module, saves partial completed-seed evidence to volume `l093-hgt-evidence`, and requests64GiB host RAM/A10G. Capacity for a complete CS epoch remains unverified. It does not auto-resume interrupted epochs or claim an unexecuted trial succeeded. The smoke invocation built its pinned image but was rejected before GPU execution: **“Please add a payment method to use A10G GPU functions.”**

Original publication-era replay, in its own compatible historical runtime (NOT_RUN):

```bash
cd labs/sources/hgt-l093/publication-2020
# Historical dependencies are documented in its README and requirements.txt.
# The author environment is PyTorch1.3.0 / PyG1.3.2; modern notebook pins are not that environment.
python train_paper_field.py --data_dir /absolute/path/to/l093-data --model_dir /absolute/path/to/existing/output --domain _CS --conv_name hgt
```

That archived trainer is complete and inspectable. Its default run is a historical release replay, not automatically a reproduction of the paper's stated dimensions/selection. A successful historical run would still require a data snapshot, preprocessing and metric audit before upgrading the verdict.

## Measured outcome and rejected inference

Nine NN fits completed. HGT: NDCG.41428±.00826, MRR1.0. R-GCN: NDCG.41700±.00483, MRR.99757±.00235. No-RTE HGT: NDCG.42214±.00667, MRR1.0. These SDs summarize three model/sampling seeds on one graph.

The **training-frequency baseline** achieves NDCG.48796±.00749 and MRR1.0 on the same HGT query IDs. Field6625, “Artificial neural network”, occurs in every training and test paper in NN. MRR is therefore trivial for a model that ranks that field first. This disproves the tempting inference that perfect NN MRR establishes useful generalization or paper parity. Baseline variability is only query-sampling variability, not initialization uncertainty.

One full-width256/three-layer NN update used3460 sampled nodes and37910 edges. This is an operator/capacity check, not CS performance. CS decoding raised MemoryError under an explicit10GiB virtual-address guard. The account-level GPU rejection is separately recorded. Full CS five-run reconstruction, exact historical framework replay and all remaining full-paper experiments are **NOT_RUN**.

The initial pilot comparison was superseded after removing an unused R-GCN parameter allocation. `_pilot_l093_results.json` records the rejected pilot summary. Delivered results come from the fresh corrected run, with compact relevant-rank records rather than enormous full ranking lists.

Portable dependency pins are in `requirements-l093-runtime.txt`; author-observed versions in `requirements-l093-observed.txt`. A clean-environment check establishes only its stated scope. Local HTML, copied Pages staging, portable notebook execution, live Colab and deployment are separate checks. Live Colab/deployment remain **NOT_CHECKED**. Learner mastery remains **PENDING_WRITTEN_DEFENSE**.
