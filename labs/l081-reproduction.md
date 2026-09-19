# L081 reproduction contract

## Named published experiment

Gilmer et al. (ICML 2017), supplementary **Table 3 / GG-NN / mu**, sparse chemical graph, no distances. Error ratio 3.94 × chemical accuracy 0.1 Debye = **0.394 Debye MAE**. This is not the best edge-network/set2set result.

Primary sources: [paper](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf), [supplement](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf), [released model](https://github.com/brain-research/mpnn/tree/4a1f0ddea3cd7de5eebc96e509da2161624aaacd). Source snapshot and Apache license are in `sources/l081/`; byte hashes in `_sources_l081.json`.

## Reproduce the delivered work

From the repository root, using its `.venv`:

```bash
.venv/bin/python labs/_verify_l081.py
.venv/bin/python labs/_run_l081.py --preset smoke --seed 81 --output /tmp/l081-fresh-81
.venv/bin/python labs/_build_l081.py
.venv/bin/python labs/_execute_l081.py
.venv/bin/python labs/_delivery_l081.py
```

The complete model is `relkit/mpnn_l081.py`; the complete loader/trainer is `relkit/qm9_l081.py`. Both are inlined in the notebook. PyG supplies graph containers and batching, not model layers. The student reducer is called by the real model; an execution check records its calls.

`requirements-l081-observed.txt` records the author environment. The notebook installs RDKit 2025.9.6 and PyG 2.6.1 in Colab, retaining the runtime's PyTorch; that is a portable recipe, not the observed author environment. Live Colab and cross-version numeric parity have NOT_CHECKED status. Use recorded versions for numerical replay. Notebook smoke downloads the same hash-checked archive (~44 MB); it does not require a repository checkout.

## Protocol audit

| Axis | Paper/release | Reconstruction and remaining gap |
|---|---|---|
| Population | Paper states 130,462 | Modern QM9 SDF; public exclusions plus logged sanitization failures. Historical reader absent. Full modern population count NOT_RUN. |
| Features | Table 1; heavy vs explicit-H choices explored | 13 Table-1-style features, RDKit donor/acceptor definitions, implicit H. Historical chemical feature flags/variant-specific H setting cannot be recovered exactly. |
| Edges | Chemical graph, four bond types | Two directions per bond, no distances, no self-loops, no nonbonded edges. |
| Initial state | Zero-padded observed features | Width 50, zero pad; no checkpoint. |
| Message | Two bond-matrix banks | Reconstructed and checked against a dense independent contraction. Molecular class assumes an undirected graph. |
| Update | Bias-free released GRU equations | Explicit reset-before-matmul implementation; NumPy equation checks. Full TF runtime NOT_CHECKED. |
| Readout | GraphLevelOutput | Two independent MLPs on concat(final,original); sigmoid-gated value, graph sum. |
| Initialization | TF variable/utility initialization | Per-matrix Xavier and zero MLP biases; framework-specific initialization is a deviation. |
| Objective | Standardized-target MSE | Train-only mean/variance; historical fit scope unstated. |
| Optimizer | Adam; batch 20 | PyTorch Adam default beta/epsilon; clipping norm 4 from released defaults. Framework parity unverified. Final minibatch can be shorter. |
| Split | 10k validation / 10k test / remainder training | Fresh NumPy seed81 permutation; exact IDs saved. Historical IDs absent. |
| Search | 50 random trials per target/model; rounds 3–8, initial LR uniform [1e-5,5e-4], decay start [.1,.9], final factor [.01,1] | `paper-budget` reproduces those declared ranges, but fixes width50/readout200. Complete original search space/trial configurations unavailable. |
| Budget | 3 million updates, validation early stopping/selection | Up to 3m per trial; validation every5000; best checkpoint. No patience termination; original cadence unavailable. |
| Selection | Validation chooses checkpoint/model | All trials see validation only; test evaluated once after selection. |
| Metric | MAE / chemical accuracy | Dipole native-unit MAE; error ratio divides by .1. No ensemble for this row. |

## Three executable budgets, all explicitly reconstructions

- `smoke`: first512 valid molecules in file order; 410/51/51 train/validation/test, 40 updates, one fixed-width trial, six rounds. Seeds81/82/83 change both split and initialization. Executed locally.
- `closer`: first12,000 valid molecules; 9600/1200/1200 split, three trials ×10,000 updates. NOT_RUN. Still a capped, file-order subset.
- `paper-budget`: full available sanitized population, 10k validation/10k test, 50 trials ×3m updates. NOT_RUN. This large cost does not recover missing historical details.

```bash
python labs/_run_l081.py --preset closer --device cuda --output /tmp/l081-closer
python labs/_run_l081.py --preset paper-budget --device cuda --output /tmp/l081-paper-budget
# Optional remote smoke/closer operator; remote execution NOT_RUN:
modal run modal/l081_paper_repro.py --preset smoke
```

Use a fresh output directory. The driver preserves split molecule IDs, configurations, validation history, selected state dict, test predictions and SHA256 hashes. It rejects completed output directories. Interrupted searches have no exact optimizer/RNG resume implementation; restart into a new directory. The full budget should be scheduled on a durable GPU host; the bounded Modal operator supports smoke/closer only.

## Evidence boundaries

`evidence/l081/{81,82,83}/` contains fresh author smoke runs. Their test MAEs are 1.038806, 1.115341, and 1.112676 Debye. These establish that the complete reconstructed model/data/trainer pipeline executes, not historical score parity. No cross-model superiority claim is supported. Run variability changes both split and initialization, not three independent datasets.

`_verify_l081_results.json` checks the toy trace, empty neighborhoods, duplicate edges, relabeling, edge order, graph isolation, dense messages, source-equation GRU arithmetic and finite gradients. Passing these is not complete source-runtime parity.

Full historical result: **NOT_RUN / protocol gaps**. Every reconstruction remains **INCOMPARABLE**, even if a score approaches .394. Missing reader/trainer/search/split details are source gaps, not details that a larger run can silently solve. Browser/notebook/delivery status is recorded separately in their result files. Nothing here marks learner mastery or deploys the course.
