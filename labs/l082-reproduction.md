# L082 reproduction contract

Named published experiment: Kipf & Welling (ICLR 2017), Table 2, GCN, Cora fixed split. Paper target: 81.5% mean test accuracy across 100 initializations. This package executes that full-data/full-schedule experiment through a visible PyTorch port of the released implementation. It does not reproduce other datasets, baselines or depth experiments from the paper.

## Commands

From the relational repository root, with its lab environment:

```bash
.venv/bin/python labs/_verify_l082.py
.venv/bin/python labs/_isolation_l082.py
.venv/bin/python labs/_run_l082.py --seeds 100
.venv/bin/python labs/_figures_l082.py
.venv/bin/python labs/_build_l082.py
.venv/bin/python labs/_execute_l082.py
.venv/bin/python labs/_browser_l082.py
.venv/bin/python labs/_delivery_l082.py
```

For a fresh CPU environment, create a Python virtual environment and install numpy, scipy, torch, nbformat, nbclient, nbconvert, beautifulsoup4, matplotlib, networkx and ipykernel. `requirements-l082-observed.txt` records the exact author environment; package availability for that snapshot on other architectures is not asserted. The notebook is fully inline and embeds the source/data manifest; missing data is downloaded from an immutable upstream revision and verified before loading. No graph-neural-network package or pretrained checkpoint is required. A complete run takes several minutes on this CPU. `--seeds 1 --output _smoke_l082_results.json` is a diagnostic only, not the 100-run experiment.

## Provenance

Repository: https://github.com/tkipf/gcn
Revision: `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`.

`_sources_l082.json` records SHA-256 for the seven raw pickles, test index file, six reference source modules, and MIT licence. The original pickles are loaded only after hash verification. Source copies are in `sources/l078`; data is in `data/l078`. The manifest pins source identity, not a claim that the most recent repository state was the exact paper submission snapshot.

## Complete experiment audit

| Component | Contract / evidence |
|---|---|
| Dataset | Entire release Cora graph: 2708 nodes, 1433 features, seven classes |
| Input | Raw release feature arrays; restore test ordering; row-normalize nonzero feature mass |
| Graph | Binary undirected adjacency following dictionary links; identity added; symmetric augmented-degree normalization |
| Split | Released Planetoid split: first 140 labels train, next 500 validation, supplied 1000 sorted test indices |
| Architecture | Two GCN layers, 16 hidden units, ReLU; no biases; no pooled graph head |
| Initialization | Glorot uniform both weight matrices; no pretrained state |
| Objective | Mean cross-entropy over training nodes plus 5e-4 times half squared norm of first weight matrix |
| Dropout | Independent 0.5 dropout on nonzero input values and dense hidden activations; off for evaluation |
| Optimizer | PyTorch Adam, lr=.01, betas=(.9,.999), eps=1e-8; explicit L2 rather than optimizer weight_decay |
| Schedule | Full batch, at most 200 epochs |
| Stopping | Release rule: zero-based epoch >10 and current regularized validation loss > previous-ten mean |
| Checkpoint | Last weights at stop; no best-checkpoint restoration |
| Test freeze | Test labels consumed only after stopping; no test-based tuning or seed selection |
| Replicates | 100 initializations, locally declared seeds0–99; same fixed split |
| Metric | Accuracy over1000 test labels; mean, sample SD (ddof1), SE=SD/sqrt100 |
| Result | See `_paper_l082_results.json`; per-seed validation traces and final scores retained |
| Local reproducibility | Solution reruns same visible functions; exact per-seed score comparison with author output |

## Differences and unrun work

- Framework is PyTorch rather than original TensorFlow1. RNG streams, sparse kernels and Adam numerical semantics are not bitwise identical. Dense forward arithmetic checks do not establish cross-framework training parity.
- Original100 seed identities were not published; seeds0–99 are a reproducible local choice.
- Paper stopping prose and released stopping code differ. This port follows the code and last-weight evaluation; it does not silently claim the prose and implementation coincide.
- The original TensorFlow1 run remains NOT_RUN. Other Table2 datasets, competing methods and random-split experiments are NOT_RUN.
- No cross-dataset significance or GNN superiority claim follows from one fixed-split experiment. The reported SE captures only initialization variation.
- Live Colab and deployed Pages checks remain NOT_CHECKED unless separately recorded. Prepared HTML, notebook execution and copied staging are different checks.

## Evidence lanes

1. Synthetic hand fixture: augmented degrees, isolated self-message, relabeling, sparse/dense propagation, held-out label intervention and exact L2 gradient.
2. Numerical implementation: sparse graph forward pass versus independent dense matrix algebra using the same weights; no training parity implied.
3. Published-experiment port: full Cora protocol, 100 actual runs, explicit differences above.

Learner completion requires the three TODOs, hand trace and written EXIT explanation; generated artifacts alone do not count as mastery.

## L082 portability and result acceptance

The canonical implementation is `labs/relkit/gcn_l082.py`. The loader deliberately shares immutable raw data and `_sources_l078.json` with L078; the notebook embeds the same manifest and downloads/hash-checks missing bytes without repository imports. `_sources_l082.json` adds this lesson's implementation hashes. Attribution: adapted from tkipf/gcn, MIT; the pinned `sources/l078/LICENCE` is retained.

Measured full port run:100 seeds, mean81.401%, sample SD0.658 percentage points. The educational acceptance band is mean within±1 percentage point of81.5%; this is a declared course criterion, not the paper's statistical criterion. No test-based hyperparameter or seed selection. The executed solution repeats all100 runs and compares every score and stopping trace.

`figures/l082/hidden-seed0.npz` stores hidden states from a separate seed0 diagnostic and their PCA coordinates. Its colors use labels only after training. This figure does not enter selection.

For exact observed packages, use `pip install -r labs/requirements-l082-observed.txt` in a compatible Python3.12 CPU environment. This snapshot may contain platform-specific versions; installation in a fresh public environment is NOT_CHECKED. The Colab bootstrap installs numpy/scipy/torch and reports runtime versions through the result JSON; it is portable but not an environment-lock claim.

The curriculum suggests PyG; this lesson deliberately uses native PyTorch sparse operations so self-loops, normalization and masked loss are visible and editable. This changes the library, not the declared GCN architecture. The source-visible solution is downloadable at `labs/solutions/0082-gcn.ipynb`.
