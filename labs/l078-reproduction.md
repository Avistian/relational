# L078 reproduction contract

Named published experiment: Kipf & Welling (ICLR 2017), Table 2, GCN, Cora fixed split. Paper target: 81.5% mean test accuracy across 100 initializations. This package executes that full-data/full-schedule experiment through a visible PyTorch port of the released implementation. It does not reproduce every experiment in either assigned paper.

## Commands

From the relational repository root, with its lab environment:

```bash
.venv/bin/python labs/_verify_l078.py
.venv/bin/python labs/_run_l078.py --seeds 100
.venv/bin/python labs/_build_l078.py
.venv/bin/python labs/_execute_l078.py
.venv/bin/python labs/_browser_l078.py
.venv/bin/python labs/_delivery_l078.py
```

For a fresh CPU environment, create a Python virtual environment and install numpy, scipy, torch, nbformat, nbclient, nbconvert, beautifulsoup4, matplotlib, networkx and ipykernel. `requirements-l078-observed.txt` records the exact author environment; package availability for that snapshot on other architectures is not asserted. The notebook is fully inline and embeds the source/data manifest; missing data is downloaded from an immutable upstream revision and verified before loading. No graph-neural-network package or pretrained checkpoint is required. A complete run takes several minutes on this CPU. `--seeds 1 --output _smoke_l078_results.json` is a diagnostic only, not the 100-run experiment.

## Provenance

Repository: https://github.com/tkipf/gcn
Revision: `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`.

`_sources_l078.json` records SHA-256 for the seven raw pickles, test index file, six reference source modules, and MIT licence. The original pickles are loaded only after hash verification. Source copies are in `sources/l078`; data is in `data/l078`. The manifest pins source identity, not a claim that the most recent repository state was the exact paper submission snapshot.

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
| Result | See `_paper_l078_results.json`; per-seed validation traces and final scores retained |
| Local reproducibility | Solution reruns same visible functions; exact per-seed score comparison with author output |

## Differences and unrun work

- Framework is PyTorch rather than original TensorFlow1. RNG streams, sparse kernels and Adam numerical semantics are not bitwise identical. Dense forward arithmetic checks do not establish cross-framework training parity.
- Original100 seed identities were not published; seeds0–99 are a reproducible local choice.
- Paper stopping prose and released stopping code differ. This port follows the code and last-weight evaluation; it does not silently claim the prose and implementation coincide.
- The original TensorFlow1 run remains NOT_RUN. Other Table2 datasets, competing methods and random-split experiments are NOT_RUN. Full Gilmer QM9 architectures/experiments are NOT_RUN; Gilmer supplies the message-passing abstraction in this bridge.
- No cross-dataset significance or GNN superiority claim follows from one fixed-split experiment. The reported SE captures only initialization variation.
- Live Colab and deployed Pages checks remain NOT_CHECKED unless separately recorded. Prepared HTML, notebook execution and copied staging are different checks.

## Evidence lanes

1. Synthetic hand fixture: exact routing, synchronous update, empty neighbors, relabeling, two-hop reach and normalization checks.
2. Numerical implementation: sparse graph forward pass versus independent dense matrix algebra using the same weights; no training parity implied.
3. Published-experiment port: full Cora protocol, 100 actual runs, explicit differences above.

Learner completion requires the three TODOs, hand trace and written EXIT explanation; generated artifacts alone do not count as mastery.
