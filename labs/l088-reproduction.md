# L088 · Graph classification / GIN reproducibility contract

Named target: Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019, Table 1, MUTAG Sum–MLP (GIN-0), 89.4 ± 5.6%. This package targets one cell, not the full nine-dataset benchmark. Check `_paper_l088_results.json` for execution coverage; a runnable full search is not evidence that every configuration finished.

## Run from a clean checkout

Python 3.12, CPU. From the repository root:

```bash
python3.12 -m venv /tmp/l088-env
/tmp/l088-env/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r labs/requirements-l088-lock.txt
/tmp/l088-env/bin/python labs/_verify_l088.py
/tmp/l088-env/bin/python labs/_run_l088.py --workers 6 --output /tmp/l088-full-paper
/tmp/l088-env/bin/python labs/_teaching_l088.py
/tmp/l088-env/bin/python labs/_audit_l088.py --paper-dir /tmp/l088-full-paper
```

The full paper command performs **8 configurations × 10 folds × 350 epochs × 50 updates = 1,400,000 optimizer updates**. It can take hours on CPU. Every configuration uses all training graphs eligible for its fold; batches are random subsets without replacement within an update, redrawn at each update. Batch 128 does not mean one epoch is one pass.

The default output is `labs/results/l088/paper`. Completed fold runs may be reused only when the implementation, runner, runtime fingerprint and prediction checksum match. No mid-fold resume is implemented. The JSON and NPZ are not trusted merely because filenames exist. A changed fingerprint requires a new output directory. Do not mix evidence from different implementations.

Alternatively, open `0088-graph-classification.ipynb`, implement its three TODO functions, and set `RUN_FULL_PAPER=True`. The notebook contains the complete implementation and performs a fresh sequential search. The solution exposes all implementations. Its default run is a **two-epoch smoke check**, explicitly separate from both the 30-epoch teaching comparison and the full paper search.

## Protocol audit

| Element | Primary source | Reconstruction and limitation |
|---|---|---|
| Source | powerful-gnns commit `9a2ce8ac3e99278307093a464a95caf0fb04b602` | Relevant original files archived under `sources/l088`; hashes in `_sources_l088.json` |
| Dataset | Author `dataset.zip`, `dataset/MUTAG/MUTAG.txt` | All 188 graphs, exact bytes; local file included; missing file downloaded from pinned commit and checked against both archive and file hashes |
| Input | `util.py` | Seven one-hot atom tags; encounter-order tag/class encoding. No edge attributes. Vocab enumerated across released data, matching source; this is categorical vocabulary discovery, not fitted label statistics |
| Splits | `separate_data`, `StratifiedKFold`, split seed 0 | Modern scikit-learn splitter; all graph IDs published. Historical splitter version/fold IDs not recovered; numerical parity not established |
| Model | `models/graphcnn.py`, `models/mlp.py` | Four updates plus input; 2-layer MLP; internal and external BatchNorm; ReLU; sum aggregation; fixed epsilon=0 |
| Readout | Source `forward` | Sum per graph at depths 0..4; separate affine heads; dropout on logits per head; add head outputs |
| Grid | Paper §7 | Width 16/32, batch 32/128, dropout 0/.5. Eight configurations. Graph pool sum and neighbor pool sum |
| Seeds | `main.py` | Torch and NumPy training RNG seed 0 reset for each fold/config; split seed 0. Fold variability is not initialization-seed variability |
| Optimizer | `main.py` | Adam .01, default beta/epsilon, no weight decay, cross-entropy |
| Schedule | `main.py` + PyTorch v1.0.0 `_LRScheduler` | Explicit `.01 * .5**((epoch-1)//50)`: epochs 1–50 .01, 51–100 .005. Historical scheduler constructor resets `last_epoch=-1`; blindly running the old call order on modern torch shifts decay |
| Budget | Release CLI default | 350 epochs, 50 fresh random-subset updates each. Paper's exact maximum epoch search log absent; 350 is release-based reconstruction |
| Selection | README, “Cross-validation strategy in the paper” | Average validation curves across ten folds; choose one common epoch. Then maximize over configs. Earliest epoch and listed configuration break exact ties (declared choice) |
| Metric | Paper / README | Mean graph classification accuracy per fold, equal-fold aggregation. Unequal folds are not pooled by graph count. No untouched test |
| Dispersion | Paper reports SD | Historical denominator unspecified. We publish both ddof=0 and ddof=1; lesson uses sample SD, labels it explicitly |
| Numerics | Source tested on torch 0.4.1/1.0.0 | Modern CPU torch; `index_add` message sum instead of sparse matrix multiplication; sorted undirected edge order. Forward/gradient agreement checked at tolerances, not bitwise historical training parity |

The implementation retains the release's unused epsilon parameter for GIN-0 to keep state dictionaries comparable. Max graph readout is an extension; it is not part of this GIN-0 paper target.

## Evidence and independent audit

- `_verify_l088.py`: hand-computed message sums and gradients, grouped readout, common-epoch selection, permutation invariance, pinned authors' model forward and parameter-gradient oracle.
- `_audit_l088.py`: verify hashes and disjoint folds; reconstruct every validation curve directly from saved logits and labels; check the schedule, grid coverage, train/validation partition and summary selection.
- `_execute_l088.py`: execute the full inline solution from an empty working directory, including hash-verified download and fresh short training; never labels this full-search execution.
- `_delivery_l088.py`: browser interaction and mobile/desktop rendering, portable images, actual copied Pages staging and local links. Live Colab and deployed site remain `NOT_CHECKED`.

Raw paper traces: `results/l088/paper/*.json`. Each NPZ contains `[350, heldout_graphs, 2]` logits plus graph IDs and labels. These permit independent accuracy reconstruction at every epoch, not just checking the reported best number. Teaching traces live separately under `results/l088/teaching`.

## Teaching comparison

`_teaching_l088.py` compares graph readouts sum/mean/max. It fixes hidden width 16, batch size 32, no dropout, seed 0 and the same ten folds; budget is 30 epochs × 10 updates. Select a common epoch separately per readout. This comparison is not the paper's Sum/Mean/Max neighbor-aggregation comparison. Its selected CV values are exploratory and can be optimistic.

## Execution coverage, deviations and unrun ledger

Executed one predetermined configuration (width16, batch32, dropout0), all10 folds ×350 epochs:87.25% mean ±7.89pp sample SD. Full8-configuration search: **INCOMPLETE** (10/80 completed fold runs). Additional scheduled fold jobs were interrupted; partial-fold evidence was discarded. The remaining search can be run or resumed with the CLI. This is not full reproduction of the selected paper cell.

Historical numerical parity: **INCOMPARABLE**. Historical fold identity, winning configuration, runtime and exact training trace are unavailable. Modern execution is a reconstruction of the disclosed paper/release protocol. Original runtime replay, GIN-ε training, other eight datasets, all other Table 1 models, and a new outer-test evaluation: **NOT_RUN**. Full grid coverage must be read from the execution JSON rather than inferred from this guide.

## Rebuild lesson artifacts

```bash
.venv/bin/python labs/_figures_l088.py
.venv/bin/python labs/_build_l088.py
.venv/bin/python labs/_execute_l088.py
.venv/bin/python labs/_delivery_l088.py
node labs/_check_pedagogy.js
```

Build before executing the solution, since rebuilding intentionally clears cell outputs. Direct dependencies are in `requirements-l088-runtime.txt`; the validated full environment is frozen in `requirements-l088-lock.txt`; live Colab is separately unverified. `NOTES.md` records package authorship, not learner mastery.
