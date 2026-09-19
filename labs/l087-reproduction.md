# L087 — reproducibility contract

## Named published target and actual evidence

Zhang & Chen (NeurIPS 2018), **Table 1: CN, AA and RA on all eight networks, ten random splits**. All 240 full-size evaluations were executed on CPU. This target covers three baseline columns, not the whole paper or the learned SEAL classifier. Published means and SDs are transcribed in `_targets_l087.json`; the complete measured table is in `_paper_l087_results.json` and the lesson.

The Python reconstruction uses the released data and deterministic documented protocol. Historical numerical parity is **INCOMPARABLE**, not MATCH: negative permutations differ from MATLAB, and the archived revision is not established as the paper-run revision. Small numerical differences do not resolve that gap.

Sources: [paper](https://arxiv.org/abs/1802.09691), [archived repository](https://github.com/muhanzhang/SEAL/tree/ca1f019a15fb0c21796042165b4e6bee73981dd3), `_sources_l087.json`. All 19 archived files have SHA256 records. The eight released graphs are symmetric, binary, loop-free; the loader preserves every node and edge. Original notices and author attribution remain in `sources/l087`.

## Protocol audit

| Dimension | Paper/released baseline protocol | Executed Python reconstruction |
|---|---|---|
| Data | USAir, NS, PB, Yeast, Celegans, Power, Router, Ecoli | Exact commit-pinned MAT bytes; no caps |
| Positive split | 90% observed / 10% held out; connected=false | Repeated uniform deletion from column-major upper triangle; floor((1-.9)*E) |
| RNG | Main.m calls rng(experiment), ten runs stated in paper | RandomState seeds 1–10; negative selection algorithm differs |
| Negatives | Uniform complement of complete static graph; no loops; balanced, disjoint train/test | Same policy; NumPy sampling without replacement, not MATLAB randperm |
| Scoring graph | Observed training positives, both directions | Same; all test edges absent |
| Architecture | CN=A², AA=A diag(1/log d) A, RA=A diag(1/d) A | Sparse products; zero nonfinite weights |
| Optimization | None: fixed heuristics | No trainable weights, initialization, HPO, selection or checkpoint |
| Metric | perfcurve ROC AUC | sklearn ROC AUC; independently audited by pairwise wins/half-ties |
| Aggregation | Mean and sample SD of ten runs per network | Same; actual sampled pairs and scores archived |
| Cross-dataset statistics | Not the target table's claim | Separate exploratory Friedman/Nemenyi over 8 dataset means |
| Original source execution | MATLAB + Statistics Toolbox | Native driver supplied, MATLAB runtime NOT_RUN |

All negatives are sampled from known static nonedges. This is a declared retrospective benchmark rule, not a deployment-ready temporal sampler. Equal integer seeds across software do not establish identical sampled data.

## Exact commands: Python reconstruction

Run from the `relational/` repository root. Author runtime: Python 3.12, CPU, one BLAS/PyTorch thread. Direct dependency versions are pinned in `labs/requirements-l087-runtime.txt`; `labs/requirements-l087-lock.txt` records the entire successfully installed clean environment.

```bash
python3.12 -m venv /tmp/l087-venv
/tmp/l087-venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r labs/requirements-l087-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/l087-venv/bin/python labs/_verify_l087.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/l087-venv/bin/python labs/_run_l087.py --track all
OPENBLAS_NUM_THREADS=1 /tmp/l087-venv/bin/python labs/_audit_l087.py
/tmp/l087-venv/bin/python labs/_figures_l087.py
/tmp/l087-venv/bin/python labs/_build_l087.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/l087-venv/bin/python labs/_execute_l087.py
```

`--track paper` runs the full named baseline experiment; `--track teaching` runs the three separate GCN teaching seeds. No preset silently changes the paper protocol. The replay overwrites its own result files; it does not resume stale runs. Default raw outputs live under `labs/results/l087/`; NPZ artifacts include full splits and scores. Result JSONs record source/implementation hashes, versions and artifact hashes. `_audit_l087.py` reconciles every default-location artifact using independent neighborhood intersections and pairwise AUC. A custom `--output` path is supported for running, but this default audit expects the default location.

Inside the self-contained notebook, `RUN_PAPER_REPRO=True` executes the same full baseline track. Its source and data manifest are inline; no private `relkit` import or project checkout is needed. Data downloads are checked before loading. The first cell supplies a pinned CPU Colab bootstrap. Live Colab remains **NOT_CHECKED**.

## Native released MATLAB replay

```bash
matlab -batch "run('labs/_original_l087.m')"
```

Requires MATLAB and Statistics and Machine Learning Toolbox (`perfcurve`). The driver calls the archived `DivideNet`, `sample_neg`, `CN`, `AA`, `RA` and `CalcAUC` directly, with seeds 1–10 and all eight datasets. It writes exact sampled pairs, score matrices and summary arrays under `labs/results/l087/matlab/`. This avoids Python's negative-permutation substitution. MATLAB is absent on the author machine, so the command is supplied but **NOT_RUN**, not verified as executable here. Original paper-run commit/environment identity remains unconfirmed even after a native replay.

## Separate teaching experiment

USAir: 332 nodes, 2,126 undirected edges. Seeds 87/88/89. Positive split 80/10/10; half of train positives are context, half are supervised. GCN has learned [N,32] node IDs, bias-free 32→32→16 maps, ReLU, context-only symmetric normalized adjacency with loops. Full-batch Adam .01, 150 epochs, no decay/dropout/HPO. Highest validation AUC selects the earliest tied epoch; test is read once after restoring the state. Fixed classification negatives are balanced and disjoint across splits.

Ranking scores both orientations of each test pair, each against 50 distinct uniformly sampled non-neighbor destinations. Full static truth is used only for filtering negatives. Average optimistic/pessimistic rank handles ties. MRR and Hits@10 average across queries. Recorded candidates make this auditable. It is neither an all-item metric nor a SEAL result, and differs from the baseline experiment's observed-edge budget.

## Verification and unrun ledger

- `_verify_l087_results.json`: undirected edge isolation, negative exhaustion, decoder values/gradients, tie handling, heuristic oracle and DRNL target removal.
- `_audit_l087_results.json`: all 80 split artifacts and 240 score/AUC reconstructions; 30 DRNL cases against the pinned authors' function.
- `_execution_l087_results.json`: solution execution from an empty directory with fresh downloads; exact match to seed87's entire canonical training/validation trace and test metrics, plus all 240 baseline evaluations with identical summaries. The student gate defaults off; the executed solution runs it.
- `_clean_environment_l087_results.json`: fresh environment installation, fresh eight-graph download, all 240 baseline evaluations with exactly matching split/score arrays, and the complete seed87 GCN trace.
- `_delivery_l087_results.json`: browser controls, portable figures, student blanks and copied Pages link checks, when run.
- Native MATLAB replay: **NOT_RUN** (MATLAB unavailable).
- Full SEAL DGCNN training, node2vec/attribute variants, other baseline columns and other paper tables: **NOT_RUN**. The curriculum assigns the paper as a skim; its trained classifier is outside this edge-decoder lab's implemented scope.
- Historical numerical parity: **INCOMPARABLE**.
- Live Colab and deployment: **NOT_CHECKED**. Local preparation is not publication.

The lesson is prepared teaching material, not a record of learner completion.
