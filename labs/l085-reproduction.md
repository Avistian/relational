# L085 reproduction contract

## Named experiment and evidence boundary

Li, Han & Wu (AAAI 2018), *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*, **Figure 2**, full Zachary karate-club setup. Primary source: https://arxiv.org/abs/1801.07606. Released code: https://github.com/liqimai/gcn/tree/3b30a2d35ca2b144bf0f36337f233d407a7e2dd6.

The named experiment is UNTRAINED. It has no trainer, accuracy target or train/test split. Adding training would change it. The complete model is visible in both notebooks. A complete loader/model/trainer is also visible for the separate Cora extension.

| Protocol dimension | Paper / release | This package |
|---|---|---|
| Data | Karate: 34 nodes, 78 edges, 2 classes | All nodes and binary undirected edges; embedded `sources/l085/karate.json` from NetworkX3.6.1; interaction weights ignored |
| Features | One-hot node identity | Identity matrix [34,34] |
| Normalization | Symmetric GCN | Add exactly one self-loop; normalize with augmented degrees |
| Architecture | Depths1–5, hidden16, output2 | All depths; no hidden layer at depth1 |
| Activations / biases | Figure prose incomplete; released models.py uses hidden ReLU, final identity; layers.py default no bias | Those release conventions; exact Figure2 driver not identified |
| Initialization | Glorot random initialization | Glorot uniform; PyTorch generator |
| Optimization / objective / schedule | Untrained | No optimizer, objective or updates |
| Dropout / inference | Untrained output visualization | eval mode, dropout disabled; linear coordinates before classifier softmax |
| Checkpoint / selection | Not specified for random picture | No selection or checkpoint; seed0 displayed in advance |
| Seeds / aggregation | Figure seed and weights not given | 100 local seed IDs0–99; RNG seed=10×seed+depth; retain every coordinate |
| Target | Qualitative class mixing illustrated across five depths | Scatter reconstruction; extra cosine distribution and pure-propagation diagnostics explicitly extensions |
| Historical equality | Original coordinates unavailable | INCOMPARABLE; do not infer exact figure parity from visually similar panels |

**Executed:** 500 full-graph untrained networks, all prescribed depths. This is a full setup reconstruction with disclosed historical gaps, not a recovered original figure. See `_paper_l085_results.json` for all coordinates, graph identity, software versions and code hash.

**NOT_RUN:** Li et al. classification Tables3–5 (co-training/self-training and other arms); original TensorFlow execution. They are different experiments from this lesson’s named target.

## Cora depth extension

Full L082 Cora data: 2708 nodes,1433 row-normalized features,7 classes. Fixed original140 training labels,500 validation labels,1000 test labels; remaining nodes contribute features/edges without supervision. Use the pinned tkipf/gcn data revision39a4089fe72ad9f055ed6fdb9746abdcfebc4d81 and SHA-256 files in `_sources_l078.json`. The loader verifies downloaded bytes before reading known upstream pickle files.

Depths `[1,2,4,8,16]`, seeds0–9, width16, bias-free Glorot-uniform layers, hidden ReLU, output logits. Adam lr .01, betas(.9,.999), eps1e-8; dropout .5 before each layer; cross-entropy on training nodes plus `.00025*sum(W_first**2)` (equivalent to .0005 times TensorFlow l2_loss). Maximum200 epochs. At zero-based epoch>10 stop when current validation objective exceeds the previous10 mean. Use the last weights, matching L082’s release stopping convention. Evaluate test labels only afterward.

This is a locally specified **extension**, not the Li release trainer: that branch penalizes all layers and exposes different split/selection options. Our L082-based diagnostic uses first-layer regularization, fixed split, locally declared seeds and dense elementwise dropout rather than sparse feature dropout. These draws have different random-number ordering. Depth changes parameter count and optimization behavior alongside mixing; the experiment does not isolate over-smoothing as the sole cause. No claim of L082 bitwise seed parity is made.

Report per-seed test/training accuracy, epochs and every validation objective; compute sample SD across10 initializations. No dataset-level statistical claim follows. Representation diagnostics use last hidden ReLU (output logits for depth1). Degree-corrected variance is scale-sensitive; pair it with mean cosine, RMS and the nonzero-row count. Cora has multiple components: a global variance need not vanish even when every component mixes internally.

## Exact commands from the repository root

Use Python3.12. The fully frozen fresh CPU runtime and observed author environment accompany the results. A new isolated Linux aarch64 Python3.12 environment passed all500 karate coordinate comparisons and a full200-epoch Cora depth2/seed0 replay with identical validation losses and metrics; see `_clean_environment_l085_results.json`. Notebook bootstrap uses the same four numerical package pins; actual live Colab compatibility is NOT_CHECKED.

```bash
python3.12 -m venv /tmp/l085-venv
/tmp/l085-venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r labs/requirements-l085-runtime.txt
/tmp/l085-venv/bin/python labs/_verify_l085.py
/tmp/l085-venv/bin/python labs/_clean_environment_l085.py
/tmp/l085-venv/bin/python labs/_run_l085.py --lane karate --seeds 100 --output labs/_paper_l085_results.json
/tmp/l085-venv/bin/python labs/_run_l085.py --lane cora --seeds 10 --output labs/_depth_l085_results.json
/tmp/l085-venv/bin/python labs/_figures_l085.py
/tmp/l085-venv/bin/python labs/_build_l085.py
```

All outputs are freshly recomputed; no resume cache or hidden checkpoint. Cora may take several minutes on CPU. Reduce seeds only for a separately labeled teaching run. The named karate experiment is inexpensive and runs in full by default. Both lanes ran locally; no cloud operator is required to access the full protocol.

The standalone notebook embeds graph bytes, source/data manifest, portable figures and complete implementation. Its default Cora run uses one seed per depth; `RUN_FULL_CORA=True` uses all ten. It does not require `relkit` imports. Author-reference scores in markdown are distinct from fresh kernel outputs.

## Validation and source ownership

`_verify_l085.py`: independent exact path coefficients, spectral limit, degree-scaled counterexample, disconnected-component boundary, scale invariance of cosine, zero-vector handling, full-forward and permutation checks. `_execution_l085_results.json` records executed solution identity and execution outcomes. `_delivery_l085_results.json` records browser and copied Pages staging checks. Live Colab and deployment remain NOT_CHECKED unless independently performed.

The source snapshots retain upstream MIT license. `_sources_l085.json` pins all inspected release files. The Cora run retains its original manifest bytes in `_sources_l085-executed.json`; the later manifest adds graph-source metadata without changing code or graph bytes. The graph is serialized and hashed; repeat runs use those bytes rather than a mutable graph-library default. `_sources_l078.json` pins the separate Cora data and preprocessing reference. Full result files fingerprint the executed model source; numerical tolerance is used for mathematical checks, not as a substitute for missing historical figure coordinates.
