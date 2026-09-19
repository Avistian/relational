# L089 · Cluster-GCN reproducibility contract

Named target: Chiang et al., KDD 2019, **Table 10, PPI test micro-F1 99.36%**. This is the deep PPI model, not Table 4's width-512 timing configuration. Source: `google-research/google-research` commit `89c16e403d42015c3133634788ed0b7965f56395`, `cluster_gcn/run_ppi.sh`. Original files and Apache license are archived under `sources/l089`; hashes are in `_sources_l089.json`.

## Run

From the repository root, Python 3.12:

```bash
python3.12 -m venv /tmp/l089-env
/tmp/l089-env/bin/python -m pip install -r labs/requirements-l089-runtime.txt
/tmp/l089-env/bin/python labs/_verify_l089.py
/tmp/l089-env/bin/python labs/_run_l089.py --preset smoke --output /tmp/l089-smoke-new
/tmp/l089-env/bin/python labs/_run_l089.py --preset paper --device cuda --seed 1 --output /tmp/l089-paper-new
/tmp/l089-env/bin/python labs/_audit_l089.py --run /tmp/l089-paper-new
```

`smoke`: 2 layers, width32, 2 epochs. `closer`: 3 layers, width128, 30 epochs. **`paper`: 5 layers, width2048, 400 epochs, all 44,906 training nodes, 50 partitions, one partition per update.** The first two are different experiments, not partial proof of Table10. All presets use the complete loader and trainer. A fresh output directory is required; unchecked reuse/resume is intentionally unavailable.

The notebook exposes the same complete implementation. Set `RUN_PAPER_REPRO=True`, choose `PAPER_PRESET='paper'`, and select a GPU runtime. The full run can exceed a free Colab session; local disk artifacts are not durable across a runtime reset. The CLI is preferable on a durable GPU host. Local CPU supports the same command, but the full width is expensive.

Remote operator (requires a Modal account permitted to use GPUs):

```bash
.venv/bin/modal run --detach modal/l089_paper_repro.py --preset paper --seed 1
.venv/bin/modal volume ls l089-cluster-gcn-evidence
.venv/bin/modal volume get l089-cluster-gcn-evidence <run-directory> /tmp/l089-remote
```

The remote image pins PyTorch2.8.0, NumPy2.2.6, SciPy1.15.3, scikit-learn1.7.1 and PyMetis2025.2.2. Local measurements use the same pinned model runtime; the complete clean environment is frozen in `requirements-l089-lock.txt`. Figure/build tooling may use the workspace environment. GPU sparse arithmetic, RNG and runtime differ from both local CPU and historical TensorFlow. Run metadata records actual versions and source hash.

## Protocol audit

| Element | Release behavior and our implementation |
|---|---|
| Data | Full Stanford GraphSAGE PPI zip; SHA256 `53aeb76e54fd41b645e7edb48b62929240b89839495396b048086fd212503fbd`; 56,944 nodes,50 inputs,121 binary outputs |
| Split | Original node flags:44,906 train /6,514 validation /5,524 test; no new random split |
| Scaling | StandardScaler fit only on training nodes; transform all features |
| Graph | Legacy node-link endpoints index the nodes list; undirected Graph edges, adjacency plus transpose. Existing self-loops get weight2 in raw adjacency |
| First layer | **Raw** `[A_train X, X]` cached before clustering; no row normalization or diagonal enhancement here. Training cache uses training-induced graph only. Evaluation cache uses full graph |
| METIS | Partition training-only graph into50 parts, seed1. Modern PyMetis2025.2.2, explicit nonrecursive k-way. Legacy wrapper/library versions and original partition IDs unavailable; save actual IDs |
| Later supports | `partition_graph` rebuilds q=1 supports with unit weights, including existing self-loops. Add identity, row-normalize, then add lambda1 times normalized diagonal. Thus existing loops are weight2 before normalization, not weight3 |
| Multi-cluster extension | Slice original adjacency on the union, restoring inter-part edges. Released q>1 path retains raw weights. The controlled teaching experiment binarizes raw adjacency for every arm to remove this confound |
| Model | Five layers; four hidden outputs of2048; concatenate neighbor and self branches; Glorot uniform weights; no affine biases; hidden per-node LayerNorm (population variance,epsilon1e-9,scale1,offset0), then ReLU |
| Dropout | .2 on concatenated layer input, including cached first-layer input and final layer input; disabled at inference |
| Output/loss | 121 logits; elementwise binary cross-entropy averaged over eligible nodes and labels |
| Optimizer | TF1-style Adam lr .01,beta(.9,.999),epsilon1e-8 before second-moment bias correction; zero weight decay. Inline implementation preserves this epsilon location |
| Epoch | Visit all50 parts; q=1 release shuffles partition order twice; NumPy RandomState seed1. One optimizer step per part |
| Budget/selection | 400 epochs; release patience1000 cannot trigger; final weights, **not** best validation checkpoint. Validation each epoch, no test selection |
| Validation | Full-graph first-layer cache; full graph partitioned into2; loss/F1 on validation nodes only |
| Test | Full graph as one partition; final model evaluated on CPU, once after training. Threshold logits strictly >0; micro-F1 pools node-label decisions |
| Seeds | Explicit fresh model seed1; repetition seeds2,3 are declared extensions. Original TF graph seed is set after model construction, and exact initial weights are absent; cross-framework RNG identity is not recoverable |
| Timing/memory | Training timer excludes partitioning and validation; report preprocessing separately. Dense-state proxy is `4 × max_batch_nodes × hidden × (layers−1)` bytes, **not measured peak RAM/VRAM**. GPU runs additionally report allocator peak per training epoch |

## Evidence and gaps

`_paper_l089_results.json` is the execution ledger. A runnable recipe, uploaded image, successful smoke run or matching normalization does not mean the full target was executed. Historical parity remains **INCOMPARABLE** because the exact original partitions, initial weights, runtime and training trace are absent. PyTorch/PyMetis is an explicit release reconstruction; the original TensorFlow1 execution is **NOT_RUN**. Reddit, Amazon2M, all paper baselines and paper timing/memory comparisons are **NOT_RUN**.

`_verify_l089.py` checks restored cut edges/local indexing, archived original normalization functions, independent dense model arithmetic, gradients, Adam epsilon arithmetic, strict F1 threshold, held-out-feature/edge/label interventions and source hashes. This is not whole-model TensorFlow parity.

`_audit_l089.py` independently recomputes F1/loss from saved test logits, verifies prediction checksum, partition coverage, epoch count and config. Fresh runs write per-epoch curves, global train IDs, local partitions, final test logits/IDs/labels, weights and runtime/source metadata. The committed teaching evidence includes prediction traces and partition IDs; regenerable `model.pt` checkpoints are ignored.

## Teaching experiment

`_teaching_l089.py` runs all PPI nodes with fixed binary raw adjacency,2 layers,width32,10 epochs,seeds1/2/3; full-batch,random50,cluster50 q1,cluster50 q5. Same graph,split,features,model,optimizer and passes; **different optimizer-step counts** (10,500,500,100). It measures the combined training strategy, not an update-matched causal effect of clustering alone. It is a single-dataset demonstration, not a universal sampler ranking. Test scores are reported for all prespecified arms; no winning arm is selected and retrained using test results. One-hop cached features cross training-cluster boundaries; later hidden propagation remains restricted.

```bash
.venv/bin/python labs/_teaching_l089.py --output /tmp/l089-teaching-fresh
.venv/bin/python labs/_figures_l089.py
.venv/bin/python labs/_build_l089.py
.venv/bin/python labs/_execute_l089.py
.venv/bin/python labs/_delivery_l089.py
```

Build before execution; rebuilding clears solution outputs. Live Colab and deployed site are separately **NOT_CHECKED** unless the delivery report explicitly says otherwise. Creating the package does not mark learner completion.
