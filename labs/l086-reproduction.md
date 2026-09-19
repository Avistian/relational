# L086 reproduction contract

## Executed target and limitations

Full reconstruction of **Fey & Lenssen 2019, Table 1, GCN / Cora / fixed split**, with all 100 declared seeds 0–99 and the complete stopping schedule. Result: **81.305% mean; 0.714 percentage-point sample SD**, target 81.5 ± 0.6%. See `_paper_l086_results.json` for every validation trajectory, selected epoch, score and implementation/data manifest hashes.

The archived PyG1.2.0 revision `d5aff37604c8e3247f5e807f2ba0ec6eeb4c661b` is a release proxy; the exact commit used for the paper's table is not established. Modern PyTorch2.13/PyG2.8 kernels, reduction order and fresh seed identities differ. Historical parity **INCOMPARABLE**. Other table cells, random splits and GTX1080Ti runtime benchmarks **NOT_RUN**. This is one full-size experiment reconstruction, not reproduction of the whole paper.

## Exact commands

From the repository root, Python3.12 on Linux CPU:

```bash
python3 -m venv /tmp/l086-repro-env
. /tmp/l086-repro-env/bin/activate
python -m pip install -r labs/requirements-l086-runtime.txt --extra-index-url https://download.pytorch.org/whl/cpu
MAX_JOBS=2 python -m pip install --no-build-isolation 'git+https://github.com/pyg-team/pyg-lib.git@edc9e2a88d1c5d0953b5f69c98b8365597c6b699'
python labs/_verify_l086.py
python labs/_run_l086.py --seeds 100 --output l086-fresh-results.json
```

The tested host is Linux aarch64. pyg-lib is built from pinned source with its pinned git submodules; a C++ compiler and git must be installed. The activated environment places cmake/ninja on PATH. Do not silently ignore build/import failure: the full-graph experiment can run without pyg-lib, but the NeighborLoader exercise cannot. For a smoke run use `--seeds 1 --output l086-smoke.json`; its status explicitly says partial. No paid compute is required.

Colab runtime compatibility is **NOT_CHECKED**. Select a CPU runtime for this pinned CPU lane; install the same requirements and pyg-lib before restarting the kernel and executing the notebook. The notebook embeds the complete loader/model/trainer and data manifest; it has no repository-code import dependency. Hosted Colab requires these files to be published first; this task creates local artifacts only.

## Protocol audit

- Data: eight SHA-256-verified Planetoid files pinned by `_sources_l078.json`; 2708 nodes,1433 features,7 classes. Full graph; train/validation/test counts140/500/1000. Original test-index reordering and per-row feature normalization preserved.
- Graph: binary edges, duplicate collapse; exactly one self-loop per node. Symmetric augmented-degree normalization. The Cora adjacency is symmetric, so the old implementation's source-degree convention matches destination-degree normalization here.
- Network: 1433→16→7; Glorot weights, zero biases in both layers, ReLU; hidden dropout0.5 only. No input dropout. Full graph in every step.
- Optimizer: Adam lr0.01, coupled weight decay0.0005 over all parameters, defaults beta=(0.9,0.999),epsilon1e-8. CE on training nodes only.
- Selection: retain lowest validation CE. After epoch100, stop if current validation CE exceeds previous-ten mean; cap200epochs. Score restored checkpoint once on test. Historical code evaluated test each epoch but never selected on it; deferred read preserves that decision.
- Replicates: 100 locally declared initializations on one fixed dataset/split; report arithmetic mean and sample SD (ddof1). Replicates are not independent datasets.
- No claims of timing parity or original-environment execution. Matching accuracy is supporting evidence, not proof of protocol identity.

## Different evidence layers

`_verify_l086.py` checks dense output/gradient parity, orientation on a directed edge, permutation equivariance, isolated nodes, self-loop replacement, built-in GCNConv agreement, disjoint-union batching, HeteroData and a real NeighborLoader batch. It also checks full-Cora output parity with L082. These checks do not substitute for training.

`_run_l086.py` executes the historical PyG protocol reconstruction. It intentionally does not use L082's distinct trainer. No early-exit smoke cap is hidden in the full command.

`_build_l086.py` rebuilds lesson, notebooks, portable figure and reference. `_execute_l086.py` executes the complete default solution in a temporary directory and compares its seed0 trajectory to the author's full run. `_delivery_l086.py` checks browser controls and local links in copied Pages assets. Rebuilds overwrite notebook outputs; execute after rebuilding.

The student edits three functions that feed actual computations: edge weighting, seed supervision and ID mapping. Never treat prepared materials as learner completion.

## Fresh-environment evidence

`_clean_environment_l086_results.json` records a successful isolated Python3.12 replay with freshly downloaded verified data and exact seed0 training-trace agreement. This replay installed the wheel built from the pinned pyg-lib source on the same host; it did not repeat compilation independently. The lock file records all packages in that fresh environment except pyg-lib, whose source-build command is separate.

Final source/notebook integrity audit: `python labs/_audit_l086.py`. To also compare a new one-seed CLI trace: `python labs/_audit_l086.py --fresh-cli l086-smoke.json`.
