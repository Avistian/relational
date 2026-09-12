# L066: original TabICL inference and evidence contract

The repaired package uses the **original February v1 checkpoint**, all pretrained layers and a visible numeric inference wrapper. It does not reproduce synthetic pretraining or the original full benchmark. [Lesson](../lessons/0066-tabicl-column-row-attention.html) · [Notebook](0066-tabicl-column-row-attention.ipynb) · [Primary paper](https://arxiv.org/html/2502.05564v1).

## Active identities and source scope

- Checkpoint `tabicl-classifier-v1-0208.ckpt`, immutable HF revision `eaf789a9b25ee8486d6f48997ba076f850bbc30b`, SHA256 `f5bae1d31181a1bb4ab8e97d2a5e62a504e856f23b4ea62c7fcc2f8eec4995b6`, 108,313,922 bytes. Download URL and release source hashes: [_sources_l066_v2.json](_sources_l066_v2.json).
- Original inference source `tabicl==0.1.4`, wheel SHA256 `10dcaec46e9e2ba37adb3c1c386f53253f210bbc2894af2e6d0506131c0abdcc`. Vendored source and BSD license: [sources/l066-v2](sources/l066-v2/LICENSE).
- Canonical implementation: [relkit/tabicl_l066_v2.py](relkit/tabicl_l066_v2.py). Three full ISABs, three RoPE row blocks, twelve full ICL blocks, all normalization/projection/FFN/head weights. All 277 state tensors load strictly; 27,051,658 trainable parameter entries plus eight fixed RoPE frequencies.
- Supported wrapper: finite numeric arrays, context-only constant filter, standardizer and two-pass outlier softening, no optional power normalizer, one feature/class view, 2–10 classes, fixed temperature 0.9. Every pretrained layer executes; categorical conversion, many-class hierarchy, 32-view ensemble and automatic GPU batching/offload are not implemented here.
- Historical `_verify_l066_results.json`, old `foundation_core.py` skeleton, and later `v1.1-0506` scores remain unchanged. They are not the active model or fresh evidence.

The paper's §3.3 explicitly chooses RoPE base 100,000, matching both original checkpoint and release. Appendix C gives a generic 10,000 example. Release classifier `_load_model` explicitly identifies `v1-0208` as the paper checkpoint and `v1.1-0506` as a later version.

## Actual local protocol

All rows of diabetes (768×8), blood transfusion (748×4) and WDBC (569×30); seeds 7, 17 and 27. A stratified 20% query split is frozen. A seeded label-blind permutation of the remaining 80% defines nested prefixes at 12.5%, 37.5% and 100%. Largest contexts are 614/598/455 rows, with 154/150/114 fixed query rows respectively. The source files record exact IDs, context labels, targets, transformed feature roster, probabilities, logits and data hashes. Each prefix refits only its own wrapper statistics. No labels choose a context size or temperature.

The original model is copied once and remains frozen. Local inference uses float32, one torch CPU thread; source/model/loader/library identities are recorded in every result. Timing divides preprocessing, column, row and ICL stages; it excludes checkpoint download/load and verification. The 27 measurements took 9.81 seconds in this author run. These shared-workspace CPU timings are not a controlled comparison with TabPFNv2 or the paper's A100 timings.

Metrics are accuracy and log loss, with paired query IDs. Means and sample SDs aggregate three split seeds per dataset. Paired t95 intervals use three overlapping splits and condition on these fixed datasets; they do not express population uncertainty. Rank calculations first average loss within datasets and weight datasets equally. Friedman chi-square p≈0.0498 and exact rank-permutation p=6/216≈0.0278 are exploratory under their independent-dataset/exchangeable-rank null. Three convenience datasets cannot establish a universal context-size law.

## Execute fresh evidence

From the repository root, using the course venv:

```bash
.venv/bin/python labs/_check_l066_v2.py
.venv/bin/python labs/_verify_l066_v2.py --preset closer --output labs/data/cache/l066-my-fresh-run.json
.venv/bin/python labs/_analyze_l066_v2.py
.venv/bin/python labs/_figures_l066_v2.py
.venv/bin/python labs/_build_l066.py
```

The check prepares checksum-pinned reference dependency wheels in `labs/data/cache/l066-reference-deps` and executes the pinned reference source in a separate process. It does not install the source model into your current namespace. The analysis and figures above rebuild the **committed author** panel by default. To analyze a different result, call `_analyze_l066_v2.analyze(your_result)` explicitly; do not overwrite the author's measurement identity. `analyze` records the complete input payload hash.

`smoke` and `lab` both run all diabetes rows, seed 7, three context sizes. `closer` runs all three datasets/seeds. `paper` deliberately raises: original synthetic pretraining and the full heterogeneous benchmark protocol are not implemented by this numeric inference operator. Fresh output paths are mandatory; there is no resume or overwrite promise. A partial failure produces no completed-result JSON.

The notebook inlines the canonical model and experiment; five live TODOs feed all measured stages. The source CHECK, experiment and EXIT must have the same current-namespace identity. An independent actual-weight digest catches edits after measurement. A runtime fingerprint also binds each actual module type, forward method and mutable setting such as LayerNorm epsilon; instance forward overrides and hooks are rejected by this certified path. The stale-evidence checks cover a changed helper inside a generator expression and changed checkpoint weights. The module identity is kept distinct from notebook compilation identity; numerical correspondence is checked on aligned IDs. The EXIT saves `data/cache/l066-student/exit-v2.json` using exclusive creation. For another run, set `L066_OUTPUT` to a new directory before running the notebook.

For broader local or Colab execution, enable `RUN_BROADER` after EXIT. It uses your current live model and functions. The shared optional unattended operator is `modal run --detach modal/foundation_repro.py --lesson 66 --preset closer`; it runs this supported local panel after shared routing. No cloud job is launched or certified by supplying the command. Actual live Colab browser execution is NOT_CHECKED.

## What the checks establish

[_check_l066_v2_results.json](_check_l066_v2_results.json) compares complete column, row and final model outputs on multiple nontrivial copied-weight fixtures, then transformed numeric inputs and final wrapper probabilities. It also probes unrelated-query changes and appends. Independent [RoPE symmetry intervention](_symmetry_l066_results.json) and [optional power-wrapper coupling](_query_coupling_l066_results.json) are separate original-source experiments. The latter records a source-version/runtime-specific counterexample outside the supported none-view wrapper; a correct model mask alone does not certify every wrapper setting.

[_verify_l066_v2_results.json](_verify_l066_v2_results.json) is fresh complete-model inference, not a rehash of old May-checkpoint measurements. [_analysis_l066_v2_results.json](_analysis_l066_v2_results.json) reconstructs its probabilities/metrics, nesting, mean/SD, paired gaps, intervals and ranks. A separate NumPy/SciPy full-stage reconstruction appears in the independent lesson review, owned by the delivery reviewer.

## Paper claim / checked local scope / missing work

| Paper item | Checked here | Remaining boundary |
|---|---|---|
| §3 full architecture and original weights | Whole-stage/model source parity and fresh local scores | Floating-point tolerances and stated finite numeric inference path only |
| Figure 4 representation collapse | A fixed-weight RoPE-off source intervention on a four-row synthetic fixture | No retraining; no balance-scale Figure 4 reproduction |
| §4.1 and Appendix B synthetic SCM/tree prior | Source-grounded explanation | Prior generation and complete pretraining NOT_RUN |
| §4.2 / D.1 curriculum to 60K rows | Exact stages described | Three-A100 two-week training NOT_RUN |
| §4.3 many-class extension | Probability-chain derivation | Hierarchical inference and 12 many-class datasets NOT_RUN |
| §5 / E 188 supported-class datasets | Three complete small numeric datasets, one view, different splits | Full roster, paper splits, 32-view preprocessing/class/column ensemble and competitors NOT_RUN |
| §4.4 / D.2 500K×500 resource result | Stage arithmetic and tiny CPU timings | FlashAttention, stage batching, host/disk offload and measured 500K run NOT_RUN |

The Appendix E supported-class roster is 132 small plus 56 large datasets; the introduction's 55-large wording is inconsistent with that appendix count. The wider 200-dataset roster includes 12 tasks above ten classes. Paper inference uses 64/16/20 splits and 32 views; local inference uses 80/20 and one view. Full paper-result verdict: **INCOMPARABLE**, not MATCH. Packaging, source parity, browser delivery and learner mastery are separate assessments.
