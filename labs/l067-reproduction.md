# L067 · Full pretrained LoCalPFN: local reproduction contract

Active scientific source: [final NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/c40daf14d7a6469e65116507c21faeb7-Paper-Conference.pdf), Sections 2–5 and Appendix A.2/A.5. [Pinned official code and checkpoint](_sources_l067_v2.json), [lesson](../lessons/0067-local-pfn-retrieval-finetuning.html), [student notebook](0067-local-pfn-retrieval-finetuning.ipynb).

## What is implemented and measured

`relkit/localpfn_l067_v2.py` uses the immutable, fully visible `relkit/tabpfn_l062_v2.py` backbone: twelve complete blocks, width 512, four heads, FFN 1024, every original pretrained tensor. The LoCalPFN wrapper uses outer train StandardScaler/clip 10, exact Euclidean neighbors, zero padding to 100 features before source-layout context-only sample normalization, division by F/100 and temperature 1 softmax. The student notebook inlines both the backbone and local algorithm; all five TODOs drive its actual experiment.

The exact normalization implementation matches the official T×B×100 contiguous masked-reduction layout. An algebraically equivalent B×N×F mean/std implementation did not reproduce a real constant local feature in float32: source cancellation amplified by epsilon 1e-6 changed predictions. The corrected source check includes this fixture; `_normalization_l067_results.json` records the isolated precision diagnostic. This is not evidence that the amplified constant feature is scientifically desirable.

Author panel: all rows of diabetes/blood transfusion/WDBC, new stratified 80/10/10 split seeds 7/17/27; dynamic k=min(floor(10√N_train),1000), B=2, Q=16 per context, 30 updates for each of two predeclared learning rates .01 and 1e-5, AdamW weight decay .01, no scheduler, validation candidates step 0 and 30. Five arms: global-all, random-k, local-frozen, local FT paper rate, local FT released CLI rate. Global-all is an added control; random-k is a label-blind prefix without replacement, whereas the released vanilla helper uses class quotas and can sample with replacement. All targets, raw row IDs, neighbors, episodes, validation histories, predictions, selected/final weights and identities are retained.

Full dataset names occur in the paper roster, but the released TabZilla folds are not used. The numeric subset does not exercise categorical one-hot behavior. Paper budget 1000 queries, longer early stopping, 95 datasets/10 folds, tuned baselines, synthetic pretraining and original GPU runtime table remain NOT_RUN. The overall paper-result verdict is **INCOMPARABLE**.

## Fresh local commands

From repository root, with `requirements-labs.txt` installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l067_v2.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l067_v2.py --preset lab --output labs/data/cache/l067-my-lab.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l067_v2.py --preset closer --output labs/data/cache/l067-my-panel.json
.venv/bin/python labs/_build_l067.py
```

Each measurement refuses an existing output path. The `lab` preset is one complete diabetes split; `closer` is the three-dataset/three-seed panel. `smoke` is a separate small-context/two-update execution check. `paper` fails explicitly; it does not rename this panel into a paper replication. The generic foundation runner delegates to this same active version, while old `_verify_l067_results.json` remains a historical six-step experiment.

The notebook default saves `labs/data/cache/l067-student/exit-v2.json` in the local repository. It uses current notebook functions and checks their identity against its own source CHECK. It never copies the author identity into its live evidence. The optional post-EXIT gate repeats the broader panel with those same live definitions. The default experiment can take several minutes on one CPU thread; the corrected nine-record author panel took 1,676.65 seconds (27 minutes 56.65 seconds), including 540 full-model updates. No GPU is required.

## Public evidence and private runtime artifacts

- `_verify_l067_v2_results.json`: fresh complete panel, probabilities, targets, original IDs, model/code/runtime hashes and weight-file SHA256 values.
- `_check_l067_v2_results.json`: full official pretrained forward, all 152 gradient tensors, full AdamW step and real float32 normalization regression.
- `_analysis_l067_v2_results.json`: dataset-level summaries, sample SD, paired seed intervals, mean ranks, Friedman and Nemenyi statistics, interpreted limitations.
- `_normalization_l067_results.json`: independently generated exact constant-feature source precision diagnostic.
- `_evidence_l067.py`: parent-owned independent raw-data/geometry/episode/weight/official-prediction reconstruction.
- `_execution_l067_v2_results.json`: actual executed teacher-cell record; teacher notebook remains gitignored.

Adapted state dicts are saved under ignored `labs/data/cache/l067-weights/<run-id>/`. Each selected and final filename plus hash is in the result; these files are local execution artifacts, not published downloads. A fresh public runner reconstructs the states. Initial checkpoint download is immutable and SHA-checked before loading. The official LoCalPFN repository exposes no license file; its source is fetched into the local ignored cache for validation, not redistributed in this package.

A provisional full-k run made before the source-specific normalization correction was stopped and preserved in the ignored cache. No result from that operator is rehashed or relabeled as a corrected measurement.

## Further work needed for paper-result reproduction

First reconcile the actual experiment roster: final Tables 2/3 list only 37/42 names despite their 47/48 captions. `_paper_audit_l067_v2_results.json` preserves this independently counted discrepancy, with rendered pages inspected. Then use the exact TabZilla released roster, train/validation/test folds and preprocessing. Reconcile the paper's anchor exclusion with the released sampler that retains anchors; reconcile its total-query notation with the released per-context query-length flag. Fix the global training class axis before evaluating batches containing only one target class. Keep paper .01 learning rate and released 1e-5 discrepancy explicit. Match the reported longer training/evaluation and baseline tuning protocols, and compute the same dataset/fold aggregation and stratified-bootstrap IQM/mean intervals. Only then assess paper score tolerances.

The local/Colab notebook path retains adapted state files alongside its evidence. No Modal job was run; the older generic Modal helper is not advertised as an adapted-state export operator for this lesson. Its current result export does not retain this runner’s cached checkpoint files.

Local source checks, copied Pages access, browser layout, live Colab and deployed byte matching are independent verification statements. Learner mastery is unassessed until the completed EXIT explanation is reviewed.
