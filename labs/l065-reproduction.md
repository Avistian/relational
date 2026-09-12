# L065 — pretrained query extraction and linear-head reproduction contract

The active lab is a new versioned experiment. It uses the complete historical TabPFN v2 classifier checkpoint, ten training folds, all 12 intermediate target states, and exhaustive validation-selected concatenation of up to three layers. It is a faithful implementation of the central query-role extraction mechanism, with explicitly completed local protocol choices. The original paper's 29-dataset result is **NOT_RUN / INCOMPARABLE**.

## Primary source and implementation boundaries

Read Ye, Liu and Chao, [arXiv 2502.17361 version 1](https://arxiv.org/html/2502.17361v1), all §6, Figure 5, Table 2, Appendix C Table 10, §4.1 evaluation, §5 token interpretation, §7.3 scale extension and Appendix D wrapper analysis. This lesson targets that version; it does not claim all later paper revisions were audited.

The 29 classification tasks are from **Tiny benchmark 2**, not the different Nature-paper task roster. Table 2 compares native, vanilla, layers 6/9/12 and validation-selected combinations. It does not say that the combined model wins on every task. Parent-owned `_paper_l065_v2_results.json` parses every published Table 10 row and distinguishes published ranks from reconstruction using rounded scores. Combined is better than native on 16 rows, tied on 4 and worse on 9. Rounded table values do not uniquely determine the published aggregate ranks; the underlying ranking precision and procedure remain unestablished.

No original §6 experiment repository was established by the targeted paper/author/code search. We therefore do not invent a matching original search algorithm. [Current official embedding documentation](https://docs.priorlabs.ai/capabilities/embeddings) describes fold extraction and a full-training final model for unseen data, supporting our practical context choice as a later implementation comparison; it is not evidence of the exact Table 10 operator.

The checked release is `tabpfn==2.0.9`, wheel SHA-256 `04e3bb989e9328d510ea4fccb6c6a36c8269630685d460af4df5eb268206bf21`. Its checkpoint is `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`. The independently written full model is unchanged `relkit/tabpfn_l064_v2.py`, SHA-256 `6f50292d3cb41597e217a4e6a7f71eb6bc120189f6295bd5975a7ec8dc75e747`.

`utils.py::_get_embeddings` selects context or query states while the supplied X remains the query batch. `model/transformer.py` selects the last group token before the decoder. `_reference_l065_v2.py` hooks all 12 actual source blocks and separately calls the public embedding API. `_check_l065_v2.py` checks the learner's live full model and extraction helpers against those states. This is one numeric fixture with missingness and binary classes, not a claim of exhaustive source parity across all configurations; broader model checks are in L064.

The release adds projected random group identities after value encoding. Ye §5 Eq. 2 is a schematic feature-level multiplication and must not silently replace that computation. Parent-owned `_token_identity_l065_results.json` checks an original-source zero-normalized-value fixture: the first block receives the added group vector even when values are zero. The fixture is a low-level encoder check; a full raw wrapper removes fully constant columns.

## Exact local experiment

| Item | Active operator |
|---|---|
| Weights | Full 12-layer, 192-wide, 6-head pretrained v2 classifier; all 81 tensors |
| Data | All rows: diabetes 768×8, blood transfusion 748×4, WDBC 569×30; binary numeric lane |
| Outer split | Stratified 80/20, then split the 80% portion 80/20; approximately 64/16/20; IDs saved |
| Repeats | Author seeds 0,1,2 per dataset; student seed 7 on diabetes |
| Extraction | Stratified 10-fold training split; complementary contexts; fixed group seed across folds |
| Evaluation | All outer training rows are support; validation/test are separate fixed query batches |
| Backbone preprocessing | One numeric view, context constant removal, internal mean replacement/sample normalization/flags; no external transform, fingerprint, class/feature permutation or outlier transform |
| Layers | Complete block outputs, one-based 1–12; target token only, before native head |
| Head | Train-fitted StandardScaler; binary L2 LogisticRegression(liblinear), tol 1e-6, max_iter 2000; convergence enforced |
| Combination search | All 298 nonempty subsets of 1–3 layers × C in 0.1,1,10 = 894 fits; local completion of unspecified search details |
| Selection | Highest validation accuracy; exact ties fewer layers, smaller C, lexicographic layer tuple; no test metric in candidates |
| Refitting | Selected head fit on the same outer training representation; validation not promoted to training |
| Arms | Raw-feature LR with training imputation; vanilla context-state layer 12; query layers 6,9,12; combined query layers; native head |
| Native probabilities | Active binary logits /0.9 then softmax; same full training support as evaluation embeddings |
| Metrics | Accuracy primary; test log loss and ROC AUC secondary; all probabilities and targets saved |
| Timing | Extraction includes ten training fold calls and two evaluation calls; selection includes heads and metrics; excludes loading/source checks |

The support coverage is approximately 90% for head-training representations and 100% for evaluation. This is a remaining context distribution difference. Fold exclusion guarantees the own-label route is absent with fixed memberships; it does not guarantee identical embedding distributions or an unbiased fitted-head training score. Original-wrapper query coupling remains possible before attention, so query batches and all identities remain fixed for the label intervention.

## Run locally

From the repository root:

```bash
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l065_v2.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l065_v2.py --preset closer --output labs/data/cache/l065-new-results.json
.venv/bin/python labs/_figures_l065_v2.py
.venv/bin/python labs/_build_l065.py
OMP_NUM_THREADS=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 labs/solutions/0065-tabpfn-query-embeddings.ipynb
```

`--preset lab` runs the complete diabetes experiment. The verifier refuses an existing output path; choose a fresh `--output` destination. New results remain separate from published author evidence; the figure builder continues to render the published `_verify_l065_v2_results.json` unless you deliberately establish a replacement reference. There is no random or reduced `smoke` lane; a shared smoke route maps explicitly to the full lab. The author verifier runs independent datasets in up to three local processes, each with one torch thread. It does not launch a cloud service. Full-data search takes several minutes per dataset, depending on CPU and concurrent workloads.

The historical wheel is checked before extraction into `data/cache/l064-source/official`; old sklearn 1.6.1 is installed there for the subprocess only. The normal notebook kernel retains its own sklearn version. The full checkpoint is downloaded and checksum-verified if missing. **Built with TabPFN**; see [Prior Labs License](sources/foundation/v2-LICENSE).

The builder API is `build_package(notebooks=True, render=True)` and `build(solution=False)`. Student notebooks contain blanks with no outputs. Teacher solutions remain ignored locally. Prepared HTML is a read-only student preview with an exercises anchor. Portable images are inline data URLs and have full-size local links.

## Live student evidence and rejection rules

The visible backbone and four TODOs execute in the notebook namespace. `run_experiment(ROOT, model, config, globals())` consumes those functions. Its identity follows function globals inside nested code objects, records defaults/model methods, live loader SPECS/CACHE, protocol, versions and backbone constants; actual model tensors get a separate digest. The source-check kernel identity must equal the measured result's and the current kernel's. EXIT also checks the current source-checker/worker file digests. There is no resume path.

The student writes `data/cache/l065-student/run-<timestamp>.json` and `student-l065-v2-exit.json`. EXIT requires the ten-fold ledger, N×12×192 training shape, all 894 candidates, split exclusion, source checks, weight identity and an interpretation. A helper/default/config/weight change means rerun source checks and measurement. Do not replace the saved identity to make an old score pass.

Author output is `_verify_l065_v2_results.json`; source checks are `_check_l065_v2_results.json`; source/code inventory is `_sources_l065_v2.json`. The paper table audit is a separate frozen-score reanalysis, not newly trained reproduction evidence. Browser and publication checks are tracked by the parent delivery workflow; local execution does not establish live Colab or public deployment.

## Preserved history and outstanding reproduction work

`_verify_l065.py`, `_verify_l065_results.json`, the old reduced shared foundation operators and their figures are historical and unchanged. They used three folds, final-layer extraction and average evaluation contexts. New scores do not inherit those operators' identities.

To reproduce the paper table, first establish the exact 29 Tiny benchmark 2 dataset versions/splits, all 15 seed definitions, wrapper views, logistic-head/scaler settings, layer-search procedure and validation/test context policy. Then extend this same extractor to those exact numeric/categorical inputs and execute that protocol. No preset name supplies these missing facts. Synthetic pretraining is outside this paper's feature-extraction experiment and has not been rerun.
