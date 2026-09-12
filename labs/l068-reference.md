## Built with TabPFN · computation card

Original source revision `a6e75afb82d13e7abb46deade3669c4106b3d636`; [license](../labs/sources/l068-v2/LICENSE.txt). [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/b2e2774c8e76afe191b5bf518f5cb727-Paper-Conference.pdf), §§3–4, Algorithm 1, Tables 1–2.

- **Two graphs:** H(time) generates correlated shifts for sparse causal relationships in G; G at one time generates many noisy rows. H is not the deployed model. The visible sampler is a paper-grounded reconstruction; the original pretraining generator was not released in the inspected tree.
- **Two kinds of training:** synthetic pretraining updates the transformer once during algorithm development. New-table inference holds weights fixed and conditions on context features, labels and times. Original 30,720,000-dataset pretraining is NOT_RUN here.
- **Time:** source min–max scaling permits τ>1; clip only to [−5,6]. A constant context uses denominator 1. Time2Vec has one affine coordinate plus 99 sine coordinates.
- **Actual architecture:** groups of two numeric values, time repeated/concatenated per group, 102→192 encoder (base 2→192; real NoT2V 3→192). One label token per row, censored queries. Twelve feature-attention/row-attention/GELU-MLP blocks; residual then non-affine LayerNorm. Six heads of width 32; query rows share first source K/V head. Query target → 192→768→10 head → K-class probabilities.
- **Protocol:** all rows in three real released datasets plus separate synthetic Blobs; three predeclared whole-domain cutoffs, source ID holdouts, paired original checkpoint IDs. Raw numeric, one view, fp32, common temperature 1. These choices differ from original random eligible splits, 3×3 repetitions and optimized 32-view wrapper.
- **Ablation:** drift_zero hides time in the same model. It is not the separately pretrained NoT2V model. The available NoT2V checkpoint is reused across cutoffs.
- **Evidence:** Chess worsens with drift; Parking improves but zero-time helps further; Blobs strongly favors temporal pretrained models. Accuracy, AUC and log loss can disagree. The 3-real-dataset bootstrap includes zero. Repetition SD is not dataset-level confidence.

## Five live operations and EXIT

`normalize_time(c,n)` · `time2vec(c,weight,bias)` · `shifted_weights(base,edge_to_relation,selected,shifts)` · `temporal_split(domains,source_count,seed,cap_per_domain=None)` · `dataset_summary(records)`.

`run_experiment(root,config=None,namespace=None)` uses those live definitions and strictly loads every pretrained tensor. Summaries are per dataset; only the next ranking step weights datasets equally. EXIT defaults to `labs/data/cache/l068-student/exit-v2.json`, binds current code/runtime/weights/source checks and refuses overwrite. A missing ID class makes full-class AUC unavailable, not zero.

[Source check](../labs/_paper_audit_l068_v2_results.json) · [Dataset/mechanism check](../labs/_check_l068_v2_results.json) · [Live identity check](../labs/_identity_l068_v2_results.json) · [Final predictions](../labs/_verify_l068_v2_results.json) · [Analysis](../labs/_analysis_l068_v2_results.json) · [Reproduction](../labs/l068-reproduction.md).
