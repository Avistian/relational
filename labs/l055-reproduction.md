# L055 reproduction contract · repaired v2

Three evidence tracks are deliberately distinct:

1. **AUTHOR_REPORT_REANALYSIS:** 2879 original temporal-study reports from the pinned authors' repository, aggregated locally across all eight tasks and the actual Figure 2 arms. This does not retrain a model.
2. **LOCAL_PROTOCOL_MEASURED / INCOMPARABLE:** 54 newly selected evaluations (108 candidate fits) using corrected TabM-mini, MLP and XGBoost on three current-release tasks and one same-pool capped split pair. Authoring time 379.7 seconds of elapsed wall time on CPU; teacher notebook independently reruns it.
3. **Fresh paper training: NOT_RUN.** No full Figure 2 retraining or paid/cloud job was launched. Larger current-implementation local/Colab/Modal operators are provided and remain INCOMPARABLE to the paper.

## Primary sources and exact versions

Paper v4: https://arxiv.org/html/2406.19380v4. Read §3–4 (benchmark construction), §5.1–5.3 (technique transfer), §5.4/Figure 2 (split intervention), Appendices A (diagnostics), B (feature construction), C (preprocessing, tuning and comparisons). Paper scope is industrially engineered, temporally evolving tasks; it is complementary to other benchmarks.

Official code: https://github.com/yandex-research/tabred/tree/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32.
`_sources_l055.json` pins substantive source file hashes. The current downloader/preprocessing and the archived paper operators have different file layouts; downloading today's archives does not prove paper-era data identity.

The paper's temporal study uses MLP, MLP-PLR, XGBoost and TabR-S (shortened to TabR in the figure legend), three corresponding sliding/random windows and 15 initialization seeds. Archived temporal XGBoost tuning configs specify **200** trials and up to 4000 trees; neural MLP configs specify **100**, while the main paper mentions 100 for most methods and 25 for FT-Transformer. Inspect per-experiment configs rather than assigning all models a generic equal-budget label.

For the three local datasets, `paper/lib/data.py:transform_num` uses training-only noisy normal quantiles with 1e-5 fit noise and NaN-to-zero afterward. Other datasets use identity normalization because their numerical inputs are already normalized. Categorical features have a training-fitted vocabulary and unknown handling. The local median/standard transform and omitted categories are deviations.

Table 3's source ranking groups sorted models within the current group's leader SD (`get_ranks_ours`); Appendix C.2 also reports Tamhane's T2 comparisons. The notebook's ordinary average ranks and exploratory Friedman/Nemenyi summaries are different procedures.

## Author-report reconstruction

`_paper_reports_l055.py` reads only `paper/exp/temporal-shift-analysis/{model}/{dataset}-{split}/evaluation/{seed}/report.json` from the pinned commit. It validates dataset, split and seed against report configuration, verifies score orientation, verifies each file's committed Git blob and saves SHA256 of the original report bytes and normalized configuration. An edited tracked report or injected report is rejected. Main/default splits, tuning metrics and ensemble reports are excluded.

The manifest has 8 × 4 × 2 × 3 × 15 = 2880 expected positions; **XGBoost / Cooking Time / random-0 / seed 1 is missing**, leaving 2879. Do not impute it. Compute available-seed means per window, then average three windows equally. The artifact also computes a common-seed sensitivity: remove seed 1 from every Cooking window-0 arm and protocol. Other windows retain all 15 seeds.

`_paper_l055_results.json` contains the per-report metrics/provenance, 64 dataset/model/protocol aggregates and a second matched-seed analysis. Each includes window counts, means and conditional sample SD. Window minima/maxima are descriptive ranges, not confidence intervals. Overlapping windows and repeated initialization seeds are not independent future deployments.

Oriented XGBoost advantage over MLP-PLR is AUROC_XGB − AUROC_MLP-PLR or RMSE_MLP-PLR − RMSE_XGB. Positive always favors XGBoost. Its advantage narrows on seven tasks in the reanalysis; Homecredit increases .005617→.006265. Ecom changes +.001537→−.003569. Units differ across tasks, so retain separate task panels.

`_check_paper_reports_l055.py` independently reconciles every record against committed Git objects and every numerical summary using a separate aggregation implementation. This establishes extraction/aggregation correctness; the reports remain the authors' measurements. Source Figure 2 was visually compared, not pixel-identically regenerated. Original error-bar construction is not assumed to equal our explicitly defined window/seed summaries.

## Current released data

Public Kaggle `irubachev/tabred`, July 2026 release. `_fetch_l055.py` checks three archives against the pinned official registry. `_check_release_l055.py` additionally compares **80 extracted arrays/metadata files** against archive member bytes and existing `_data_l055.json` hashes; this closes the stale extracted-cache gap. The saved release check binds the measured v2 result hash and explicitly records that it was a post-run audit. The current notebook setup and v2 command operator verify extracted bytes before new runs.

| Task | Current rows | Predictor columns used | Omitted categorical columns | Paper Table 2 rows/features |
|---|---:|---:|---:|---:|
| Ecom Offers | 160057 | 113 numeric + 6 binary | 0 | 106K / 119 |
| Homesite Insurance | 260753 | 253 numeric + 23 binary | 23 | 224K / 296 |
| Sberbank Housing | 28321 | 365 numeric + 17 binary | 10 | 20K / 387 |

Timestamp metadata is excluded from predictors. Sberbank target is `log(price_doc / full_sq)`, confirmed in pinned preprocessing code. Its RMSE is not currency RMSE or original-unit MAE. Table 2 summary totals are not proof that data content changed. `_check_release_report_alignment_l055.py` compares current class counts to original pinned MLP report supports for two classification tasks across default plus all six random/temporal splits. **24 of 42 partition class-count comparisons match**: all 21 Homesite partitions and Ecom's three default partitions. All 18 Ecom random/temporal partitions have matching total sizes but different class counts; Ecom temporal-0 test is [13802, 6198] currently versus [13796, 6204] reported, and random-0 test is [15087, 4913] versus [15129, 4871]. This establishes partial alignment and a concrete unresolved Ecom membership/label discrepancy. It does not identify its cause or prove row/feature-byte identity for matching partitions. Full observations are in `_release_report_alignment_l055_results.json`.

## Current local experiment

The historical `temporal_experiment.py` / `tabm.py` / `_verify_l055_results.json` remain unchanged. Historical mini had an extra first output adapter, member backbone biases and incorrect fan-in initialization. Current teaching uses **`temporal_experiment_v2.py` and the corrected, frozen `tabm_v2.py`**, with newly measured `_verify_l055_v2_results.json`. No old score or source hash was relabeled.

For each released temporal window, sample train/validation/test once using label-blind seeds 550/551/552 and caps 1500/600/600. Concatenate those IDs; `paired_random_split` uses seed 5550+window to randomly assign the same pool back into equal-sized partitions. Thus **exact pool identity and partition counts match at the actual fitted scale**. The random assignment is a derived capped analogue of §5.4, not the downloaded `random-0` indices. In the old experiment the two independently capped pool unions could differ.

The released temporal groups are ordered nondecreasingly and share boundary timestamps on all three tasks. Preserve those indices and disclose ties. The separate live whole-timestamp splitter demonstrates a stricter half-open policy, at the cost of changed counts. Available metadata does not establish feature-arrival times or label delays; the multi-clock example is synthetic. This benchmark experiment assumes released feature/label suitability and cannot certify point-in-time deployment legality.

Train-only median imputation (zero if entirely missing), followed by filled-training mean/std scaling (unit scale if constant); apply frozen state to held-out rows. Neural regression labels are standardized using training targets and predictions restored before scoring. Score 1−AUROC or released-target RMSE, lower is better.

Three model seeds 0/1/2; two candidates per arm. MLP and corrected TabM-mini: two width64 blocks, ReLU, dropout .1; AdamW weight decay1e-4, batch256, 32 epochs; learning rates .001/.003. Mini has k8, one first input adapter, shared backbone biases/weights and independent small heads, mean member loss and mean probabilities/values. XGBoost: 120 histogram trees, learning rate .05, depths3/6, row/column sampling .8, one thread. Candidate counts match; compute/search coverage do not.

First minimum validation error selects a candidate, with neural checkpoints selected by validation during training. Only the selected candidate predicts test. No refit on train+validation. The notebook displays complete visible model/trainer code and proves the live random assignment, preprocessor and selector are called by an actual real-data smoke run.

## Local results and uncertainty

Current Ecom winner changes from TabM-mini to XGBoost; Homesite retains XGBoost; Sberbank changes from XGBoost to MLP. All Ecom errors increase and all Sberbank errors decrease under temporal evaluation. These are small conditional point-estimate comparisons, not stable architecture rankings.

Within each fixed local split, display mean ± sample SD over three model seeds. The plotted 95% Student-t interval is mean ±4.30265×SD/√3, conditional on rows, recipe and split. It omits new tasks, periods, feature reconstruction and protocol-selection uncertainty. Random/time seed labels do not create paired per-example test observations.

Average seeds within each dataset, then rank three arms. Mean ranks MLP/TabM-mini/XGBoost: random 2.667/1.667/1.667; temporal 2.333/2.000/1.667. Friedman p = .3679 / .7165; Nemenyi CD 1.9136. Three datasets have little power; nonsignificance is not equivalence. Raw predictions/targets, validation candidate errors, selected recipes, index/pool hashes and measured source identities are saved.

## Run, rebuild and extend

From repository root:

```bash
.venv/bin/python labs/_check_l055.py
.venv/bin/python labs/_check_l055_v2.py
.venv/bin/python labs/_check_release_l055.py
.venv/bin/python labs/_paper_reports_l055.py
.venv/bin/python labs/_verify_l055_v2.py --preset smoke
.venv/bin/python labs/_verify_l055_v2.py --preset windows-smoke
.venv/bin/python labs/_verify_l055_v2.py --preset local
```

Read-only author source checks use the extractor's ignored pinned checkout; the independent checker accepts `--checkout /path/to/pinned/tabred` (inspect its CLI). No authentication or paid compute is needed to reanalyze public logs.

Canonical current presentation: `_figures_l055_v2.py`, `_build_l055.py`, `_refresh_l055_v2.py`, `lessons/depth/0055.md`. Execute the teacher solution after code changes; refresh preserves outputs only when executable cells are exactly identical. Teacher notebooks stay gitignored.

After the EXIT, a larger current implementation can be run with:

```bash
.venv/bin/python labs/_verify_l055_v2.py --preset closer
modal run --detach modal/l055_paper_repro.py --preset closer
```

The gated Colab cell calls the student's live functions, with three temporal windows and their paired capped random assignments, 6000/2000/2000 caps, three seeds, 64 epochs and 300 trees. This studies window sensitivity but still omits original arms, categorical features, searches and data-version reconciliation. **Larger local/Colab/Modal runs NOT_RUN**. Modal source updated to v2; no job launched. Colab payload packaging is checked, live Colab UI is not.

Fresh paper training requires first reconciling the old `paper/lib` data format, exact samples, feature preprocessing and temporal split construction with archived configs; then running all four original arms, selected recipes and 15 seeds across all windows. `paper/README.md` documents the original environment and `bin/go.py` route. The current scale-up flag is not a fidelity certificate or a claim that this separate reconstruction was completed.

## Final local delivery

The final teacher executed all 29 code cells; 54 selected predictions matched the separate v2 reference. Eight image payloads, five blank student TODOs, live identities, the completed teacher EXIT, canonical regeneration and prepared HTML passed `labs/_delivery_l055_v2_results.json`. Its training-suite elapsed wall time was 1120.751 seconds in this notebook environment, versus 379.678 seconds for the separate reference operator; these are diagnostics, not a controlled efficiency result. The bounded three-window smoke also passed (18 selected evaluations, 12.8 seconds elapsed); the larger closer preset remains unrun.
