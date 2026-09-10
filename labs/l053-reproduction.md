# L053 reproducibility contract

Scope: complete numeric RealMLP-TD-S forward path, binary classification/regression and a visible training recipe. Full TD, categories, embedding paths, bagging and refitting are excluded. This is not the full RealMLP benchmark.

## Primary sources

- Paper: https://arxiv.org/html/2407.04491v3 (§2, §3, Table A.1, Appendix A.2).
- Standalone repository revision: `a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e`.
- Original author files and MIT notice: `sources/l053/`; hashes: `_sources_l053.json`.
- `_source_check_l053.py` executes the unmodified author preprocessing, scaling, NTP and Mish classes and schedule expression. Full numeric forward/input-gradient comparisons copy nonzero weights across three hidden stages for classification and regression. Output max error < 5.4e-7, input-gradient max error < 1.7e-8. The extended audit also checks all parameter gradients and four matched-batch Adam updates and moment states, for both task types, using the pinned author optimizer construction and group assignments. Maximum parameter-gradient error is 1.79e-7; maximum updated-parameter error is 2.39e-7; maximum moment-state error is 2.99e-8. This bounded optimizer check does not check full shuffled training, RNG consumption, checkpoint selection or pytabkit training parity.

## Data and local protocol

Reuse the verified TabR archive files and byte-range fetcher `_fetch_l052.py`, with hashes in `_data_l052.json`. These are real California Housing, House 16H and Higgs Small arrays. They are a practical substitute for the full RealMLP benchmark archive; common dataset names do not establish split equivalence. The RealMLP meta-train/meta-test suites and full Grinsztajn suite remain unrun. The teaching cache is compact and already available; exact RealMLP benchmark split acquisition is deferred, not claimed impossible.

Released TabR splits retained. Label-blind row caps 1200 train / 600 validation / 600 test; subsample seeds 53/54/55 respectively; model seeds 0/1/2. Selected row indices, raw file hashes and package versions are saved in `_verify_l053_results.json`. Median imputation and robust scaling fit training rows only. Regression target statistics use training labels only, matching the authors' standalone code; the paper wording uses train+validation labels. Numeric missing-row removal is not mirrored.

Neural model: 3×64 hidden units, SELU class./Mish reg., learned input scaling, NTP, standard-normal hidden initialization and zero head. Adam β=(.9,.95), epsilon 1e-8, no decay/dropout; bases .04/.07 and group factors 6/1/.1. Four coslog cycles compressed into 64 epochs, batch 256, incomplete batch dropped like standalone. All epochs run; last best validation error selects checkpoint. Two logits for binary classification, cross-entropy label smoothing .1; MSE for regression, no output clipping.

XGB-fixed is the first candidate, depth3/rate.03, not the library default. XGB-tuned selects among depths {3,6} × rates {.03,.1,.2}, 150-tree cap, validation patience20, subsample/column sample .8, hist algorithm, one thread. Same raw imputed values and splits for every tree candidate; no neural transform for trees. Candidate selection uses validation error; fixed and selected candidates are evaluated on test after selection. This compares different search procedures and does not equalize compute. Search records are preserved. The fixed arm is reused within the tuned arm's search.

Errors: original-unit RMSE or classification error; lower is better. Three-seed means, sample SD, and 95% t intervals are conditional on fixed splits. Across three tasks: mean ranks, exploratory approximate Friedman test, Nemenyi CD. Three seeds are not three datasets. No cross-unit mean RMSE is reported. Local experiment runtime about 30 CPU seconds.

## Larger evidence and operators

`_paper_repro_l053_closer_summary.json` records a completed CPU run: California 6000 training rows, full released validation/test, width256 and 256epochs, three seeds. Multiple conditions change from the learning run; this is not a controlled width-only or data-only ablation. No tuned-XGB larger comparison is claimed.

`_paper_repro_l053_measured.py` archives the exact operator used for this measurement, matched by `contract.operator`. The current operator only corrects the ledger's dataset/metric presentation from a benchmark-level label to California/RMSE. The exported measured summary retains its original contract and explicitly records that display correction; numerical evidence is unchanged.

Current operator: `_paper_repro_l053.py`. Presets:

- `smoke`: 160/64 row caps, width16, 2epochs, one seed, California only.
- `closer`: 6000 train rows, full released validation/test, width256, 256epochs, three seeds.
- `paper`: full released California split, width256, 256epochs, ten **model seeds**, not ten new data splits. This restores model settings only; full benchmark remains NOT_RUN.

All presets remain INCOMPARABLE to the paper benchmark. There is no valid single California numeric target for its benchmark-wide superiority claim. A MATCH tolerance would be misleading before aligning datasets, splits and aggregation.

From repo root: `python labs/_paper_repro_l053.py --preset closer` (use a new `--out` directory if code/data/environment changed). Modal: `modal run --detach modal/l053_paper_repro.py --preset closer`. The Colab gate is OFF by default and explicitly passes live model, preprocessing and runner objects. Resume boundaries are completed seeds. The identity includes reachable lesson helpers, configuration, data hashes, package/Python versions and device. Same-name edits are rejected. Interrupted partial seeds rerun.

## Rebuild and checks

Run `_check_l053.py`, `_source_check_l053.py`, `_verify_l053.py`, `_figures_l053.py`, `_build_l053.py`, then execute the solution and render student HTML via `_render_l053.py`. `_delivery_check_l053.py` audits numerical provenance, live-code paths, notebook blanks/images, links and copied Pages staging. Figures show author snapshots alongside separate live notebook plots.

Browser, live Colab UI, Modal execution and Pages deployment are separate from local generation. Their actual status is recorded in `_delivery_l053_results.json`; local prepared HTML does not establish either Colab rendering or remote publication.


## September audit: separately measured component intervention

`_ablation_l053.py` reruns the unchanged local neural baseline and a robust-scaling-only subclass that removes just smooth clipping. Both retain width64/64epochs, the same row selections, model seeds, fit-only statistics, target scaling, optimizer groups, schedule and validation selection. No learning-rate retuning. `_ablation_l053_results.json` stores both arms' predictions, validation histories, source/data hashes, versions and paired-seed summaries. All baseline predictions match the historical local artifact exactly. The measured baseline and intervention took 62.6 and 55.7 CPU wall seconds in this audit environment; runtimes vary and are not a speed comparison.

Mean robust-only minus standard error (three paired seeds): California +0.00912 RMSE, conditional 95% interval [−0.06332, 0.08156]; House +195.98 RMSE [−10912.49, 11304.44]; Higgs +0.01111 classification-error fraction [−0.05543, 0.07766]. Every interval includes zero. They describe model-seed variation on fixed capped splits, not new-split or new-dataset uncertainty. The three positive means are not sufficient evidence of a consistent benefit.

The related primary target is paper v3 Appendix B.2, Table B.2: removing smooth clipping increases shifted-geometric error by 0.5% [−0.4,1.4] on meta-train classification and 9.5% [4.4,14.8] on meta-train regression. Those are relative benchmark-score differences over ten random splits, not our original-unit paired differences. This targeted mechanism experiment improves interpretability; Table B.2 reproduction remains **INCOMPARABLE**. Full original benchmark acquisition and execution remain **NOT_RUN**.

Rerun locally: `.venv/bin/python labs/_ablation_l053.py`. The notebook inlines the intervention class and pairing function; it reuses its own live baseline and fits nine additional neural models. Four student TODO functions remain substantive and active. The written EXIT is required before saving `data/cache/l053-student/exit.json`; this gitignored artifact contains comparison/intervention predictions, row selections, histories, versions, paired results, live callable fingerprints and the learner's explanation. It is evidence for review, not automatic mastery.

Rebuild order after this repair: `_figures_l053.py`, targeted `_lesson_depth.py 53`, refresh the 053-only architecture preview/PNG, `_build_l053.py`, execute all solution cells, `_render_l053.py`, `_check_l053.py`, `_source_check_l053.py`, `_gate_check_l053.py`, `_delivery_check_l053.py`. Do not rerun broad architecture/depth generators over other lessons. The historical model, training harness, local result and larger measured operator remain byte-identical.

Numerical edge boundary: the pinned author scaler adds `1e-30` to positive denominators; the historical teaching scaler uses an exact reciprocal with an explicit zero branch. Ordinary-scale parity is verified, not equality for arbitrarily tiny positive IQR/range. Both the mathematical clipping derivative and strict ordering statements concern exact arithmetic; floating-point saturation can merge tail values and the squared term can overflow for extreme inputs. The historical measured implementation is preserved rather than silently changing its numerical domain.
