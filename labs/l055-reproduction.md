# L055 reproduction contract

**Verdict: INCOMPARABLE to TabReD Figure 2.** This is a measured evaluation-protocol lab, not a reproduction of the original model ranking.

## Sources and version boundary

- Paper: https://arxiv.org/html/2406.19380v4, especially §5.4 / Figure 2 and Appendix C.2.
- Official repository pinned at `b5ef15b3749f30da7a1eb8fba21a5b54d706bf32`.
- Public Kaggle `irubachev/tabred`, July 10, 2026 preprocessed release. Three archive hashes match the pinned registry. The fetcher rejects unexpected bytes; it never silently moves to a changed release.
- Ecom Offers: 160057 rows, 113 numeric + 6 binary columns, no categorical columns.
- Homesite Insurance: 260753 rows, 253 numeric + 23 binary; 23 categorical columns omitted.
- Sberbank Housing: 28321 rows, 365 numeric + 17 binary; 10 categorical columns omitted. Target = `log(price_doc / full_sq)`, confirmed in pinned `preprocessing/sberbank_housing.py`. RMSE is on that released log target, not raw prices.
- Metadata is excluded from predictors. The first metadata column gives event ordering; Ecom's second metadata column is also excluded.
- Release counts differ from paper Table 2. Official provenance does not certify identity to the paper experiment. See `_sources_l055.json` for source URLs and `_data_l055.json` for array hashes.

## Local experiment

Same released underlying row pool and partition sizes for `random-0` and `sliding-window-0`. Within each partition use label-blind sampling without replacement, caps 1500/600/600 and fixed sampling seeds 550/551/552. All model arms see identical rows within a protocol. The capped row unions across protocols need not coincide, an additional sampling difference.

Released temporal splits are ordered nondecreasingly but share boundary timestamps on all three tasks. We preserve official indices for this comparison. A separate live exercise keeps whole timestamp groups in half-open intervals; it is not silently substituted into the benchmark experiment. Per-feature availability and label arrival metadata are not supplied here; the timing diagram is synthetic, not a verified property of these arrays.

Preprocess train-only: replace nonfinite entries by training column medians (zero if entirely missing), then standardize by filled-training mean/std (unit scale if constant). Numeric and binary features share this policy across arms. Neural regression training uses training-target standardization, with predictions restored before metrics. Classification objective is cross entropy; regression objective is squared error. Score 1−AUROC or RMSE, lower is better.

Three model seeds (0/1/2); two candidates per arm. All candidates are trained using only train/validation. First minimum validation error selects the candidate, then only its test predictions are computed. The neural checkpoint is the first minimum validation metric over 32 epochs. No test-selected changes or refit on train+validation.

- MLP: two width64 Linear–ReLU–Dropout(.1) blocks, head; AdamW weight decay 1e−4, batch256, learning-rate candidates .001/.003.
- TabM-mini: same blocks, k=8, first member adapters, shared weights and separate heads; mean member training loss, mean prediction; same neural candidate recipe.
- XGBoost: histogram trees, 120 estimators, learning rate .05, depth candidates 3/6, row/column sampling .8, one thread.

These are teaching configurations, not a full search over each model. Candidate count is equal; compute and search-space coverage are not. All code is visible in the notebook, including the reused L054 neural models and the trainer. TODO preprocessing/selection functions feed the actual training experiment; call-counter smoke checks establish the connection.

## Evidence and uncertainty

`_verify_l055_results.json` stores environment versions, source hashes, capped index hashes, selected candidates, all validation candidate metrics, selected test predictions/targets, seed errors, per-task ranks and summaries. The executed teacher notebook reruns the protocol. The delivery checker reconciles stored predictions and notebook results.

Report sample SD over three training seeds. The plotted 95% Student-t intervals use mean ± 4.30265 × SD / sqrt(3). They condition on one split pair, row caps and settings. They omit new periods, new tasks, new splits and data-preparation uncertainty. Seed labels do not create matched random/time test observations.

Average seeds within each dataset, then rank the three arms. Friedman and Nemenyi are separate exploratory summaries for each protocol over three task units. This is not the paper's uncertainty-aware ranking/Tamhane procedure. The low dataset count limits inference; nonsignificance is not equivalence.

Local result: Ecom and Homesite deteriorate for all arms and retain XGBoost as winner. Sberbank improves for all arms; point-estimate winner changes from TabM-mini to MLP. Ecom's TabM-minus-XGBoost error gap widens .01267 → .03922, unlike the paper's headline shrinking XGBoost lead. Different arms, omitted categories, caps, preprocessing, search, sampling and data version preclude a contradiction claim.

## Run and scale up

From the repository root:

```bash
.venv/bin/python labs/_check_l055.py
.venv/bin/python labs/_verify_l055.py --preset smoke
.venv/bin/python labs/_verify_l055.py --preset local
.venv/bin/python labs/_verify_l055.py --preset closer
modal run --detach modal/l055_paper_repro.py --preset closer
```

`closer`: three released split pairs, 3 seeds, caps6000/2000/2000, 64 neural epochs and 300 trees, otherwise the same code/candidates. It is supplied but NOT_RUN during authoring. No resume cache: rerunning retrains; download archives are cached and checksummed. Modal writes `closer-results.json` into volume `relational-l055`; retrieve it with `modal volume get relational-l055 closer-results.json`. The cloud image pins a separate environment, which is recorded in its results; exact local score parity is not promised.

The notebook's gated scale-up calls the current live student functions. Modal bundles the repository’s canonical files. To scale modified student functions on Modal, first port and verify them in the canonical files. Cloud training was not launched.

**Full paper procedure: NOT_RUN and not implemented as a misleading `paper` resource preset.** Restore the original Figure 2 arms (MLP, MLP-PLR, XGBoost, TabR-S), all eight tasks, full feature policy, dataset-version alignment, preprocessing and original tuning configurations, three matched temporal/random windows and 15 initialization seeds. Resolve the paper/release identity before choosing a numerical tolerance. More epochs alone cannot close these gaps.

Prepared HTML and inline PNG payloads can be checked locally. Browser, live Colab and deployed Pages behavior are separate verification targets; all remain NOT_CHECKED unless recorded otherwise in `_delivery_l055_results.json`.
