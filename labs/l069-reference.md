## The contract card

[Primary paper v1](https://arxiv.org/html/2505.16226v1) · [Pinned released source](https://github.com/LAMDA-NeSy/Evaluation-on-Tabular-Model-in-Open-Environments/tree/744c010457f68284faa7ae6ded793a8b3f3e03a4) · [Source audit](../labs/_source_check_l069_v2_results.json) · [Model check](../labs/_model_check_l069_v2_results.json)

| Axis | Changed quantity | Held fixed | What must be reported |
|---|---|---|---|
| Emerging class | Context label support | Required query IDs | Novel prevalence, detection score, unsupported rows, known/all-row metrics |
| Feature removal | Query feature availability | Context, weights, query IDs | Actual removed fraction, fallback, clean and shifted scores |
| Feature addition | Available schema | Fitted meaning of existing columns | Alignment policy and whether new information was actually used |
| Distribution shift | Input distribution or conditional label rule | Declared source context | Source/target laws or domain definitions; comparable controls |
| Objective change | Metric or decision cost | Saved predictions for rescoring | Class weighting, F1 averaging, threshold-selection boundary |

## Essential calculations

- Continuous novelty: `1 − max_c p(c|x,context)`. It is not a calibrated probability of novelty.
- Released interval detector: `1{.4 ≤ max p ≤ .6}`. Below `.4` it can miss a more uncertain multiclass row.
- Binary-score ROC-AUC: `(TPR + TNR)/2`. Continuous ROC-AUC instead counts ordered novel–known pairs, with half credit for ties.
- AP: sum recall increments times precision at score thresholds; not trapezoidal PR-AUC. A constant score's AP equals novel prevalence.
- Unsupported target: zero probability. Diagnostic clipped loss uses epsilon `10⁻¹²`, penalty `27.6310` nats per unsupported row. Always preserve the full required denominator.
- Mean feature fallback: query column becomes its **context** mean. All other columns, IDs and context labels remain unchanged.
- Absolute accuracy change: shifted−clean. Relative change: `(shifted−clean)/clean`. `.852→.809` means −.043 or −4.3 percentage points; relative −5.05%.
- Balanced accuracy: mean class recall. Macro F1: mean class F1 over the union of true and predicted labels (sklearn default). Balanced accuracy uses true-present classes. Neither equals the minority class's F1.
- Pure concept reversal with identical binary query X: unchanged predictions, accuracy and AUC become their complements. A distance based only on fixed `f(X)` remains unchanged.

## Evidence boundaries

Full historical v2 checkpoint, all 81 tensors, one raw numeric view at temperature .9. Local model source is `tabpfn_l069_v2.py`: immutable L064 architecture with CPU SDPA scheduled in four dimensions and identical float32-rounded attention scale. Full original 2.0.9 forward/input/parameter-gradient and wrapper checks pass. The original four-view wrapper is separately checked, not used in local scores.

Novelty: full CMC/red/white, all held classes, seeds 42/2023/789, balanced known/novel query sets. Natural-prevalence class 0 task is separate and preserves every stratified 20% test row. Features: full Iris/CMC/red, same seeds, six nominal levels using nested label-blind masks and context means. Generated threshold-law controls are synthetic. XGBoost 3.3.0 uses fixed 100/depth 6/eta .3 histogram trees. No tuning or model selection.

Average held classes within seed, then seeds within dataset. Sample SD measures split/mask variability; only three real dataset units support descriptive paired uncertainty. Original tuned seven-model, four-axis paper benchmark remains NOT_RUN and local results INCOMPARABLE. Current released source uses later package defaults and contains aggregation/metric/protocol discrepancies; original files and results remain preserved.

## EXIT requirements

Six live functions, immediate checks, independent source/model checks, new predictions, exact class/row accounting, and a written failure plus negative control. The fresh notebook artifact lives under `labs/data/cache/l069-student/<run-id>/exit-v2.json`. Successful execution is evidence of a runnable lab, not learner mastery or live Colab verification.
