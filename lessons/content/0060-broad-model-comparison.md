## The checkpoint is a defensible comparison

A future relational model must beat a credible single-table procedure. That means more than placing five model names in a table: the competitor includes the feature representation, optimizer, search, validation rule and resource budget. Your outcome here is a report whose selected models, predictions, uncertainty and recommendation can all be reconstructed. A failed neural improvement is a valid outcome.

**Retrieve first, without notes.** From Lesson 058, why can a carefully chosen collection of datasets still favor a particular family? From Lesson 059, which labels may select a candidate, and which labels may estimate its performance? From Lesson 054, does TabM train on the loss of the average prediction or the average of member losses? Write your answers before reading the code.

This is an evaluation checkpoint, introducing no new model. We reuse the numerical MLP, RealMLP-TD-S and corrected TabM-mini taught earlier, with XGBoost and CatBoost as established comparator libraries. The new implementation skill is the **evidence chain**: validate the partitions, freeze a choice, reconstruct its metric, check the declared panel, and compute paired effects. Five live notebook functions implement these operations; a small fresh five-arm training run and the full saved-panel audit both use them.

The earlier 210-record result remains available as historical evidence. It used `relkit/tabm.py`, whose mini architecture retained extra adapters and used incorrect fan-in initialization. The corrected measurement uses `relkit/tabm_v2.py` and a separate `checkpoint_l060_v2.py`; its source hashes and all 210 newly fitted results are in the [v2 evidence](../labs/_verify_l060_v2_results.json). Historical numbers must never be relabeled as corrected-model evidence.

## Four papers ask different comparison questions

Read [TabArena v1 §§2–3 and Appendices A–D](https://arxiv.org/html/2506.16791v1), then [TabReD v4 §5.4 and Appendix C](https://arxiv.org/html/2406.19380v4). These are complementary protocol references. The first focuses on curated IID tasks and peak attainable procedures; the second asks how evaluation changes when future rows replace shuffled rows. Their rankings cannot be stitched into one leaderboard.

| Source and exact scope | Load-bearing contribution | What this checkpoint does |
|---|---|---|
| TabArena v1 §§2.1–2.3, A.4–A.5, C.1–C.2 | 51 curated tasks; inner eight-fold ensembles, one default plus 200 sampled configurations; repeated outer evaluation; explicit preprocessing and prediction artifacts | Imports the idea of an explicit procedure and auditable artifacts; uses a small fixed holdout, two candidates, no outer CV or post-hoc ensemble |
| TabReD v4 §5.4 and Appendix C | Three paired random/temporal splits and 15 initializations study changes in relative performance | Uses three released datasets, one split pair, three seeds and reduced row counts; estimates a conditional regime contrast |
| [TabM v3 §§3.3–3.4, D.2–D.3](https://arxiv.org/html/2410.24210v3) | A minimal first-input adapter creates member diversity; shared backbone and separate heads; memberwise training, prediction averaging | Corrected mini architecture, eight members, shared training batches, no nonlinear feature embeddings, smaller training budget |
| [RealMLP v2 §3, A.2, C.6](https://arxiv.org/html/2407.04491v2) | A complete tuned-default recipe, simplified TD-S variant, validation checkpoint selection and explicit aggregate uncertainty | Reduced TD-S at width 64 and 24 epochs; log-loss validation instead of the paper's main accuracy selection |

TabArena's main binary metric is AUROC, while this checkpoint uses log loss. Its default RealMLP configuration disables label smoothing for its metric; ours retains the simplified original classification recipe's smoothing. TabM's paper also uses a special standard-deviation-based ranking rule, whereas ours uses ordinary average-tie ranks. Matching model names therefore does not imply matching procedures. The [claim/source inventory](../labs/_sources_l060_v2.json) and [reproduction contract](../labs/l060-reproduction.md) pin the exact sources, implementation dependencies and remaining work.

## Freeze the experiment before fitting

Bring the seven decisions from Lesson 059: **metric, candidate procedures, split identity, tuning budget, selection rule, freeze point, postponed adjustment**. If you have its `exit.json`, compare that ledger to this protocol. A disagreement is a declared change before a fresh run, not permission to rewrite a decision after seeing the test result. The author correction was motivated by a code defect already discovered; it is a new measured operator, not independent confirmation of a hypothesis invented from its new scores.

Eight familiar public datasets—diabetes, blood transfusion, kc1, phoneme, German credit, churn, bank marketing and adult income—form the continuity panel. They are real data, but this convenient course roster has not passed TabArena's full curation. The released Ecom Offers, Homesite Insurance and Sberbank Housing tasks add random and temporal panels. There are **14 dataset/regime cells, 11 underlying datasets and 210 selected fits**: 14 × 5 methods × 3 seeds. The two regimes of one TabReD dataset are related observations.

| Decision | Frozen local value | Why it matters |
|---|---|---|
| Public rows | At most 900, sampled without labels by RNG 60 | Small-data comparison; different from full datasets |
| Public partitions | 60/20/20; first split RNG 60, second RNG 61; both stratified | Explicit split identity, identical rows across arms |
| TabReD rows | Released `random-0` and `sliding-window-0`; caps 540/180/180 | Matches regime pair at one window, not full paper scope |
| TabReD cap draws | RNG 550/551/552 for train/validation/test; sampled IDs sorted | Label-blind sampling reproducible from released pools |
| Model seeds | 0, 1, 2 crossed with all arms | Same seed keys permit pairing, not identical random draws across libraries |
| Candidate budget | Two configurations per arm; every candidate fitted | Search cost includes both the winner and loser |
| Selection | Minimum validation loss, first candidate on ties | Test labels never choose a candidate |
| Neural checkpoint | Lowest validation error among 24 epochs, last epoch on ties | Epoch selection is additional adaptation |
| Failure | Stop with an incomplete artifact; no rank summary until complete | A failed task cannot silently disappear from the denominator |

No arm is refitted on train plus validation after selection. The selected fitted object is evaluated as it stands. Changing this would create another procedure with a different training size. Equal candidate counts do not equalize search power: neural methods also compare epochs, and one tree configuration can cost more than another. Equal wall-clock limits or published defaults would each answer a different practical question.

## Preserve each baseline's important recipe

The common encoder fits numeric medians and categorical one-hot columns on training rows only. Validation and test transform through that fixed state. Unseen categories become an all-zero one-hot block. For example, training cities `{A,B}` become `[1,0]` and `[0,1]`; future city `C` becomes `[0,0]`, not a third column learned from test data. Numeric missingness receives the training median; a column entirely missing in training is filled with zero because its median is undefined. The categorical path casts values to strings, so a missing categorical value can itself be a category if observed in training.

Common one-hot input is a **controlled representation restriction**. It suppresses CatBoost's native categorical machinery, so this is not a test of the strongest available mixed-type CatBoost pipeline. For TabReD we retain numeric and binary columns and count omitted categorical columns. Data and feature hashes document the actual inputs. A clean splitter cannot undo a feature that already contains future information: availability of source aggregates and label arrival times remains an independent audit question.

| Arm | Candidate difference | Fit and inference path |
|---|---|---|
| XGBoost | Depth 3 or 6 | 100 histogram boosting rounds, learning rate .05, row/column subsampling .8 |
| CatBoost | Depth 4 or 6 | 100 iterations at .05, one thread, common one-hot input, no validation early stopping |
| MLP | AdamW learning rate .001 or .003 | Training-derived standardization; three width-64 ReLU/dropout blocks, dropout .1, weight decay 1e−4 |
| Reduced RealMLP-TD-S | Base learning rate multiplier 1 or .5 | Robust scaling and smooth clipping; trainable feature scales, NTP layers, SELU/Mish, Adam (.9,.95), coslog4 schedule and parameter groups; classification smoothing .1 |
| Corrected TabM-mini | AdamW learning rate .001 or .003 | Standardization, first-input adapter only, three shared width-64 blocks, shared backbone biases, eight distinct heads, dropout .1, weight decay 1e−4 |

The neural models always complete 24 epochs; validation selects the returned state. The RealMLP schedule is compressed into that duration, so “24 out of 256 epochs” would be misleading: we traverse a complete shortened schedule. The corrected TabM-mini removes the extra first-output adapter and member backbone biases and initializes stored `[input,output]` weights by their actual input dimension. The existing Lesson 054 copied-weight source check validates the numeric architecture; it does not certify this benchmark recipe.

For TabM classification, member logits have shape `[batch,8,2]`. Training averages eight cross-entropies; inference softmaxes each member then averages probabilities. With true class 1 and member probabilities .9 and .1, mean member loss is `(-ln .9 - ln .1)/2 ≈ 1.204`, while loss of the averaged probability is `-ln .5 ≈ .693`. The second objective permits compensation between members; the measured operator uses the first. Neither eight members nor three model seeds create extra independent datasets.

<!--figure:protocol-->

## Test labels enter once the choice is frozen

Follow the diagram's boundary. The fitter receives training arrays and validation arrays, returns one frozen predictor plus its validation error and selected epoch, and has no test argument. The selector sees only the two validation errors. Only then does the selected predictor receive test features and the metric function receive test labels. A notebook CHECK changes saved targets and confirms that candidate selection is unchanged while metric reconstruction detects the changed evidence.

Suppose candidate validation errors are `[.43,.39]`, but their unknown test errors would be `[.37,.46]`. The valid choice is candidate 1, with reported test error .46. Reporting .37 would select on the evaluation set. If the deployment decision itself chooses between all five families after this report, its final chosen-family estimate may also be optimistic: a new untouched evaluation or nested selection is needed for that extra adaptive step. The saved five-arm table still validly describes its declared fixed procedures.

Row identity is equally necessary. A probability vector has no meaning until linked to target rows and class order. Each v2 record contains test IDs in order, targets, `P(y=1)` or scalar regression predictions, class schema, all candidate validation losses, each candidate's selected epoch and the winning index. The audit demands that every arm/seed of a task uses the same target sequence and the exact declared test-ID order. This catches accidental shuffling; it cannot prove that the raw dataset's semantic feature construction was leak-free.

## Derive the reported errors

For binary labels `y_i ∈ {0,1}` and probabilities `p_i=P(y_i=1)`, the probability assigned to the observed label is `p_i` when `y_i=1` and `1−p_i` otherwise. Taking the negative log turns confident mistakes into large penalties; averaging gives

`log_loss = −(1/n) Σ_i [y_i ln(p_i) + (1−y_i) ln(1−p_i)]`.

For labels `[1,0]` and probabilities `[.8,.3]`, the contributions are `.22314` and `.35667`, giving `.28991`. Threshold accuracy would discard how confident those predictions were. The implementation checks finite aligned one-dimensional arrays, binary labels and probabilities in `[0,1]`, then clips to float64 machine epsilon and its complement to make endpoint logs finite. A malformed probability of 1.1 is rejected before clipping; clipping is not a license to repair invalid outputs silently.

For regression, square each residual in the released target's units, average, and take the square root: `RMSE = sqrt(mean((y−ŷ)²))`. Targets `[1,5]` and predictions `[4,1]` give squared residuals `[9,16]` and RMSE `sqrt(12.5) ≈ 3.5355`. For the released Sberbank data the target is `log(price_doc/full_sq)`: RMSE is in those logarithmic units, not currency and not raw sale-price RMSE. Its train and validation time ranges meet on a tied timestamp in split 0. The artifact accurately records non-strict ordering; a strictly later label-availability claim would require more evidence.

Do not average raw log loss and RMSE across tasks: the units differ, and a change of target units could determine the result. Compare raw effects within each task, then use dimensionless ranks to summarize a declared task collection.

## Read the result at three resolutions

**Predict first:** if A's losses across two seeds are `[0,1]` and B's are `[.4,.4]`, which method should rank first after seed averaging? A wins one seed emphatically; B has the better average loss, .4 versus .5. If you rank separately per seed and average ranks, both receive 1.5. That is a different estimand: mean seedwise ordering rather than ordering of estimated mean loss. Move the slider and find the tie at A's second loss .8.

<!--figure:aggregation-->

Write `e_ds(m)` for error on dataset `d`, seed `s`, method `m`. First estimate `ē_d(m)=Σ_s e_ds(m)/S` with `S=3`. Within each dataset, assign rank 1 to the smallest mean error and average occupied positions on exact ties. Then average the `N` dataset ranks. The roster is supplied explicitly to `aggregate_panel`; inferring it only from successful records would fail to notice a completely absent dataset or model. Random and temporal panels are summarized separately.

The **Friedman null** is no systematic method effect in the dataset blocks. For `K` methods, rank dispersion is summarized by `Q=12N/[K(K+1)] × Σ_m [mean_rank(m)−(K+1)/2]²` without ties; SciPy applies a tie correction. Its chi-square reference has `K−1` degrees of freedom and is an approximation. Eleven tasks and five methods are a small panel; three temporal tasks are especially unsuitable for treating asymptotic p-values as precise. We label them exploratory and also compute a within-task method-label permutation calibration: 20,000 draws for the random panel and all 14,400 permutations after fixing the first of three temporal rows. This calibration assumes exchangeable method labels within independent task blocks under the null. It does not make a convenience sample representative.

A **Nemenyi critical difference** compares two mean ranks after accounting for the whole declared method pool. We compute `CD=q(.95,K,∞) × sqrt(K(K+1)/(12N))`, where `q` is SciPy's studentized-range quantile. The `1/sqrt(2)` convention is already absorbed in this form. Its threshold is exploratory here, particularly with three tasks, and is not a confidence interval on an error score. Consult [Demšar §3.2](https://jmlr.org/papers/v7/demsar06a.html) for the comparison framework and small-sample caution. No rejection does not establish equivalence.

<!--figure:ranks-->

### Pair first, then quantify variation

Within one dataset, compute `δ_s=e_ds(A)−e_ds(B)` using matching seed keys. Its mean is the measured error gap; a negative value favors A. The sample SD is `sqrt(Σ_s(δ_s−mean δ)²/(S−1))`; an approximate conditional 95% t interval is `mean δ ± t(.975,S−1) × SD/sqrt(S)`. With three seeds the multiplier is about 4.303. Pairing retains covariance: `Var(A−B)=Var(A)+Var(B)−2Cov(A,B)`. Subtracting unrelated means and adding their separate SDs ignores this structure.

These intervals describe only seed variation conditional on fixed rows, recipes and split. They are not calibrated coverage guarantees for a new enterprise, another temporal window or the general population of tabular datasets. Deterministic seeds can even give SD zero for an arm with little stochastic variation; that does not make its generalization error known exactly. Intervals are unadjusted across the many inspected task/model contrasts.

<!--figure:results-->

<!--figure:comparison-->

### A measured reversal, with its uncertainty

The corrected Ecom Offers comparison makes the difference between a rank and a defensible decision concrete. Here every gap is **TabM-mini minus XGBoost**, so negative favors TabM; the intervals use three paired seeds on the fixed split.

| Task/regime | Mean raw loss gap | Conditional t95 | Interpretation |
|---|---:|---|---|
| Ecom Offers / random | +0.04180 | [0.03109, 0.05250] | TabM has larger log loss |
| Ecom Offers / temporal | −0.11314 | [−0.17868, −0.04760] | TabM has smaller log loss |
| Homesite / temporal | +0.11602 | [0.07275, 0.15929] | The Ecom preference does not transfer |

The Ecom relative gap changes by `−0.11314 − (+0.04180) = −0.15494`. This is a paired-task regime contrast, not a claim that every neural method becomes more robust in the future. A three-seed interval excluding zero concerns this fixed recipe/window only. The evidence does not isolate whether class balance, temporal changes, feature behavior or other differences drove the reversal. Inspect the saved per-row probabilities and class proportions before proposing a new experiment. The [analysis artifact](../labs/_analysis_l060_v2_results.json) contains all 56 non-baseline paired contrasts and their seed differences.

The results above are author-reference measurements, independent of the notebook's unexecuted student cells. The notebook reconstructs every score from the saved predictions and displays full seed points, conditional intervals, candidate choices and costs. Raw values remain available next to the ranked view so a rank difference cannot conceal a negligible practical gap.

## Quality and cost determine a conditional decision

A search-time comparison must include losing configurations. Our `fit_selection_seconds` spans all candidate fits and validation evaluations; `predict_seconds` spans the selected predictor on the complete test matrix. Common data loading and common median/one-hot preprocessing are outside those timers. Neural standardization/robust scaling is inside the fitter. The author run experienced variable host contention, so these timings are descriptive records and cannot establish hardware efficiency superiority. This is one-thread CPU evidence, not a cold end-to-end request latency benchmark and not GPU throughput. The historical `last_invocation_seconds` was only a resumed tail, so it must never be presented as total historical suite cost.

For a deployment decision, specify a task, an acceptable loss difference and a cost limit before examining a new test set. A procedure is **Pareto dominated** if another is no worse on both declared objectives and strictly better on at least one. A nondominated procedure merely remains eligible; it need not be preferred. A score-first ranking answers a different question from minimizing loss subject to a 10 ms prediction budget.

Compare the random-versus-temporal **paired model gap**, not only the two winners. If A−B is −.02 on shuffled rows and +.10 on later rows, the gap changes by .12. This suggests differential sensitivity to the regime. It does not identify temporal drift as the sole cause: windows can differ in difficulty, class balance, feature distributions and legal information. A follow-up should specify which factor is held fixed. The three released tasks are paired across regimes, so comparing the aggregate 11-task random rank to the aggregate three-task temporal rank confounds regime with task composition.

## Exit: write a report a skeptic can rerun

Complete the five code tasks, run the independent CHECK cells, reconstruct all 210 v2 metrics, and run the small fresh five-arm experiment through your live functions. Save `data/cache/l060-student/exit.json` with the source text of the live operations, evidence hashes, complete declared panel, reconstructed summaries, paired effects, permutation calibration, call counts and the seven-decision ledger. Write at least 120 words explaining one task's practical gap, one regime contrast, the timing definition and an evaluation that could reverse your recommendation. Code checks verify the artifact chain; the teacher must still assess your reasoning.

Then take the explicit **NEXT STEP** after EXIT: run the corrected `lab` panel again from a fresh output path, or the larger `closer` operator with 4,000 public rows, 128 neural epochs and 800 tree iterations. The notebook gate is off by default; a CPU Modal wrapper exists for unattended runs. Neither is a paper-reproduction preset. Full TabArena reproduction instead requires its curated suite, versioned wrapper, nested folds, candidate search and ensemble treatment; full TabReD replication requires the original task release, full feature treatment, three split pairs, model roster, metric and evaluation seeds. The [contract](../labs/l060-reproduction.md) separates these paths.

Connect this checkpoint to what comes next. Lesson 061 asks how a learned prior can replace per-task fitting; later pretrained predictors must still pass the same label-access, version and cost accounting. A future relational model adds an entity graph and point-in-time features, increasing the importance of these boundaries. The single-table baseline is the measured procedure it must beat, not an under-specified model name. Explain the boundary in your own words and ask the teacher about any operation you cannot trace.
