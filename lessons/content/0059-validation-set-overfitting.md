## The validation set becomes training data for your decisions

**Outcome:** implement and audit the boundary between choosing a procedure and estimating its performance. You will trace a winner selected from noise, reconstruct the paper's kernel ridge and leave-one-out equations, and run a nested evaluation that survives a held-out-label mutation test. This is an evaluation lesson; the kernel model is a fully visible measuring instrument.

A **parameter** is fitted inside a model, such as a regression coefficient. A **hyperparameter** controls that fit, such as regularization strength. A **selection criterion** is a score used to choose hyperparameters, features, checkpoints, or ensemble weights. A **performance estimate** answers how well the resulting procedure predicts new observations. The same number cannot automatically serve both purposes after you minimize it.

Suppose your relational predictor gains .01 in validation AUC after your eighth feature-history redesign. That may be useful development. It does not yet establish that relational structure beats a strong flat-table baseline: the procedure includes all eight designs and the decision to retain this one. In lesson 057, out-of-fold (OOF) means each row was excluded from fitting its own base predictor. The row's target can nevertheless fit the ensemble selector. In lesson 058, selecting tasks to preserve known-model rankings can specialize the benchmark itself. Here the adaptive object is the model-selection procedure.

**Retrieve first:** explain why an OOF combiner needs its own evaluation boundary. Then name one decision that consults validation labels without computing a gradient. Keep your answer until the EXIT teach-back.

The primary reading is [Cawley and Talbot (2010), full paper](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). Read §§2–3 for the instrument and data, §4 for selection overfitting, and §5 for biased evaluation protocols. These are separate empirical arguments. The paper does not introduce a new neural architecture or claim every larger search must worsen test performance.

## A minimum selects lucky noise

Fix the training sample and candidate predictors. Candidate j has unknown expected new-data error μ_j. Its observed validation estimate is μ_j + ε_j, where ε_j is sampling noise. If the estimates are unbiased for each fixed candidate, the expectation of each ε_j is zero. The selected index j* minimizes the **observed** estimates, so ε_j* is no longer an unselected draw. This conditioning is the source of optimism.

For three equally good candidates with true error .30, suppose observed errors are .33, .26, .31. The second wins, but its underlying error remains .30 in this example. With unequal true errors, selection mixes useful discovery and lucky noise: finding a better candidate and overestimating its improvement can happen together. Perfectly duplicated candidates create no additional opportunities; correlated candidates lie between duplication and independent draws. Raw search count alone does not determine the bias.

A small exact example makes the expectation visible. Each of two equally good candidates independently reports .4 or .6, each with probability one half, around true error .5. The four equally likely pairs have minima .4, .4, .4, .6. Their mean is .45. The expected fresh-test error remains .5. **Optimism** here is fresh-test loss minus the selected estimate; positive values mean the selection score looked too favorable.

<!--figure:mechanism-->

A constant upward bias need not change which candidate wins. Adding .1 to every estimate preserves the argmin. Independent fluctuations can reverse the ordering even when their mean is zero. This is the distinction in the paper's §4.1/Figure 3: good selection needs a reliable minimum, whereas good evaluation needs a reliable reported level. It does not imply biased evaluation is acceptable.

## Run a negative control you can explain completely

A **negative control** removes predictive signal while preserving the operation being tested. The historical lab generates independent Bernoulli(.5) labels and predictions, selects the lowest classification error on 80 validation rows, and evaluates the frozen candidate on 2,000 independent test rows. Labels are balanced **in expectation**, not forced to exactly half of each class. All candidate predictors have expected error .5.

There are 200 repetitions and candidate budgets 1, 4, 16, 64, 256. Within a repetition, candidates are nested prefixes and each candidate has its own reproducible test prediction vector. Budget comparisons are therefore paired. The saved `seed` field is the repetition index; the actual RNG seed is 59 plus that index. Separate validation/test streams prevent changing the budget from accidentally moving the test stream. Generating every candidate's test predictions after selection is fine if only the frozen index is evaluated; searching those test predictions would be a different, invalid protocol.

**Predict:** must the selected validation minimum fall as the candidate budget grows? Must the selected test error be monotone? Which claim is a deterministic property of nested sets and which is an expectation over random data?

The preserved 200-repeat evidence gives validation error .495875 at one candidate and .342188 at 256; fresh-test error at 256 is .497868. Optimism is .155680, with Monte Carlo SE .001652. A **Monte Carlo standard error** is the sample SD of per-repeat gaps divided by √200; it measures precision of this simulation average, not uncertainty across real datasets. The 256-minus-1 comparison must subtract within each repetition before computing its uncertainty.

<!--figure:results-->

The historical operator and JSON remain unchanged. The new lab replays all 1,000 saved repeat-budget rows with your live selector, then reanalyzes them with your paired-summary function. This validates historical arithmetic without pretending these are newly trained models. It also computes an independent exact expectation: if B_j is Binomial(n,.5), then

`E[min_j B_j/n] = (1/n) Σ[k=0..n−1] P(B > k)^M`.

Here n is the validation row count and M the number of independent candidates. A nonnegative integer variable equals the number of thresholds it exceeds; taking expectations gives the survival sum. Independence across candidates turns the probability that **every** candidate exceeds a threshold into a power. At M=1 the result is .5. At n=1, M=2 it is .25. This formula is exact for this independent control, not a correction formula for correlated real models.

## What the paper actually measures

The noise control teaches optimistic selection, but it does not reproduce the paper's signal-bearing experiments. In §3.1 the authors specify four equally likely Gaussian components in two dimensions. Two positive components have centers (.4,.7) and (−.3,.7); negative components have centers (−.7,.3) and (.3,.3). Each coordinate has variance .04, hence standard deviation .2. There is real signal and unavoidable overlap. The paper reports a Bayes classification error of about 12.38%; this is a published optimum for classification, not a target value for mean squared error.

Figure 2 uses 1,000 independent 64-row datasets, an automatic relevance determination (ARD) kernel, four-fold cross-validation MSE, and iterative hyperparameter optimization. Its expected test curve initially improves and later rises. Figures 4–8 instead hold 256 fitting rows fixed, vary independent validation samples of 64 or 256 rows, and inspect a smoothed classification error surface. More validation data makes the selected hyperparameters less variable in that experiment. These are different experiments; neither is a random-label test.

The practical evidence comes from thirteen benchmark datasets (§3.2/Table 1), with 100 released train/test realizations per dataset except image and splice with 20. In Table 2, a more flexible ARD kernel can obtain a better selected PRESS criterion yet worse test classification error than an isotropic RBF kernel. More representational capacity can coexist with a less reliable tuning procedure. This finding does not establish that ARD is intrinsically inferior or that every ensemble should use fewer members.

In §5.2, the problematic “median” protocol tunes the first five training realizations, takes componentwise medians of their hyperparameters, then reuses those settings across all realizations. Test rows in one split can have influenced selection in another. Moreover, averaging reduces selection variability, so the evaluated procedure differs from tuning anew in each split. In §5.3, “external” selection uses the whole dataset before outer cross-validation; each outer test row can influence its hyperparameters. The remedy is to rerun selection internally within each outer training portion. The revised lab executes this latter distinction.

## Reconstruct the measuring instrument: model architecture

Kernel ridge regression predicts a real number f(x), then classifies by f(x)≥0 for +1. A **kernel** measures similarity to training rows without explicitly constructing nonlinear features. The Gaussian radial basis function (RBF) kernel is exp(−η‖x−z‖²). Increasing positive η makes similarity decay faster with distance. ARD uses one η_q per input coordinate: exp(−Σ_q η_q(x_q−z_q)²). Equal coordinate scales recover RBF.

For x=(.4,.7), z=(−.3,.7), and η=(2,.25), the weighted squared distance is 2×.7²+.25×0²=.98, so similarity is exp(−.98)≈.3753. Changing the second scale cannot affect this pair because its second coordinates agree. That is a computation you should be able to predict before running the kernel.

Let X contain n rows and d=2 features, K be its n×n kernel matrix, y the length-n ±1 targets, α the length-n coefficients, and b an unpenalized intercept. The paper's objective is one half of squared feature-space weight norm plus `Σ(y−f)²/(2λ)`, with positive regularizer λ. Stationarity gives α=(y−f)/λ and Σα=0. Substituting f=Kα+b produces the augmented linear system (paper Eq.3):

`C = [[K + λI, 1], [1ᵀ, 0]];  C [α; b] = [y; 0]`.

I is the identity matrix and 1 is an all-ones column. The final equation constrains Σα=0; the final unknown supplies b. Dropping that row/column silently changes the estimator. The visible code solves this system in float64. For new X*, predictions are K(X*,X)α+b. No labels of X* enter fitting or selection. These real-valued scores are not probabilities: the classification head returns +1 for f≥0 and −1 otherwise, while the selection criterion uses squared score residuals. The signed, unnormalized coefficients differ from TabR’s nonnegative softmax neighbor weights; a distant query approaches b as its similarities decay. The tiny implementation forms the full inverse to expose its diagonal; its cubic solve cost and quadratic storage are unsuitable for a large benchmark without a more efficient solver.

<!--figure:architecture-->

The connected diagram separates fitting/selection from inference. For each candidate, the solve already fits all available development rows; LOO scores that fixed recipe through deleted residuals. The minimum chooses which fitted model is frozen. At inference only new features enter; scores feed the squared-error evaluation and the sign head supplies classification. The following operation table expands one numerical kernel calculation.

<!--figure:kernel-->

Leave-one-out (LOO) removes row i, fits the same fixed hyperparameters on all other rows, and predicts row i. The paper's §2.1 identity gives its deleted residual as `r_i^(−i) = α_i / (C⁻¹)_ii`, using the **upper-left n diagonal entries of the augmented inverse**. PRESS is the sum of these squared deleted residuals; the lab divides by n to report MSE. Dividing by the fixed row count leaves candidate ordering unchanged. Ordinary fitted residuals use a model that has seen y_i and are not the same quantity.

The lab independently verifies this identity against actually leaving out each row and refitting. It also tests the intercept with constant training targets and a distant query. The provided model source is readable in the notebook; your live tasks implement selection, deleted residuals, candidate search, outer nesting, and paired inference.

## Nest the decision that is actually adaptive

An **outer fold** is an evaluation partition. For each outer fold, the remaining rows are the development sample. LOO selection is performed again on that development sample; the selected model is fitted on those rows and predicts only the outer-held rows. Each row receives exactly one OOF prediction. The final reported outer MSE averages squared errors across all rows. This estimates the whole selection-and-fit procedure at the outer training size.

In the worked protocol, a 64-row dataset is split into four equal label-blind shuffled blocks. Each internal selection sees 48 rows; each of its LOO fits effectively trains on 47. After selection, a 48-row refit predicts the held 16. In the deliberately contaminated external comparator, LOO selection first sees all 64 targets and freezes one candidate for all four outer refits. Those 48-row models still excluded their evaluation rows from coefficient fitting, but their **choice** already depended on those rows.

<!--figure:protocol-->

**Trace before running:** change the targets in outer fold 2. Its own internal selected hyperparameters and predictions must remain unchanged. Other folds may change because those rows legitimately belong to their training portion. The CHECK enforces this distinction. Mere disjointness of model fitting rows is insufficient to protect a globally selected hyperparameter.

A model refitted on 64 rows and a model refitted on 48 rows have different evaluation targets. The experiment therefore creates fresh independent rows only after freezing every choice and evaluates **each outer-fitted model** on those fresh rows. It averages their fresh MSE within the repetition. Subtracting each protocol's outer score from its own matched fresh score measures optimism without attributing a training-size effect to leakage. The 64-row refit's fresh score separately evaluates selected LOO optimism; its LOO fits used 63 rows, so that comparison retains a small training-size difference.

Nested evaluation does not repair a poor selector. Its job is to include that selector's mistakes in the estimate. Nor does it protect a researcher who redesigns the candidate grid after viewing outer scores. That redesign needs new evidence. For customer histories or temporal data, replace random folds with deployment-appropriate entity/time boundaries and label-availability rules; random nesting cannot remove temporal leakage.

## Measure the boundary, with uncertainty

The new protocol is fixed before the author run: 27 candidates from λ∈{.01,.1,1} and each of two η coordinates in {.25,2,16}; analytic LOO mean PRESS; first-index ties; four equal outer folds; 64 development rows; 4,096 fresh rows per independent repetition; no learned preprocessing or feature selection. There are 30 teaching repetitions, seeds 5900–5929, followed by a required 1,000-repetition precision track, seeds 5900–6899. The longer run contains the first 30; they are not independent confirmations.

The same candidates, development rows, outer folds and fresh rows are paired across internal/external protocols. All four folds are averaged **inside** each repetition. The independent units are fresh draws of the entire synthetic dataset, not the four overlapping folds or the 4,096 test rows treated as separate experiments. Intervals are paired Student-t intervals for the mean repeat-level difference; they quantify Monte Carlo precision under this generator. They are not benchmark-dataset confidence intervals and are exploratory across multiple contrasts.

**Predict:** will the external outer score look better? Can its actual fresh score also be better because its selection legitimately used more information, even though its claimed evaluation was contaminated? Should the internal estimated optimism be exactly zero in a finite run?

<!--figure:nested_results-->

On the 30-repeat teaching run, internal outer MSE is .482096 versus matched fresh MSE .509442; the optimism estimate is .027346 with 95% interval [−.028857,.083549]. External outer MSE is .416708 versus its matched fresh MSE .485978; optimism is .069269 [.022023,.116516]. The internal-minus-external outer-score gap is .065387 [.045771,.085004]. These gaps are distinct quantities. The wider internal interval includes zero; that neither proves exactly zero bias nor establishes equivalence. The external fresh score being lower does not make its external outer score a valid estimate.

The [1,000-repeat evidence](../labs/_verify_l059_closer_results.json) gives internal matched optimism .002660 with t95 [−.005395,.010715], and external matched optimism .063726 [.055963,.071489]. External fresh MSE is .492255 versus internal .510056: the external selector saw 64 labels rather than 48, so this is not an equal-information-budget model contest. The bias comparison evaluates each frozen procedure against its own matched fresh target. The figure shows the precision extension. Read its complete contrast summary rather than selecting whichever seed, contrast or interval best supports the story. A run that fails to show the anticipated pattern would remain valid evidence under the prespecified protocol; the lab's correctness checks do not require a particular statistical victory.

## Audit an ensemble without blaming diversity

A uniform mixture of a fixed declared member set has no fitted weights, but the decision to choose that member set can still be adaptive. Greedy ensemble selection fits weights to OOF predictions. Selecting its maximum steps, base library, loss, folds or preferred prefix adds decisions. Protect the entire path, then compare it with a single-family procedure selected by the **same inner evidence**. The family with best final test loss is an oracle comparator, not a deployable choice.

A large OOF-to-test gap can also reflect temporal shift, dependence among customers, a 48-versus-64-row refit difference, or ordinary sampling uncertainty. The current synthetic generator removes temporal shift and controls sample size; it cannot diagnose every real failure. A useful discriminating experiment freezes all recipes and changes only the amount of selection data, or evaluates the complete procedure on new outer blocks.

The bigger lesson connects strong defaults (053), ensembles (057), benchmark construction (058), and the upcoming model comparison (060). Strong defaults reduce local decisions but inherit the assumptions of their original development. Ensembling may reduce prediction variance while increasing selection flexibility. Benchmark reuse can adapt research decisions over years. A fair relational claim compares complete pipelines on aligned future entities/times, gives each a declared tuning budget, and reserves evidence for the final decision. Architectural novelty does not excuse a weaker evaluation boundary.

## Exit: make adaptation countable

Finish all five live functions, replay the historical control, run the 30-repeat kernel protocol and required 1,000-repeat precision track, then submit `data/cache/l059-student/exit.json` and your completed notebook. The artifact records numerical results, live function sources, source evidence hashes, and an adaptation ledger. No learner mastery is inferred from building or executing the teacher notebook.

Write 80–180 words identifying: the selected object; why coefficient-level OOF exclusion is insufficient; the training size behind each estimate; one measured interval and its unit; and one next experiment that could contradict your interpretation. Freeze a lesson-060 candidate list, metric, split identity, selection rule and budget. Name one tempting adjustment you will postpone until a separately labeled follow-up. Ask the tutor to check this argument, especially any numerical result that seems to contradict the expected mechanism.

## Reproduction boundary and reading route

**Equation validation:** the augmented KRR system and exact LOO identity pass independent refit tests. **Historical evidence:** all 1,000 null rows replay under the preserved operator. **New procedure:** internal/external selection and matched fresh evaluation run with your live notebook functions. **Original results:** INCOMPARABLE, not reproduced.

The new run uses the paper's exact synthetic distribution and kernel/LOO equations, but substitutes a finite 27-candidate ARD grid and outer four-fold protocol. It does not reproduce Figure 2's iterative optimizer/four-fold inner criterion, Figures 4–8's SER surface and fixed fitting sample, or Table 8's thirteen real benchmarks and outer ten-fold RBF experiment. Matching 1,000 repetitions is a precision choice, not protocol parity. The original HTTP GKM link in footnote 2 works, although its HTTPS endpoint fails certificate verification. We pinned the downloaded archive and inspected its KRR, RBF, LOO, criterion, cross-validation and simplex source. Its KRR implementation uses a Cholesky solve with the same intercept and a Schur-corrected inverse diagonal. A translated source-algebra check agrees on 18 fixtures; native MATLAB execution, optimizer trajectories and exact 2010 experiment-version parity are not established. Its SSE adapter reports half the residual sum of squares, while our fixed-n mean PRESS has the same argmin. See the [source comparison](../labs/_source_check_l059_v2_results.json) and [pinned reference files](../labs/sources/l059/README.md). There is no appendix in this 29-page article; the full substantive §§2–6 and figure/table captions were inspected.

Use the [reference card](../reference/0059-validation-set-overfitting.html), [equation/behavior checks](../labs/_check_l059_v2_results.json), [source inventory](../labs/_sources_l059_v2.json), [measured evidence](../labs/_verify_l059_v2_results.json), and [reproduction commands](../labs/l059-reproduction.md) together. The latter documents the operator and what remains unrun. Read §4.1 for criterion variance, §4.4 for regularization/averaging strategies, and §§5.1–5.3 for the full evaluation boundary; none makes repeated final-test use free.
