## The validation set becomes training data for your decisions

> **In plain terms.** Every time you look at a validation score and pick the option that scores best, you are letting that data teach your decision. The score that guided the choice can no longer be trusted to report how good the choice was. This lesson makes that leak measurable.

**What you will build.** You will implement and audit the boundary between two jobs: *choosing* a procedure and *estimating* how well it performs. You will trace a "winner" selected from pure noise. You will reconstruct the paper's kernel ridge and leave-one-out equations. You will run a nested evaluation that survives a held-out-label mutation test. This is an evaluation lesson; the kernel model is just a fully visible measuring instrument.

### Four terms, each defined once

- A **parameter** is fitted inside a model, such as a regression coefficient.
- A **hyperparameter** controls that fit, such as the regularization strength.
- A **selection criterion** is a score used to choose hyperparameters, features, checkpoints, or ensemble weights.
- A **performance estimate** answers how well the resulting procedure predicts new observations.

The key rule: once you minimize a number to make a choice, that same number cannot automatically also serve as an honest performance estimate.

**Why this matters for the mission.** Suppose your relational predictor gains .01 in validation AUC after your eighth feature-history redesign. That may be useful development work. It does not yet establish that relational structure beats a strong flat-table baseline, because the procedure you should be evaluating includes all eight designs *and* the decision to keep this one. Recall the pattern from earlier lessons. In lesson 057, out-of-fold (OOF) means each row was excluded from fitting its own base predictor — yet that row's target could still fit the ensemble selector. In lesson 058, selecting tasks to preserve known-model rankings can specialize the benchmark itself. Here the adaptive object is the model-selection procedure.

**Retrieve first.** Explain why an OOF combiner needs its own evaluation boundary. Then name one decision that consults validation labels without ever computing a gradient. Hold your answer until the EXIT teach-back.

**The primary reading.** Work from [Cawley and Talbot (2010), full paper](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). Read §§2–3 for the instrument and data, §4 for selection overfitting, and §5 for biased evaluation protocols. These are separate empirical arguments.

> **Scope check.** The paper does not introduce a new neural architecture, and it does not claim that every larger search must worsen test performance.

## A minimum selects lucky noise

> **In plain terms.** Line up several equally good options, each measured with a bit of random noise. Pick the lowest score. You will tend to pick the one that got lucky, so its winning score looks better than the option truly is.

### Where the optimism comes from

Fix the training sample and the candidate predictors. Candidate `j` has an unknown expected new-data error `μ_j`. Its observed validation estimate is `μ_j + ε_j`, where `ε_j` is sampling noise. If each estimate is unbiased for its fixed candidate, then each `ε_j` has expectation zero. But the selected index `j*` minimizes the **observed** estimates, so `ε_j*` is no longer just an unselected draw — you chose it *because* it was low. This conditioning is the source of the optimism.

**Worked example.** Take three equally good candidates whose true error is .30. Suppose their observed errors come out .33, .26, .31. The second wins, yet its underlying error is still .30. With unequal true errors, selection mixes two things at once: it can genuinely find a better candidate *and* overestimate that candidate's improvement. Duplicating a candidate perfectly adds no new opportunity to get lucky; correlated candidates lie somewhere between exact duplication and fully independent draws. So the raw count of things you searched does not, by itself, determine the bias.

### An exact two-candidate calculation

**Worked example.** Let each of two equally good candidates independently report .4 or .6, each with probability one half, around a true error of .5. The four equally likely pairs have minima .4, .4, .4, and .6. Their mean is .45, which is below the true .5. The expected fresh-test error is still .5. We define **optimism** here as the fresh-test loss minus the selected estimate; a positive value means the selection score looked too favorable.

<!--figure:mechanism-->

### Selection reliability is not evaluation reliability

A constant upward bias need not change which candidate wins: adding .1 to every estimate preserves the argmin. But independent fluctuations can reverse the ordering even when their mean is zero. This is the distinction drawn in the paper's §4.1 / Figure 3. Good *selection* needs a reliable minimum. Good *evaluation* needs a reliable reported level. Neither observation implies that biased evaluation is acceptable.

## Run a negative control you can explain completely

> **In plain terms.** To see selection bias with nothing else mixed in, run an experiment where there is *no* real signal at all. Any apparent "winner" is then pure luck, and any gap between its selection score and its true error is exactly the bias you wanted to measure.

### The control setup

A **negative control** removes the predictive signal while keeping the operation you are testing. The historical lab generates independent Bernoulli(.5) labels and predictions, selects the lowest classification error on 80 validation rows, and then evaluates that frozen candidate on 2,000 independent test rows. The labels are balanced **in expectation**, not forced to exactly half of each class. Every candidate predictor has an expected error of .5.

**How the budgets are paired.** There are 200 repetitions and candidate budgets of 1, 4, 16, 64, and 256. Within one repetition, the candidates are nested prefixes, and each candidate has its own reproducible test-prediction vector. Because the budgets share the same stream, budget comparisons are paired. The saved `seed` field is the repetition index; the actual RNG seed is 59 plus that index. Keeping separate validation and test streams stops a change in budget from accidentally shifting the test stream.

> **Scope check.** Generating every candidate's test predictions *after* selection is fine, as long as only the frozen index is evaluated. Searching those test predictions would be a different — and invalid — protocol.

**Predict.** Must the selected validation minimum fall as the candidate budget grows? Must the selected test error be monotone? Which of these is a deterministic property of nested sets, and which is an expectation over random data?

### What the preserved run shows

The preserved 200-repeat evidence gives a validation error of .495875 at one candidate and .342188 at 256; the fresh-test error at 256 is .497868. The optimism is .155680, with a Monte Carlo SE of .001652. A **Monte Carlo standard error** is the sample SD of the per-repeat gaps divided by √200; it measures the precision of this simulation's average, not uncertainty across real datasets. The 256-minus-1 comparison must subtract within each repetition *before* computing its uncertainty.

<!--figure:results-->

### An exact expectation to check against

The historical operator and its JSON remain unchanged. The new lab replays all 1,000 saved repeat-budget rows with your live selector, then reanalyzes them with your paired-summary function. This validates the historical arithmetic without pretending these are newly trained models. It also computes an independent exact expectation. If `B_j` is Binomial(n,.5), then

`E[min_j B_j/n] = (1/n) Σ[k=0..n−1] P(B > k)^M`.

Here `n` is the validation row count and `M` the number of independent candidates. The identity uses a small trick: a nonnegative integer variable equals the number of thresholds it exceeds, so taking expectations gives a survival sum. Independence across candidates turns "the probability that **every** candidate exceeds a threshold" into a power. At `M=1` the result is .5. At `n=1, M=2` it is .25.

> **Scope check.** This formula is exact for this independent control only. It is not a correction formula for correlated real models.

## What the paper actually measures

> **In plain terms.** The noise control shows the *mechanism* of optimistic selection. The paper's own experiments add real signal, so keep straight which experiment supports which claim, and don't read one experiment's number as another's.

### The synthetic distribution (§3.1)

The noise control teaches optimistic selection, but it does not reproduce the paper's signal-bearing experiments. In §3.1 the authors specify four equally likely Gaussian components in two dimensions. The two positive components have centers (.4,.7) and (−.3,.7); the negative components have centers (−.7,.3) and (.3,.3). Each coordinate has variance .04, hence a standard deviation of .2. There is real signal here, and unavoidable overlap. The paper reports a Bayes classification error of about 12.38%. That is a published optimum *for classification*, not a target value for mean squared error.

### Three experiments that are easy to conflate

- **Figure 2** uses 1,000 independent 64-row datasets, an automatic relevance determination (ARD) kernel, four-fold cross-validation MSE, and iterative hyperparameter optimization. Its expected test curve first improves and later rises.
- **Figures 4–8** instead hold 256 fitting rows fixed, vary independent validation samples of 64 or 256 rows, and inspect a smoothed classification-error surface. More validation data makes the selected hyperparameters less variable in that experiment.

These are different experiments, and neither is a random-label test.

### The real-benchmark evidence (§3.2)

The practical evidence comes from thirteen benchmark datasets (§3.2 / Table 1), with 100 released train/test realizations per dataset, except *image* and *splice* which have 20. In Table 2, a more flexible ARD kernel can obtain a better selected PRESS criterion yet a *worse* test classification error than an isotropic RBF kernel. In other words, more representational capacity can coexist with a less reliable tuning procedure.

> **Scope check.** This finding does not establish that ARD is intrinsically inferior, nor that every ensemble should use fewer members.

### Two biased protocols the paper diagnoses (§5)

In §5.2, the problematic "median" protocol tunes on the first five training realizations, takes componentwise medians of their hyperparameters, then reuses those settings across all realizations. Test rows in one split can then have influenced selection in another. Averaging also reduces selection variability, so the evaluated procedure is no longer the same as tuning anew in each split. In §5.3, "external" selection uses the whole dataset *before* outer cross-validation, so each outer test row can influence its own hyperparameters. The remedy is to rerun selection internally, within each outer training portion. The revised lab executes this internal-selection distinction.

## Reconstruct the measuring instrument: model architecture

> **In plain terms.** The measuring tool is kernel ridge regression: it scores a new point by how similar it is to the training points. You need to understand it well enough to trust the leave-one-out shortcut, because that shortcut is what makes the selection experiments cheap.

### Kernels and the ARD variant

Kernel ridge regression predicts a real number `f(x)`, then classifies it as +1 when `f(x)≥0`. A **kernel** measures similarity to training rows without explicitly building nonlinear features. The Gaussian radial basis function (RBF) kernel is `exp(−η‖x−z‖²)`. A larger positive `η` makes similarity decay faster with distance. ARD (automatic relevance determination) uses one scale `η_q` per input coordinate: `exp(−Σ_q η_q(x_q−z_q)²)`. When all coordinate scales are equal, ARD reduces to the plain RBF kernel.

**Worked example.** For `x=(.4,.7)`, `z=(−.3,.7)`, and `η=(2,.25)`, the weighted squared distance is `2×.7²+.25×0²=.98`, so the similarity is `exp(−.98)≈.3753`. Changing the *second* scale cannot affect this pair, because the two points share the same second coordinate. You should be able to predict this before running the kernel.

### The augmented KRR system (Eq. 3)

Set up the pieces one at a time. Let `X` contain `n` rows and `d=2` features. Let `K` be its `n×n` kernel matrix, `y` the length-`n` ±1 targets, `α` the length-`n` coefficients, and `b` an unpenalized intercept. The paper's objective is one half of the squared feature-space weight norm plus `Σ(y−f)²/(2λ)`, with a positive regularizer `λ`. Stationarity gives `α=(y−f)/λ` and `Σα=0`. Substituting `f=Kα+b` produces the augmented linear system (paper Eq.3):

`C = [[K + λI, 1], [1ᵀ, 0]];  C [α; b] = [y; 0]`.

Here `I` is the identity matrix and `1` is an all-ones column. The final equation constrains `Σα=0`, and the final unknown supplies `b`. Dropping that row and column would silently change the estimator. The visible code solves this system in float64.

**Prediction and its consequences.** For new points `X*`, predictions are `K(X*,X)α+b`. No labels of `X*` enter fitting or selection. These real-valued scores are not probabilities: the classification head returns +1 for `f≥0` and −1 otherwise, while the selection criterion uses squared score residuals. These signed, unnormalized coefficients differ from TabR's nonnegative softmax neighbor weights; as a distant query's similarities decay, its prediction approaches `b`.

> **Scope check.** The tiny implementation forms the full inverse in order to expose its diagonal. Its cubic solve cost and quadratic storage make it unsuitable for a large benchmark without a more efficient solver.

<!--figure:architecture-->

**How fitting, selection, and inference connect.** The connected diagram separates fitting and selection from inference. For each candidate, the solve already fits all available development rows; LOO scores that fixed recipe through deleted residuals. The minimum then chooses which fitted model is frozen. At inference, only new features enter: the scores feed the squared-error evaluation, and the sign head supplies classification. The following operation table expands one numerical kernel calculation.

<!--figure:kernel-->

### The leave-one-out shortcut

**Leave-one-out (LOO)** removes row `i`, refits the *same fixed hyperparameters* on all other rows, and predicts row `i`. The paper's §2.1 identity gives its deleted residual directly as `r_i^(−i) = α_i / (C⁻¹)_ii`, using the **upper-left n diagonal entries of the augmented inverse**. PRESS is the sum of these squared deleted residuals; the lab divides by `n` to report MSE. Dividing by the fixed row count leaves the candidate ordering unchanged. Ordinary fitted residuals use a model that has already seen `y_i`, so they are a different quantity.

The lab independently verifies this identity against actually leaving out each row and refitting. It also tests the intercept with constant training targets and a distant query. The provided model source is readable in the notebook; your live tasks implement selection, deleted residuals, candidate search, outer nesting, and paired inference.

## Nest the decision that is actually adaptive

> **In plain terms.** To measure a selection procedure honestly, you must redo the selection *inside* each evaluation fold, using only that fold's training rows. If the choice was made once using all the data, then every "held-out" row already helped make the choice.

### What nesting means

An **outer fold** is an evaluation partition. For each outer fold, the remaining rows are the development sample. LOO selection is run again *on that development sample*; the selected model is fitted on those rows and predicts only the outer-held rows. Each row receives exactly one OOF prediction. The final reported outer MSE averages the squared errors across all rows. This estimates the whole selection-and-fit procedure at the outer training size.

**The concrete sizes.** In the worked protocol, a 64-row dataset is split into four equal, label-blind, shuffled blocks. Each internal selection sees 48 rows, and each of its LOO fits effectively trains on 47. After selection, a 48-row refit predicts the held-out 16.

**The contaminated comparator.** In the deliberately contaminated *external* comparator, LOO selection first sees all 64 targets and freezes one candidate for all four outer refits. Those 48-row models still excluded their own evaluation rows from *coefficient fitting* — but their **choice** of hyperparameters already depended on those rows. That is the leak.

<!--figure:protocol-->

**Trace before running.** Change the targets in outer fold 2. Its own internally selected hyperparameters and predictions must stay unchanged. The other folds may change, because those rows legitimately belong to their training portion. The CHECK enforces this distinction. The lesson: merely keeping the *fitting* rows disjoint is not enough to protect a hyperparameter that was selected globally.

### Measuring optimism without confounding it

A model refitted on 64 rows and a model refitted on 48 rows have different evaluation targets. So the experiment creates fresh independent rows only *after* every choice is frozen, and evaluates **each outer-fitted model** on those fresh rows, averaging their fresh MSE within the repetition. Subtracting each protocol's outer score from its own matched fresh score measures optimism without blaming a training-size effect on leakage. The 64-row refit's fresh score separately evaluates the selected-LOO optimism; its LOO fits used 63 rows, so that particular comparison keeps a small training-size difference.

> **Scope check.** Nested evaluation does not repair a poor selector; its job is only to include that selector's mistakes in the estimate. It also does not protect a researcher who redesigns the candidate grid after viewing outer scores — that redesign needs new evidence. For customer histories or temporal data, replace random folds with deployment-appropriate entity and time boundaries and label-availability rules; random nesting cannot remove temporal leakage.

## Measure the boundary, with uncertainty

> **In plain terms.** Now run the honest (internal) protocol and the contaminated (external) one side by side on the same data, and report the gap with an interval. The point is to *measure* the bias, not to win a contest.

### The prespecified protocol

The protocol is fixed before the author run: 27 candidates drawn from `λ∈{.01,.1,1}` and each of two `η` coordinates in `{.25,2,16}`; analytic LOO mean PRESS; first-index ties; four equal outer folds; 64 development rows; 4,096 fresh rows per independent repetition; and no learned preprocessing or feature selection. There are 30 teaching repetitions (seeds 5900–5929), followed by a required 1,000-repetition precision track (seeds 5900–6899). The longer run *contains* the first 30, so they are not independent confirmations.

**The unit of analysis.** The same candidates, development rows, outer folds, and fresh rows are paired across the internal and external protocols. All four folds are averaged **inside** each repetition. The independent units are fresh draws of the entire synthetic dataset — not the four overlapping folds, and not the 4,096 test rows treated as separate experiments.

> **Scope check.** The intervals are paired Student-t intervals for the mean repeat-level difference. They quantify Monte Carlo precision under this generator. They are not benchmark-dataset confidence intervals, and they are exploratory across multiple contrasts.

**Predict.** Will the external outer score look better? Can its *actual* fresh score also be better, because its selection legitimately used more information, even though its *claimed* evaluation was contaminated? Should the internal estimated optimism be exactly zero in a finite run?

<!--figure:nested_results-->

### The 30-repeat teaching run

On the 30-repeat teaching run, the internal outer MSE is .482096 versus a matched fresh MSE of .509442; the optimism estimate is .027346 with a 95% interval of [−.028857,.083549]. The external outer MSE is .416708 versus its matched fresh MSE of .485978; its optimism is .069269 [.022023,.116516]. The internal-minus-external outer-score gap is .065387 [.045771,.085004]. These are distinct quantities. The wider internal interval includes zero; that neither proves the bias is exactly zero nor establishes equivalence. The external fresh score being lower does not make its external *outer* score a valid estimate.

### The 1,000-repeat precision track

The [1,000-repeat evidence](../labs/_verify_l059_closer_results.json) gives an internal matched optimism of .002660 with t95 [−.005395,.010715], and an external matched optimism of .063726 [.055963,.071489]. The external fresh MSE is .492255 versus the internal .510056: the external selector saw 64 labels rather than 48, so this is not an equal-information-budget model contest. The bias comparison evaluates each frozen procedure against its own matched fresh target. The figure shows this precision extension.

> **Scope check.** Read the complete contrast summary rather than picking whichever seed, contrast, or interval best supports the story. A run that fails to show the anticipated pattern would still be valid evidence under the prespecified protocol; the lab's correctness checks do not require a particular statistical victory.

## Audit an ensemble without blaming diversity

> **In plain terms.** An ensemble can leak selection bias too. Even when the mixture weights are fixed, the *choice* of which members to include, and how many steps to run, is itself an adaptive decision that must be protected.

**Where the hidden decisions live.** A uniform mixture over a fixed, declared member set has no fitted weights — but the decision to *choose* that member set can still be adaptive. Greedy ensemble selection fits weights to OOF predictions. Choosing its maximum steps, base library, loss, folds, or preferred prefix each adds a decision. Protect the entire path. Then compare it against a single-family procedure selected by the **same inner evidence**. The family with the best final test loss is an oracle comparator, not a deployable choice.

> **Scope check.** A large OOF-to-test gap can also reflect temporal shift, dependence among customers, a 48-versus-64-row refit difference, or ordinary sampling uncertainty. The current synthetic generator removes temporal shift and controls sample size, so it cannot diagnose every real failure. A useful discriminating experiment freezes all recipes and changes only the amount of selection data, or evaluates the complete procedure on new outer blocks.

**How this connects across the course.** The bigger lesson ties together strong defaults (053), ensembles (057), benchmark construction (058), and the upcoming model comparison (060). Strong defaults reduce local decisions but inherit the assumptions of their original development. Ensembling may reduce prediction variance while increasing selection flexibility. Benchmark reuse can adapt research decisions over years. A fair relational claim compares complete pipelines on aligned future entities and times, gives each a declared tuning budget, and reserves evidence for the final decision. Architectural novelty does not excuse a weaker evaluation boundary.

## Exit: make adaptation countable

> **In plain terms.** The exit makes you count every adaptive decision and tie each reported number to the exact procedure and training size that produced it.

**What to submit.** Finish all five live functions, replay the historical control, run the 30-repeat kernel protocol and the required 1,000-repeat precision track, then submit `data/cache/l059-student/exit.json` and your completed notebook. The artifact records the numerical results, your live function sources, the source-evidence hashes, and an adaptation ledger. No learner mastery is inferred from merely building or executing the teacher notebook.

**What to write.** In 80–180 words, identify: the selected object; why coefficient-level OOF exclusion is insufficient; the training size behind each estimate; one measured interval and its unit; and one next experiment that could contradict your interpretation. Freeze a lesson-060 candidate list, metric, split identity, selection rule, and budget. Name one tempting adjustment you will postpone until a separately labeled follow-up. Ask the tutor to check this argument, especially any numerical result that seems to contradict the expected mechanism.

## Reproduction boundary and reading route

> **In plain terms.** This closing section states, plainly, exactly what was and was not reproduced — so no reader mistakes this teaching run for a reproduction of the paper's original results.

### What each layer establishes

- **Equation validation.** The augmented KRR system and the exact LOO identity pass independent refit tests.
- **Historical evidence.** All 1,000 null rows replay under the preserved operator.
- **New procedure.** Internal/external selection and matched fresh evaluation run with your live notebook functions.
- **Original results.** INCOMPARABLE, not reproduced.

> **Scope check.** The new run uses the paper's exact synthetic distribution and kernel/LOO equations, but substitutes a finite 27-candidate ARD grid and an outer four-fold protocol. It does not reproduce Figure 2's iterative optimizer/four-fold inner criterion, Figures 4–8's SER surface and fixed fitting sample, or Table 8's thirteen real benchmarks and outer ten-fold RBF experiment. Matching 1,000 repetitions is a precision choice, not protocol parity.

### What was inspected in the original source

The original HTTP GKM link in footnote 2 works, although its HTTPS endpoint fails certificate verification. We pinned the downloaded archive and inspected its KRR, RBF, LOO, criterion, cross-validation, and simplex source. Its KRR implementation uses a Cholesky solve with the same intercept and a Schur-corrected inverse diagonal. A translated source-algebra check agrees on 18 fixtures.

> **Scope check.** Native MATLAB execution, optimizer trajectories, and exact 2010 experiment-version parity are not established. The source's SSE adapter reports half the residual sum of squares, while our fixed-`n` mean PRESS has the same argmin. See the [source comparison](../labs/_source_check_l059_v2_results.json) and [pinned reference files](../labs/sources/l059/README.md). There is no appendix in this 29-page article; the full substantive §§2–6 and the figure/table captions were inspected.

### Reading route

Use the [reference card](../reference/0059-validation-set-overfitting.html), the [equation/behavior checks](../labs/_check_l059_v2_results.json), the [source inventory](../labs/_sources_l059_v2.json), the [measured evidence](../labs/_verify_l059_v2_results.json), and the [reproduction commands](../labs/l059-reproduction.md) together. The last of these documents the operator and what remains unrun. Read §4.1 for criterion variance, §4.4 for regularization and averaging strategies, and §§5.1–5.3 for the full evaluation boundary; none of them makes repeated final-test use free.
