## The validation set becomes training data for your decisions

Your goal is to diagnose a procedure that improves its validation score while learning nothing useful. A model's training loss measures how its parameters fit training examples. Its validation loss guides decisions such as hyperparameters, stopping epochs, preprocessing, retrieved context sizes and ensemble weights. Once a decision depends on validation labels, those labels have trained the **selection procedure**, even if no gradient used them.

This extends lesson 57: an out-of-fold prediction excludes a row from fitting its base predictor, but the row's label still fits the combiner. Read [Cawley and Talbot, Section 2](https://jmlr.org/papers/v11/cawley10a.html) for the distinction between overfitting the selection criterion and overfitting model parameters. Read [TabArena's protocol](https://arxiv.org/html/2506.16791v1) with that distinction in mind. A benchmark can constrain individual trials while users adapt repeatedly to its reported results.

## A minimum selects lucky noise

Write each candidate's observed validation error as true error plus estimation noise: `E_hat_j = E_j + epsilon_j`. Selection chooses the smallest observed value. Even when each noise term is centered at zero, the noise of the selected candidate tends to be negative. Independence is not required for this effect, although strong correlation between candidates can reduce the effective search size.

For three equally good candidates with true error `.30`, suppose noise is `+.03, −.04, +.01`. Validation reports `.33, .26, .31`, so the second candidate wins. Its expected new-data error remains `.30` under the equal-quality assumption. The apparent gain is a lucky selection event. Repeating candidate construction after inspecting validation makes the family of decisions still larger.

<!--figure:mechanism-->

## Run a negative control you can explain completely

The lab generates balanced random labels and independent random candidate predictions. Each candidate's expected classification error is one half. It selects among 1, 4, 16, 64 or 256 candidates using 80 validation examples, then evaluates that frozen choice on 2,000 separately generated test examples. The validation and test streams use separate random-number generators, so changing the number of candidates cannot accidentally reuse validation draws as test data.

Across 200 repetitions, report mean validation error, test error and **optimism**, defined here as `test_error − validation_error`. Positive optimism means the selected validation score looked too favorable. The standard error across these independent synthetic repetitions quantifies Monte Carlo precision; it is not a confidence interval for a real tabular benchmark. Candidate prefixes are nested within a repetition, which makes budget comparisons paired.

Predict the two curves before revealing them. Increasing search should improve the selected validation minimum; it should not systematically improve independent test performance when every candidate is noise. A single repetition can fluctuate in either direction. That exception is useful: the statistical statement concerns the selection distribution, not monotonic test error in every run.

<!--figure:results-->

## Nest the decision that is actually adaptive

An outer split evaluates an entire procedure. Within its training portion, create inner fitting and validation data; perform all recipe and ensemble decisions there; refit only according to a declared rule; then score once on the outer test. Repeat outer splits when your evaluation target allows it. For temporal deployment, preserve chronology rather than randomly nesting future rows into earlier fitting sets.

The common mistake is to nest only the last numerical optimizer. If you chose features, model families or an ensemble library after reading outer scores, that redesign also belongs inside the evaluated procedure—or requires new evaluation evidence. A second test set is not permanently untouched once you repeatedly use it to guide decisions. [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html).

OOF predictions alone are not a certificate against meta-level leakage. Suppose you train a head on OOF embeddings from several folds: check whether any “held-out” meta-evaluation target influenced the base models that generated the head's fitting features. Distinguish an audit of frozen predictions from a fresh nested evaluation. Record who fitted each layer and which labels they could access.

## Audit an ensemble without blaming diversity

For the lesson-57 combiner, count the base-family recipes, fold schemes, preprocessing variants, weight searches, maximum step counts and prefix-selection rules that were tried. Compare a prespecified uniform mixture, a prespecified greedy procedure and a single family selected by the same inner evidence. Do not call the family with minimum test loss a deployable best-single comparator; that identity is an oracle available only after evaluation.

A growing OOF-to-test gap can arise from selection overfitting, distribution shift, different fold versus refit behavior, or ordinary uncertainty. The pure-noise experiment isolates one mechanism, not every explanation for a disappointing ensemble. A useful diagnosis proposes a discriminating experiment: freeze the whole procedure and evaluate a new outer split, or hold predictions fixed while varying only the amount of selection data.

## Exit: make adaptation countable

Implement the selector without a test-label argument, run the negative control and report paired budget differences. Then write a decision ledger for lesson 60 with the metric, candidate lists, stopping rule, split identities and a freeze point. Name one tempting adjustment you will postpone until a separately labeled follow-up experiment. Explain why a better validation number can coexist with no improvement in the underlying predictor.
