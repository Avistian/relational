/**
 * Spaced-retrieval question pool (single source of truth for RetrievalBank).
 *
 * Loaded as a plain <script> so it works on file:// and GitHub Pages (no fetch/CORS).
 * Assigns window.RETRIEVAL_POOL = [ item, ... ].
 *
 * Each item:
 *   id          unique string, stable forever (Leitner state is keyed on it) — never renumber
 *   lesson      integer origin lesson (used for spacing: only resurfaces in lessons AFTER this)
 *   quarter     "Q1" | "Q2" | ...  (used to interleave across quarters)
 *   concept     short tag (used to avoid two same-concept items in one warm-up)
 *   question    prompt (retrieval — recall from memory, not recognition of just-read text)
 *   options     [{ label, value }]  — keep labels similar length (quiz-fairness standard #2)
 *   correct     value of the correct option
 *   explain     one-sentence why (shown after answering)
 *   misconception  true if this item mirrors a row in misconceptions.md (kept in sync)
 *
 * ADD an item whenever a lesson ships a durable, testable idea, and whenever a misconception
 * is logged in misconceptions.md. Do NOT change an existing id.
 */
(function (global) {
  "use strict";

  global.RETRIEVAL_POOL = [
    // ---- Q1: leakage & evaluation spine ----
    {
      id: "l001-forcing", lesson: 1, quarter: "Q1", concept: "design-matrix",
      question: "What forces every prediction to use only information available at prediction time?",
      options: [
        { label: "The point-in-time cutoff on the design matrix", value: "a" },
        { label: "Using a larger model with more parameters", value: "b" },
        { label: "Shuffling the rows before you train it", value: "c" },
        { label: "Scaling all of the columns to unit norm", value: "d" }
      ],
      correct: "a",
      explain: "The design matrix is built as of a cutoff time, so features can only see the past — that is the structural mechanism that prevents temporal leakage."
    },
    {
      id: "l002-pit", lesson: 2, quarter: "Q1", concept: "point-in-time",
      question: "A point-in-time aggregate over a customer's past orders must exclude which rows?",
      options: [
        { label: "Any order dated at or after the cutoff", value: "a" },
        { label: "Any order that has a missing amount", value: "b" },
        { label: "Any order from a different customer", value: "c" },
        { label: "Any order older than one full year", value: "d" }
      ],
      correct: "a",
      explain: "A PIT aggregate may only see events strictly before the prediction cutoff; including same-or-later orders leaks the future into the feature."
    },
    {
      id: "l004-bestscore", lesson: 4, quarter: "Q1", concept: "nested-cv", misconception: true,
      question: "Why is a grid search's reported `best_score_` an optimistic estimate of true performance?",
      options: [
        { label: "It is a validation score chosen by the same data", value: "a" },
        { label: "It is measured only on the training partition", value: "b" },
        { label: "It always uses far too few cross-validation folds", value: "c" },
        { label: "It secretly averages in the test-set accuracy too", value: "d" }
      ],
      correct: "a",
      explain: "`best_score_` is the best validation score, but it was selected using that same validation data — selection bias. A nested CV gives the honest number."
    },
    {
      id: "l005-perfold", lesson: 5, quarter: "Q1", concept: "pipeline-scope", misconception: true,
      question: "Where must a data-learning step (scaler, SelectKBest, encoder) be fit to avoid leakage?",
      options: [
        { label: "Inside the pipeline, per fold, on training data", value: "a" },
        { label: "Once on the full dataset before you split", value: "b" },
        { label: "Once on the test set to match its statistics", value: "c" },
        { label: "Separately on each column after prediction", value: "d" }
      ],
      correct: "a",
      explain: "Fitting on all data before CV leaks (0.78 vs 0.44 honest on pure noise). Wrap it in a Pipeline so each fold fits on its own training portion."
    },
    {
      id: "l006-mnar", lesson: 6, quarter: "Q1", concept: "missingness",
      question: "Missingness that depends on the unobserved value itself is called which mechanism?",
      options: [
        { label: "MNAR — missing not at random", value: "a" },
        { label: "MCAR — missing completely at random", value: "b" },
        { label: "MAR — missing at random given X", value: "c" },
        { label: "MICE — multiple imputed chained eqs", value: "d" }
      ],
      correct: "a",
      explain: "MNAR means the probability of being missing depends on the missing value itself, which imputation cannot fully repair; an indicator preserves the pattern."
    },
    {
      id: "l007-smote", lesson: 7, quarter: "Q1", concept: "imbalance", misconception: true,
      question: "How do you resample an imbalanced dataset with SMOTE without leaking?",
      options: [
        { label: "Resample inside each CV fold via a pipeline", value: "a" },
        { label: "Resample the whole dataset, then split it", value: "b" },
        { label: "Resample the test fold to match the train", value: "c" },
        { label: "Resample only after the model is fitted", value: "d" }
      ],
      correct: "a",
      explain: "SMOTE before the split synthesises points from rows that later land in validation (F1 0.887 mirage vs 0.479 honest). Keep it inside the fold with an imblearn Pipeline."
    },
    {
      id: "l008-prbaseline", lesson: 8, quarter: "Q1", concept: "pr-baseline", misconception: true,
      question: "A no-skill classifier's PR-AUC equals what value?",
      options: [
        { label: "The prevalence (positive rate) of the data", value: "a" },
        { label: "Exactly one half, just like ROC-AUC does", value: "b" },
        { label: "Zero, because it never ranks anything well", value: "c" },
        { label: "One minus the prevalence of the positives", value: "d" }
      ],
      correct: "a",
      explain: "Unlike ROC (baseline 0.5), the PR baseline is the prevalence, so PR curves are the honest lens on rare-positive problems."
    },
    {
      id: "l008-calib", lesson: 8, quarter: "Q1", concept: "calibration", misconception: true,
      question: "On what data must a probability calibrator (isotonic/sigmoid) be fit?",
      options: [
        { label: "Held-out data the base model did not train on", value: "a" },
        { label: "The same training data as the base model used", value: "b" },
        { label: "The test set, to match its distribution best", value: "c" },
        { label: "The full dataset including every prediction row", value: "d" }
      ],
      correct: "a",
      explain: "Fitting the calibrator on the base model's own training data leaks; use held-out folds (CalibratedClassifierCV handles this)."
    },
    {
      id: "l009-target-enc", lesson: 9, quarter: "Q1", concept: "target-encoding", misconception: true,
      question: "Target encoding a high-cardinality category is safe only when computed how?",
      options: [
        { label: "Out-of-fold, so a row never sees its own label", value: "a" },
        { label: "On the full column mean for every single row", value: "b" },
        { label: "After the model has already been trained once", value: "c" },
        { label: "Using the test-set labels to fill rare levels", value: "d" }
      ],
      correct: "a",
      explain: "Whole-column target encoding leaks the label (0.76 vs 0.50 on a pure-noise category); out-of-fold encoding is the leak-free version."
    },

    // ---- Q2: trees, boosting, ensembling ----
    {
      id: "l011-gini", lesson: 11, quarter: "Q2", concept: "gini", misconception: true,
      question: "For a binary node with positive fraction p, what is the Gini impurity?",
      options: [
        { label: "1 − p² − (1−p)²", value: "a" },
        { label: "p, the positive fraction itself", value: "b" },
        { label: "−p·log p − (1−p)·log(1−p)", value: "c" },
        { label: "p² + (1−p)², the purity term", value: "d" }
      ],
      correct: "a",
      explain: "Gini = 1 − Σpᵢ² = 1 − p² − (1−p)²: the chance two random draws from the node have different labels (option c is entropy)."
    },
    {
      id: "l011-gain", lesson: 11, quarter: "Q2", concept: "split-gain", misconception: true,
      question: "How is the gain of a decision-tree split computed?",
      options: [
        { label: "Parent impurity − weighted children impurity", value: "a" },
        { label: "Weighted children impurity − parent impurity", value: "b" },
        { label: "Sum of the two children impurities directly", value: "c" },
        { label: "Parent impurity times the number of children", value: "d" }
      ],
      correct: "a",
      explain: "Gain is the impurity drop: parent minus the weighted average of the children. In XGBoost the regularized version also subtracts a γ toll."
    },
    {
      id: "l014-leafweight", lesson: 14, quarter: "Q2", concept: "leaf-weight", misconception: true,
      question: "In XGBoost, the optimal weight of a leaf is which expression?",
      options: [
        { label: "w* = −G / (H + λ)", value: "a" },
        { label: "w* = the mean residual in the leaf", value: "b" },
        { label: "w* = −H / (G + λ)", value: "c" },
        { label: "w* = G / (H − λ)", value: "d" }
      ],
      correct: "a",
      explain: "w* = −(sum of gradients)/(sum of hessians + λ). λ shrinks the weight toward zero; this matched XGBoost's leaf output exactly in the lab."
    },
    {
      id: "l014-gh", lesson: 14, quarter: "Q2", concept: "grad-hess",
      question: "In gradient boosting, g and h for a row are derivatives of the loss with respect to what?",
      options: [
        { label: "The current model score F for that row", value: "a" },
        { label: "The raw input feature values of the row", value: "b" },
        { label: "The learning-rate hyperparameter η used", value: "c" },
        { label: "The leaf weights of the previous whole tree", value: "d" }
      ],
      correct: "a",
      explain: "g and h are the first/second derivatives of the loss w.r.t. the current additive score F, recomputed each round (squared: g=F−y, h=1; logistic: g=p−y, h=p(1−p))."
    },
    {
      id: "l015-numleaves", lesson: 15, quarter: "Q2", concept: "leaf-wise", misconception: true,
      question: "What is LightGBM's primary capacity knob, given how it grows trees?",
      options: [
        { label: "num_leaves, because growth is leaf-wise", value: "a" },
        { label: "max_depth, because growth is level-wise", value: "b" },
        { label: "n_estimators, the number of boosting rounds", value: "c" },
        { label: "learning_rate, the shrinkage per each round", value: "d" }
      ],
      correct: "a",
      explain: "LightGBM grows best-first (leaf-wise), so num_leaves controls capacity; max_depth only caps it. Level-wise growers lean on max_depth instead."
    },
    {
      id: "l015-goss", lesson: 15, quarter: "Q2", concept: "goss", misconception: true,
      question: "How does GOSS stay unbiased after subsampling small-gradient rows?",
      options: [
        { label: "It amplifies the sampled rows by (1−a)/b", value: "a" },
        { label: "It simply discards every small-gradient row", value: "b" },
        { label: "It reweights the large-gradient rows downward", value: "c" },
        { label: "It re-runs each split on the full dataset again", value: "d" }
      ],
      correct: "a",
      explain: "GOSS keeps the top-a gradients and samples b of the rest, then scales those sampled rows by (1−a)/b so the gain estimate stays unbiased."
    },
    {
      id: "l016-orderedts", lesson: 16, quarter: "Q2", concept: "ordered-ts", misconception: true,
      question: "Why does CatBoost use ordered target statistics instead of a whole-column mean?",
      options: [
        { label: "A whole-column mean leaks a row's own label", value: "a" },
        { label: "A whole-column mean is far slower to compute", value: "b" },
        { label: "Ordered TS need less memory than a full mean", value: "c" },
        { label: "Ordered TS let categories be one-hot encoded", value: "d" }
      ],
      correct: "a",
      explain: "Greedy TS encodes a row using its own target (worst on rare categories); ordered TS encodes row i from its permutation prefix plus a prior, killing the leak."
    },
    {
      id: "l017-random", lesson: 17, quarter: "Q2", concept: "random-search",
      question: "Why does random search beat grid search at equal budget when only a few knobs matter?",
      options: [
        { label: "It tries ~n distinct values on the key axis", value: "a" },
        { label: "It always samples the exact optimum by luck", value: "b" },
        { label: "It evaluates each configuration far more times", value: "c" },
        { label: "It ignores the unimportant knobs entirely first", value: "d" }
      ],
      correct: "a",
      explain: "A grid spends its budget on √n distinct values per axis; random search gives ~n distinct values on the important axis, so it explores it far better under low effective dimensionality."
    },
    {
      id: "l018-oof", lesson: 18, quarter: "Q2", concept: "stacking", misconception: true,
      question: "A stack's meta-learner must train on base predictions produced how?",
      options: [
        { label: "Out-of-fold, so bases never predict their own train rows", value: "a" },
        { label: "In-sample, using the bases' own training rows", value: "b" },
        { label: "On the test set, to match the final distribution", value: "c" },
        { label: "On a single fixed 50/50 split, computed just once", value: "d" }
      ],
      correct: "a",
      explain: "In-sample base preds leak — the meta-learner crowns the memorizer (1-NN weight +3.00, train PR-AUC 1.000 mirage). Out-of-fold meta-features are honest (test 0.930)."
    },
    {
      id: "l019-biases", lesson: 19, quarter: "Q2", concept: "inductive-bias", misconception: true,
      question: "What is the honest reason tree ensembles win on typical tabular data?",
      options: [
        { label: "Irregular targets, junk features, and orientation together", value: "a" },
        { label: "Trees are simply more powerful than any neural net", value: "b" },
        { label: "Neural nets cannot represent non-linear functions", value: "c" },
        { label: "Trees always have strictly more parameters to fit", value: "d" }
      ],
      correct: "a",
      explain: "With clean, smooth, all-informative features an MLP won. Trees win because typical tabular data has non-smooth targets, many junk columns, and meaningful axes at once."
    },
    {
      id: "l019-rotation", lesson: 19, quarter: "Q2", concept: "rotation",
      question: "Why is a tree's lack of rotational invariance an advantage on tables?",
      options: [
        { label: "Individual columns are meaningful, so axes matter", value: "a" },
        { label: "Rotations are too expensive to apply to a tree", value: "b" },
        { label: "Invariance always forces a model to overfit fast", value: "c" },
        { label: "Trees rotate the data internally before splitting", value: "d" }
      ],
      correct: "a",
      explain: "Columns like age or balance mean something individually; a tree splits on single axes to exploit that, while a rotation-invariant MLP can't tell a real column from a scrambled mixture."
    },

    // ---- Q3: evaluation rigor & benchmark literacy ----
    {
      id: "l021-temporal", lesson: 21, quarter: "Q3", concept: "temporal-split", misconception: true,
      question: "On time-ordered, drifting data, why does a random split report an optimistic score?",
      options: [
        { label: "It puts deployment-period rows into the training folds", value: "a" },
        { label: "It shrinks the training set below a usable size", value: "b" },
        { label: "It changes the scoring metric to PR-AUC quietly", value: "c" },
        { label: "It always selects too few cross-validation folds", value: "d" }
      ],
      correct: "a",
      explain: "A random split scatters test rows across all time, so training folds share the deployment period; under drift the model learns the current rule it is scored on (random-CV 0.846 vs temporal 0.758)."
    },
    {
      id: "l021-timeseriessplit", lesson: 21, quarter: "Q3", concept: "timeseries-cv",
      question: "How does TimeSeriesSplit differ from a shuffled KFold?",
      options: [
        { label: "Each fold trains on the past, tests on the next block", value: "a" },
        { label: "It stratifies every fold to the base positive rate", value: "b" },
        { label: "It resamples the minority class within each fold", value: "c" },
        { label: "It fits the scaler once on the whole dataset first", value: "d" }
      ],
      correct: "a",
      explain: "TimeSeriesSplit is an expanding window: every fold trains on an earlier block and validates on the next, never shuffling, so it cannot leak the future into training."
    },
    {
      id: "l022-illegit", lesson: 22, quarter: "Q3", concept: "illegitimate-feature", misconception: true,
      question: "What makes a feature 'illegitimate' (leak type L2), and why is it so hard to catch?",
      options: [
        { label: "Its value would not be knowable at prediction time", value: "a" },
        { label: "It has too many missing values to be useful", value: "b" },
        { label: "It is on a different scale than the other columns", value: "c" },
        { label: "It is only weakly correlated with the target", value: "d" }
      ],
      correct: "a",
      explain: "L2 is a provenance leak — a column computed from/after the outcome. It passes every pipeline and split check (it just looks very predictive), so only provenance auditing catches it."
    },
    {
      id: "l022-collapse", lesson: 22, quarter: "Q3", concept: "repro-collapse",
      question: "A complex model beats logistic regression by a huge margin. Why treat that as a leak hypothesis first?",
      options: [
        { label: "A flexible model exploits a leak harder, inflating the gap", value: "a" },
        { label: "Complex models are always overfit by construction", value: "b" },
        { label: "Logistic regression cannot model any real signal", value: "c" },
        { label: "Accuracy is the wrong metric for every dataset", value: "d" }
      ],
      correct: "a",
      explain: "Kapoor & Narayanan's civil-war study: remove the leak and complex ≈ LR (our demo: gap +0.217 → −0.009). A big gap measures leak-exploitation, not modelling skill."
    },
    {
      id: "l023-corrected", lesson: 23, quarter: "Q3", concept: "significance-testing", misconception: true,
      question: "Why is a paired t-test on k-fold CV fold scores anticonservative (too many false 'wins')?",
      options: [
        { label: "Overlapping training folds correlate the scores, so it underestimates variance", value: "a" },
        { label: "It needs at least thirty folds before the t-distribution is valid", value: "b" },
        { label: "Accuracy is the wrong metric to average over the folds", value: "c" },
        { label: "It silently ignores the class imbalance within each fold", value: "d" }
      ],
      correct: "a",
      explain: "k-fold training sets share most rows, so fold scores are positively correlated; the naive SE σ/√n assumes independence and is too small (naive p=1.2e−5 vs corrected 0.19). Fix: the corrected resampled t-test."
    },
    {
      id: "l023-cd-diagram", lesson: 23, quarter: "Q3", concept: "rank-tests",
      question: "Comparing many models over many datasets, when are two models 'not significantly different'?",
      options: [
        { label: "Their average ranks differ by less than the critical difference", value: "a" },
        { label: "Their mean accuracies differ by less than one percent", value: "b" },
        { label: "The Friedman omnibus test failed to reject its null", value: "c" },
        { label: "They were both trained with the same random seed", value: "d" }
      ],
      correct: "a",
      explain: "Demšar: rank per dataset, Friedman omnibus, then Nemenyi CD = q_α·√(k(k+1)/6N). Ranks closer than CD are joined in a CD diagram — here LogReg (1.08) and NB (2.08), gap 1.0 < CD 1.354."
    },
    {
      id: "l024-budget-curve", lesson: 24, quarter: "Q3", concept: "benchmark-protocol", misconception: true,
      question: "Why does the Grinsztajn benchmark report a random-search budget curve instead of each model's single best tuned score?",
      options: [
        { label: "One tuned number hides how much tuning each model needed and its default quality", value: "a" },
        { label: "Random search always reaches a higher optimum than grid search", value: "b" },
        { label: "A curve is required to run the Friedman significance test", value: "c" },
        { label: "It removes the need for any validation split", value: "d" }
      ],
      correct: "a",
      explain: "A single peak conflates model quality with tuning effort and erases the default/ceiling split. The budget curve (select by validation, report test, avg over orderings) shows both — on credit-g the GBT leads at every budget (default gap +0.062 → tuned +0.015)."
    },
    {
      id: "l024-normalization", lesson: 24, quarter: "Q3", concept: "benchmark-protocol",
      question: "Why does Grinsztajn affine-normalize test scores (worst→0, best→1) per dataset before averaging across the suite?",
      options: [
        { label: "Raw accuracies are incommensurable across datasets, so one would dominate the mean", value: "a" },
        { label: "It converts each accuracy into a calibrated probability", value: "b" },
        { label: "Normalization forces every model to average exactly 0.5", value: "c" },
        { label: "It lets you skip having a separate test set", value: "d" }
      ],
      correct: "a",
      explain: "An 80% on an easy vs a hard dataset are not comparable (echo of Demšar, L023). Per-dataset min-max scaling bounds each dataset to [0,1] so the aggregate isn't hijacked by one high-variance task."
    },
    {
      id: "l025-smoothness", lesson: 25, quarter: "Q3", concept: "inductive-bias", misconception: true,
      question: "Why do neural nets underperform trees on typical tabular data — and what proves it is the reason?",
      options: [
        { label: "The MLP's smoothness bias mismatches irregular targets; smoothing the target closes the gap", value: "a" },
        { label: "Trees simply have far more parameters than any neural network does", value: "b" },
        { label: "Neural nets cannot represent non-linear or non-smooth functions at all", value: "c" },
        { label: "Trees are trained with a much smaller learning rate than the MLP", value: "d" }
      ],
      correct: "a",
      explain: "An MLP is biased toward smooth (low-frequency) functions (Rahaman spectral bias); real tabular targets are irregular. Gaussian-smoothing the target erases the high-frequency structure and the tree-vs-MLP gap collapses (repro: +0.33 R² → ~0) — the edge was the irregularity, not raw model strength."
    },
    {
      id: "l025-piecewise", lesson: 25, quarter: "Q3", concept: "smoothness",
      question: "What lets a decision tree follow a jagged target that an MLP over-smooths?",
      options: [
        { label: "It is piecewise-constant, so one split makes an arbitrarily sharp step", value: "a" },
        { label: "It fits a global smooth curve through all of the training points", value: "b" },
        { label: "It rotates the feature space until the target looks smooth", value: "c" },
        { label: "It always trains for many more gradient steps than the MLP", value: "d" }
      ],
      correct: "a",
      explain: "A tree partitions space into axis-aligned boxes and predicts a constant per box, so a single split produces a sharp jump — the natural way to render a high-frequency target. Gradient descent, by contrast, reaches smooth functions first (spectral bias)."
    },
    {
      id: "l026-rotation", lesson: 26, quarter: "Q3", concept: "rotation", misconception: true,
      question: "Randomly rotating the features (same Q on train+test) barely moves an MLP but collapses a tree. Why?",
      options: [
        { label: "Rotation is lossless but scrambles the axes, and the tree's edge was the axis alignment; the MLP is invariant", value: "a" },
        { label: "Rotation injects noise, which hurts the higher-capacity tree more than the MLP", value: "b" },
        { label: "Rotation deletes the uninformative features that the tree depended on", value: "c" },
        { label: "Rotation regularizes the tree, which had simply been overfitting the data", value: "d" }
      ],
      correct: "a",
      explain: "A rotation Q is invertible (no information lost); it only destroys the alignment of the signal with the meaningful columns. An MLP's linear first layer absorbs it (W·(Qx)=(WQ)·x) so it is invariant (+0.008), while the axis-aligned tree collapses (−0.24) — the ranking reverses."
    },
    {
      id: "l026-invariance", lesson: 26, quarter: "Q3", concept: "rotation",
      question: "Which model is (essentially) rotationally invariant, and why is that a liability on tabular data?",
      options: [
        { label: "The MLP — its linear first layer absorbs any rotation, but tabular columns are individually meaningful so the axes carry signal", value: "a" },
        { label: "The decision tree — its axis-aligned splits already ignore orientation", value: "b" },
        { label: "The random forest — bagging averages away any rotational structure", value: "c" },
        { label: "None — every model behaves identically under a rotation of the features", value: "d" }
      ],
      correct: "a",
      explain: "An MLP/ResNet is rotation-invariant (W·(Qx)=(WQ)·x); a tree and an FT-Transformer are not. On tables the original basis is privileged (age, balance mean something), so invariance blends meaningful columns away — Ng 2004: sample complexity then grows ≥ linearly in the number of junk features."
    },
    {
      id: "l027-uninformative", lesson: 27, quarter: "Q3", concept: "inductive-bias", misconception: true,
      question: "You add many pure-noise columns to a table where an MLP was winning. What happens, and why?",
      options: [
        { label: "The MLP degrades much faster than the trees and loses its lead — it has no gate against junk", value: "a" },
        { label: "Nothing changes — noise columns carry no signal, so every model ignores them equally", value: "b" },
        { label: "The trees degrade faster because they overfit the extra columns", value: "c" },
        { label: "Every model improves — extra columns act as regularization", value: "d" }
      ],
      correct: "a",
      explain: "Grinsztajn §5.3 (Finding 2): MLPs are not robust to uninformative features. A tree gates junk out by gain (never splits on ~0-gain noise); an MLP wires every feature into its first layer and (rotationally invariant, Ng 2004) needs ≥ linearly more samples per junk feature. Repro: adding 100 junk cols cost the MLP 0.084 vs the GBT 0.032, reversing the ranking."
    },
    {
      id: "l027-gate", lesson: 27, quarter: "Q3", concept: "implicit-feature-selection",
      question: "What is the mechanism that makes a decision tree robust to uninformative features?",
      options: [
        { label: "Greedy, gain-gated split selection — a noise column's best split has ~0 gain, so it is never chosen (implicit feature selection)", value: "a" },
        { label: "It standardizes every feature, which cancels out noise columns exactly", value: "b" },
        { label: "It is rotationally invariant, so it treats all features symmetrically", value: "c" },
        { label: "Bagging averages many trees, which deletes noise features automatically", value: "d" }
      ],
      correct: "a",
      explain: "At each node the tree keeps only the highest-gain split; a pure-noise column removes almost no impurity (root-split gain ~118× lower than an informative feature), so it is gated out for free. Caveat: deep spurious noise splits are why MDI importance over-credits noise — measure root gain, not MDI."
    }
  ];
})(window);
