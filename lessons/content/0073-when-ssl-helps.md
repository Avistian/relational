## Start with retrieval

Without reopening Lesson 72, write three answers: What does SCARF predict during pretraining? Which parameters move in a frozen probe? Can unlabeled test rows enter an inductive training pipeline? Then complete the spaced warm-up.

<div id="warmup"></div>

**Your tangible win:** build a fair label-budget curve, compute paired gains, and explain whether the measured grid contains a crossover. Finishing the notebook is not the same as demonstrating that explanation.

[Lesson 71](0071-vime-masked-tabular-ssl.html) introduced learning by repairing corrupted rows. [Lesson 72](0072-scarf-subtab-contrastive-views.html) introduced learning by recognizing companion views. Here the question changes: when does that additional training help the prediction task? This is an experimental-design lesson; it introduces no new model.

<!-- depth-walkthrough:start -->
<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · LESSON 073</p><p><strong>Build on what you know.</strong> Lessons 71–72 supplied mechanisms. Lessons 56 and 60 supplied dataset-balanced comparison discipline. Here the intervention is pretraining, so matching the supervised architecture is essential.</p><p><strong>The next question.</strong> <a href="0074-carte-cross-table-transfer.html">Lesson 74</a> changes a different assumption: the source and target can have different schemas. A label-efficiency gain within one fixed schema does not answer that transfer question.</p><p><a href="../reference/0071-0090-model-map.html">Open the SSL → relational → graph model map</a> · Work the cold retrieval first, then spend 15–20 minutes tracing this overview before the detailed mechanism and lab.</p></div>

## Make “SSL helps” a measurable statement

<figure class="model-map"><div class="model-map-scroll" tabindex="0" role="region" aria-label="Architecture diagram; scroll horizontally on narrow screens"><img src="../assets/architectures/073-budget.svg" alt="BUDGET architecture: follow the labeled data, model, loss and prediction paths. A step-by-step text explanation follows." loading="lazy"></div><figcaption>Read the arrows as data dependencies. Teal: learned computation; amber: training objective; violet: readout or prediction. This is a computation overview; exact settings and paper/release differences are specified below.</figcaption></figure>

### Paper reading: distinguish the claim from its conditions

Use [SCARF §4.3](https://arxiv.org/html/2106.15147v2#S4.SS3) to identify the label-limited experiment, and [Oliver et al.'s evaluation study](https://arxiv.org/html/1804.09170v2) to ask what makes an SSL comparison informative. Record the supervised baseline, access to unlabeled examples, and validation procedure. This lesson's budget sweep is a local controlled experiment; its crossover analysis is not an additional claim made by those papers.

### Construct one valid point on the curve

1. **Freeze the row split first.** Fit scaling on training rows, then hide training labels from the pretrainer. This makes “unlabeled” a statement about access to y, not permission to use arbitrary future or test rows.
2. **Create a nested label ladder.** If the training budget grows from 20 to 40 labels, keep the original 20. Otherwise the curve changes both label count and which examples are available. Repeat the ladder across declared seeds to expose this dependence.
3. **Match the comparison within a seed.** Suppose SSL accuracies are [0.80,0.75,0.85] and scratch accuracies are [0.78,0.76,0.81]. The paired gains are [+2,−1,+4] percentage points, with mean +1.67. Preserve those pairs rather than comparing each SSL run against whichever scratch run looks convenient.
4. **Count all labels used to choose the model.** Forty training labels plus 100 validation labels means 140 development labels. Validation is necessary, but it is not free annotation. Two methods that tune with different amounts of labeled validation data have different budgets.
5. **Interpret a crossing conservatively.** If the observed gain changes from positive at 20 labels to negative at 40, the sampled curve brackets a sign change. It does not establish that every dataset has a critical budget between 20 and 40. More seeds reduce Monte Carlo uncertainty within this experiment; more independent datasets address a different question.

<details><summary>Check: does winning against a random frozen encoder establish useful SSL?</summary><p>It establishes improvement over that frozen comparator. It does not establish improvement over an equally trained supervised model, a strong tree baseline, or a cheaper end-to-end recipe. The comparator determines the scope of the conclusion.</p></details>

**Your intermediate artifact:** write the five row sets—pretraining, labeled training, validation, test, and donor pool—and draw every allowed overlap. Submit this alongside the curve so its information budget can be reconstructed.

<!-- depth-walkthrough:end -->

## 1 · Turn “helps” into a comparison

> **In plain terms.** A representation can learn something without becoming your best predictor. Name the alternative before declaring success.

**Self-supervised learning** creates training targets from features rather than human annotations. In SCARF, a clean row identifies its corrupted companion among a batch of candidates. **Semi-supervised learning** uses both labeled and unlabeled examples for a prediction task. Self-supervised pretraining followed by supervised learning is one way to work in that setting. The abbreviation SSL has been used for both; here it means self-supervision unless stated otherwise.

The [tabular SSL survey, §§2–5](https://arxiv.org/html/2402.01204v3) organizes predictive, contrastive, hybrid and transfer approaches. Those categories describe how a representation is learned. They do not supply a universal label fraction at which it becomes useful.

**A frozen probe** holds the encoder fixed and trains a classifier on its representations. **Fine-tuning** updates the encoder as well as the new classification head. **Training from scratch** starts the same encoder from its original random weights. The [SCARF paper, Figure 1 and §3](https://arxiv.org/html/2106.15147v2) discards its projection head and fine-tunes the encoder and classification head. Our preceding frozen-probe lab measured a different transfer procedure.

| Question | Treatment minus control | What the difference contains |
|---|---|---|
| Did pretraining improve the supervised network? | SCARF fine-tuned − scratch | Pretraining, including its extra computation |
| Did the learned frozen representation help? | SCARF frozen − random frozen | Representation learning under a linear probe |
| Is it a useful prediction recipe here? | SCARF fine-tuned − raw logistic / tree | Architecture, optimization and representation differences |

The primary comparison is fixed before seeing results: **SCARF fine-tuned minus scratch**. It matches encoder architecture, head initialization, supervised batches, learning rate and epoch count. It does not match total compute: pretraining adds work. The practical controls are a raw-feature logistic classifier and a fixed histogram gradient-boosted tree model. They are useful local checks, not exhaustively tuned champions.

**Predict.** Could SCARF beat random frozen features yet lose to the raw inputs? Explain how information discarded by the encoder could produce that outcome.

## 2 · Draw the information boundary before training

> **In plain terms.** Labels are scarce; held-out examples still stay held out.

Split row IDs once into training (about 70%), validation (about 10%) and test (about 20%). A **validation set** supports model choices. The **test set** estimates the final recipe after those choices. An **inductive** experiment predicts future unseen rows; a **transductive** experiment permits using the unlabeled evaluation rows during training. This lab is inductive.

<!--figure:boundary-->

The feature scaler learns means and variances from the entire training pool. SCARF donors and pretraining batches come from that pool too. No validation or test feature contributes to a fitted transformation. A frozen probe fits a second scaler using only its labeled training representations. Its regularization strength is selected from C = 0.1, 1, 10 by validation accuracy, with the first candidate winning a tie. All other recipes have fixed hyperparameters and epoch counts.

**Subtle boundary.** All recipes receive the same train-pool feature scaling, including the scratch baseline. Therefore the primary difference isolates SCARF pretraining on top of shared preprocessing. It does not compare access to all unlabeled information versus access to none.

The benchmark's outer splits are stratified: class proportions guide assignment, using existing benchmark labels. Within the training split, label acquisition is blind to the label values. We assume the list of possible classes is known. This is an offline benchmark simulation, not a claim that a deployed annotation system knows hidden labels.

## 3 · Make the label budgets nested and count them honestly

A **label fraction** f is the fraction of training-pool rows whose target is available to the supervised learner. Our grid is 10%, 20%, 40%, 70%, 100%. The training feature pool stays fixed as f changes. At 100%, self-supervision still adds an objective, even though it adds no extra training rows beyond those with labels.

A **nested subset** preserves every row from the smaller budget. Shuffle training IDs once without reading their labels; take successively longer prefixes. Independently sampling each budget would mix the effect of more labels with replacing the labeled examples. Nesting reduces this avoidable variation; it does not make adjacent budgets independent.

<!--figure:nesting-->

**Worked example.** There are 700 training rows, 100 validation rows and 200 test rows. At f = 0.1, the learner uses 70 training labels. Model development has access to 70 + 100 = 170 labels. That is 21.25% of the 800 development rows, not 10%. The 200 test labels are evaluation annotations and must be accounted for separately if collecting this dataset costs money.

The training count is floor(f × training rows). Flooring means actual fractions can differ slightly from nominal ones. A very small blind sample can omit a class. The lab does not search seeds to repair an inconvenient sample; a one-class logistic training set raises an explicit error. Missing classes should be reported as part of the regime, or addressed by an annotation policy defined before evaluation.

**Prediction before moving the control:** if validation labels remain fixed, does halving f halve the total development-label cost? Keep the original 10% case as your reference.

<div id="budget-trace" class="contrastive-widget"></div>

[Oliver et al., §4.6 and §5](https://arxiv.org/html/1804.09170v2) examine validation-set realism and matched baselines in semi-supervised evaluation. Their image experiments motivate this audit; they do not establish a numerical tabular crossover. Here all arms have the same available validation budget, although fixed recipes do not use it to select parameters. We report available labels rather than pretending every method consumed them in the same way.

## 4 · Subtract within a seed before averaging

> **In plain terms.** Compare two runners on the same course before averaging their finishing times.

A **paired repetition** uses the same outer split, labeled subset and relevant random initialization in both arms. Our seed controls a split and model initialization together. Three repetitions therefore mix split and training variation; they do not separate those sources of uncertainty.

**Worked example.** At one budget, treatment accuracy is [0.80, 0.72, 0.90] and control accuracy is [0.75, 0.74, 0.86], ordered by the same seed IDs. Pairwise differences are [+0.05, −0.02, +0.04]. Their mean is +0.0233, or +2.33 **percentage points**. This is not a 2.33% relative improvement.

Accuracy A is the proportion of test rows classified correctly. For repetition s, define g(s,f) = A(treatment,s,f) − A(control,s,f). The mean gain is the sum of the S paired gains divided by S. Join results by seed ID; their storage order can differ.

**Uncertainty.** The sample standard deviation describes the spread of repetition results. The displayed paired t interval is mean gain ± t(0.975,S−1) × SD(g)/sqrt(S). With three gains in the worked example, SD is about 0.0379 and the interval is about [−7.07, +11.74] percentage points. A positive average can coexist with large uncertainty.

> **Scope check.** Our t intervals are descriptive sensitivity summaries over three repetitions of overlapping finite datasets. They do not provide independent-dataset evidence, account for all test-sampling uncertainty, or give simultaneous 95% coverage over the five budgets. More seeds cannot replace more independent datasets.

## 5 · A crossover is a result, not a required ending

A **crossover bracket** here means adjacent tested fractions with strictly opposite mean-gain signs. This operational definition reports where the sampled recipe ordering reverses. It does not estimate an exact threshold between fractions.

**Worked example.** Gains [+3, +1, −2, −1, −3] percentage points at fractions [0.1, 0.2, 0.4, 0.7, 1.0] give a bracket [0.2, 0.4]. The reverse transition is also a sign reversal. A curve may have several; return all of them. Exact zeros are observed ties and are reported separately. All-positive or all-negative means contain no strict adjacent crossing. That does not prove the underlying population curves never cross.

Do not smooth a curve until it tells the preferred story. Do not use a test-derived crossing to choose your deployment method and then claim that same test as independent confirmation. Freeze the follow-up choice and evaluate it on fresh data or within an outer evaluation protocol.

<div id="prediction"></div>

## 6 · Run the fixed experiment and read what happened

**Held fixed:** train feature pool, split IDs, nested label order, architecture within each matched comparison, 40 SCARF pretraining epochs, 60 supervised epochs, Adam learning rate 0.001, batch size 64, corruption rate 0.6 and temperature 1. The compact encoder is d → 64 → 64 with ReLU; the pretraining projector is 64 → 64 → 32. At fine-tuning it is replaced by a 64 → number-of-classes linear head.

**Varied:** the label fraction and six prediction recipes. **Measured:** test accuracy, paired gains, training label counts, validation availability and strict mean-sign reversal brackets. Each dataset/seed pretrains once; each budget receives a fresh copy of that checkpoint. Fine-tuning at 40% never starts from the model trained at 20%.

The three Tier-A offline datasets are sklearn's Wine, Breast Cancer Wisconsin Diagnostic and handwritten Digits. Digits is flattened image data, included to test a contrasting geometry; it is not evidence about ordinary business tables. Five budgets × six arms × three seeds × three datasets produce 270 selected test evaluations. Three probe arms each try three C values using validation only. Fixed neural and tree recipes have no validation search.

<!--figure:curves-->

<!--results-->

<!--figure:gains-->

**Observed pattern.** On Digits, frozen SCARF beats random features at every measured budget, yet fine-tuned SCARF has a negative mean gain over scratch from 20% onward. On Wine, the primary comparison ties at 40%, 70% and 100%. On Breast Cancer, the 40–70% bracket is built from mean differences smaller than one percentage point. These patterns make the choice of control and test-set resolution central to the conclusion.

A zero-width interval can occur when all three paired gains happen to be identical. It reports zero observed repetition spread, not certainty about new samples.

**Read in order.** First inspect the primary matched neural comparison. Next ask whether the frozen representation comparison agrees. Finally inspect raw logistic and trees before recommending a recipe. A disagreement between probe and fine-tuning results is a diagnosis target, not an arithmetic contradiction: supervised updates can change what information the encoder preserves.

<!--figure:ranks-->

The rank audit averages seeds within dataset, ranks the six methods within each dataset, and then weights the three datasets equally. Lower rank is better. The Friedman test and Nemenyi critical difference are exploratory summaries at each fraction. Five related rank tests and only three datasets do not justify a universal method ordering. Neither fractions nor seeds count as extra datasets.

> **Scope check.** These are author measurements of a fixed compact protocol. They are INCOMPARABLE to the SCARF benchmark's 69 datasets, 30 trials and training/selection recipe. The paper evaluates 25% and 100% labeled conditions; our denser grid is a teaching experiment, not a reconstructed paper figure. A longer run improves convergence evidence without fixing the dataset and protocol mismatch.

## 7 · Diagnose before adding another experiment

| Observation | Plausible explanation | Follow-up that separates explanations |
|---|---|---|
| Frozen SCARF improves, fine-tuning does not | Supervised optimization erases or cannot exploit its advantage | Log paired validation trajectories; predefine a learning-rate grid |
| Both neural arms lose to raw logistic | The compact encoding or optimizer harms useful coordinates | Compare representation information and longer fixed training |
| Benefit appears only at low f | A learned representation reduces the burden on scarce labels | Repeat with a separately sampled evaluation dataset |
| More pretraining data fails to help | Extra rows carry irrelevant structure or distribution mismatch | Hold labels fixed and vary only eligible unlabeled rows |
| A single seed determines a crossing | Split or optimization sensitivity | Report per-seed gains and run predefined additional seeds |

These are hypotheses, not findings of causal mechanisms. In particular this lab does not vary unlabeled pool size, distribution shift, corruption rate or total compute. Changing them all together would make an explanation harder. Return to your VIME notes: propose a VIME arm with the same boundary and label budget, and state whether it tests an objective or an entire recipe.

## 8 · Lab and exit ticket

Implement three live functions: nested label selection, seed-keyed paired gains, and all strict adjacent crossover brackets. Each has an independent arithmetic CHECK. Read the provided model and training code, then run the full experiment. Your functions must feed that execution; the notebook checks their bindings.

**Before running**, predict the direction of the primary gain at both ends of the grid for each dataset. **After running**, submit your actual curve, counts, paired gains and crossover report. In 150–250 words explain one failed prediction, distinguish a sign reversal from an established threshold, and design one follow-up with exactly one changed factor. Include a practical baseline and the validation-label cost.

Structural EXIT checks verify the row boundary, nested subsets and saved prediction scores. Your written explanation still needs tutor review. Ask the tutor about any unclear step; bring the result and your reasoning, not just “the cell passed.”

**Spaced follow-up:** tomorrow, reconstruct the six comparisons without opening this page. In a week, audit an SSL result from a paper for validation-label cost and baseline matching.

**Mission connection.** Before attributing a future gain to relational structure, match supervision and feature access. This experiment raises the evidential bar; it provides no direct evidence that relational models outperform single-table models. Lesson 74 moves to cross-table transfer, where source-table identity and schema access introduce additional boundaries.

**Primary reading:** [SCARF, Figure 1, §4 and Appendix B](https://arxiv.org/html/2106.15147v2). Read alongside [the tabular SSL survey](https://arxiv.org/html/2402.01204v3) and [Oliver et al.'s evaluation recommendations](https://arxiv.org/html/1804.09170v2). Reproduce their questions about evidence before borrowing their conclusions.

## Paper reproduction track

The [paper reproduction section](../labs/html/0073-when-ssl-helps.html#paper-reproduction) in the companion notebook includes the full runnable paper/release implementation after EXIT, separately from its compact teaching experiment. Read the [reproduction guide](../labs/reproductions/README.md) for the named target, full-data commands, source pins and remaining gaps. The full model and training code are visible in the notebook. Smoke execution and source equivalence are checks; published-result reproduction requires the aligned experiment and its measured comparison.
