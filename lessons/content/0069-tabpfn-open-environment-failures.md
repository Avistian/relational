## Start with the prediction contract

> **In plain terms.** Before you can call a model "broken," you have to say what job it was doing. This lesson turns a vague "is it robust?" question into a precise experiment, and then says exactly what that experiment's score does and does not measure.

**What this lesson trains.** Your skill here is to turn an open-environment claim into a controlled experiment and then explain exactly what its score measures. By the EXIT you will have implemented six live operations and run the complete historical TabPFN v2 weights. You will separate two failures that look identical from the outside: being unable to *name* a new class, and being unable to *detect* an unfamiliar row. You will also test what happens when a feature is removed. You will separate a change in the inputs from a change in the label rule. You will interpret several scoring objectives computed on one fixed set of predictions.

**Retrieve first, without notes.** Answer these from memory before reading on. What does a PFN learn during pretraining? Which labels are allowed to enter its context? Why does a temporal holdout answer a different question from a random split? What must be held fixed before two losses can be compared? These connect back to lessons [062](0062-tabpfn-v1.html), [064](0064-tabpfn-v2.html), [066](0066-tabicl-column-row-attention.html) and [068](0068-pfns-under-temporal-shift.html); the essential definitions are restated below.

### The four ways a contract can break

**A prediction contract** is the set of promises around one prediction. It says what one row represents, which columns are available when an action is taken, which labels may occur, and which errors matter. Hold that definition, because a model can return a perfectly shaped probability vector while quietly violating it.

Four different things can break, and they are not the same failure.

- A new disease category breaks **label support**: the answer you need is not among the classes the model can output.
- A disconnected sensor breaks **feature availability**: a column the model expects is gone.
- A changed relationship between measurements and disease breaks the **conditional label rule**: the same features now imply different labels.
- A new cost of a missed case changes the **evaluation objective**: the same predictions are now scored differently.

These are four distinct questions, even when all are called “robustness.” The whole lesson is about keeping them apart.

**The reading.** The primary source is [Cheng et al., *Realistic Evaluation of TabPFN v2 in Open Environments*, v1, Sections 4–5 and Appendices C–H](https://arxiv.org/html/2505.16226v1). It provides the four-axis organizing framework used above. Treat its conclusions as claims to examine against tables, definitions and implementation, not as settled results. This is an evaluation lesson; it introduces no new model.

## Trace the whole evaluation before interpreting a failure

> **In plain terms.** You cannot diagnose where a prediction went wrong until you can point to every stage it passed through. This section walks one row from raw data to final score.

**Two words for the two halves of the table.** A **context** is the labeled training table supplied to a pretrained PFN. **Query rows** are the rows whose features are available but whose labels are hidden during prediction. The pretrained weights stay fixed: constructing a new context is *not* gradient training.

**What our implementation actually computes.** Our complete v2 implementation carries numeric values and missing-value flags through two-feature groups, then feature attention, then context-only row attention, repeated in twelve blocks. Its output head has ten coordinates, but each prediction is restricted and normalized to the classes present in the context. So the original classifier cannot acquire the name of an unseen class merely because its head has unused output coordinates. Hold onto that fact; it is the heart of the first failure mode below.

**The exact model we measure.** The measured variant is the full historical Nature classifier. It has width 192, six attention heads, hidden width 768, and twelve layers, for 7,244,554 parameters in 81 checkpoint tensors. The checkpoint SHA starts `f65a35685aeef42e3`. We run one raw numeric view, model seed 0 and temperature 0.9.

> **Scope check.** The historical default wrapper averages four transformed views, not one. Matching the complete network does **not** make this single-view protocol identical to that wrapper. Lesson [064](0064-tabpfn-v2.html) derives the architecture and checks the original 2.0.9 implementation.

<!--figure:pipeline-->

**Worked trace of one query.** Follow a single query row through the figure, one stage at a time. Keep its original row ID. Construct a context that does not contain its label. Apply the declared intervention. Predict on exactly those query features. Map the probability columns back to the context labels. Then score every required query. Change a denominator at that last step and you have changed the experiment. The saved JSON preserves the splits, original IDs, targets, class maps, probabilities and live code identities, so the path can be reconstructed.

> **Scope check.** Context-only attention does not by itself imply unconditional independence of query predictions. Historical numeric preprocessing counts active variation across all supplied rows. We keep each declared query set in one call. Changing query batching can therefore change preprocessing and is a protocol change; it is not a free memory optimization.

## What the released source actually certifies

> **In plain terms.** The public code you can download is not necessarily the code that produced the paper's tables. This section says what it can and cannot vouch for.

**The pinned snapshot.** The [official repository snapshot](https://github.com/LAMDA-NeSy/Evaluation-on-Tabular-Model-in-Open-Environments/tree/744c010457f68284faa7ae6ded793a8b3f3e03a4) is pinned by commit and by per-file hashes. Its current README describes v2.5, and its requirements select `tabpfn==6.4.1`; that was released after the May 2025 v1 paper. So it cannot certify which code or checkpoint produced the original tables. We retain the original files under `labs/sources/l069-v2` and execute isolated source fragments as independent checks.

**A concrete defect in the released scoring script.** The current `newclass.py` does several things worth naming one at a time. It converts every feature to an integer. It uses seed 42. It applies a fixed confidence interval. It passes the resulting **binary** decision into ROC-AUC and average precision. It overwrites the result for each held-out class, so its final saved value is only the last class, even though the paper describes averaging classes over three seeds. The integer conversion also destroys real distinctions: on Wine, volatile acidity .70 and .88 both become 0, while density .9978 becomes 0. Those are actual first-row values in the released red-wine CSV.

> **Scope check.** Our run preserves the released CSV floating-point values, all class results and three actual split seeds. These corrections deliberately break whole-script parity: we are fixing the script, not reproducing its output.

**What we can and cannot certify.** The source audit verifies the parts we retain: all sixteen class partitions across CMC/red/white at seed 42, the binary scoring fragment, and all fifteen Iris feature-removal subsets. It separately records what we change. The model check compares complete logits and gradients against the isolated original TabPFN 2.0.9 source; the lesson's faster CPU attention kernel changes scheduling, not the checkpoint or the mathematical scale. Keep three claims apart: **source primitive parity, protocol alignment and result reproduction are three different claims.**

## Emerging classes: naming and detecting are separate tasks

> **In plain terms.** If a genuinely new category shows up, the model has no output slot for it — but a separate detector might still notice the row "looks unfamiliar." Naming and noticing are different jobs.

**Worked example.** Suppose the context labels are A and B. A query has probabilities `[0.2, 0.8]`, but its true label is C. The classifier can choose A or B; it can never correctly name C. This is a **support failure**: the correct answer is outside the supported label set, independent of whether its internal representation is useful. A separate detector might still flag the row as unfamiliar. That detector predicts “known or novel,” not the new semantic class name.

**Building the leave-one-class-out task.** Build the paper-shaped **leave-one-class-out** task in named steps. Choose one held-out label. Put every row of that label into the novel query set. Sample an equal number of rows from the remaining labels as known queries. Make all other known rows the context. Then repeat the whole procedure for every label. Context and query IDs stay disjoint, and their union is the full dataset. A held-out class is an intervention within a dataset, not an independent benchmark dataset.

> **Scope check.** This balanced design fixes novel prevalence at one half. It is useful for comparing detectors under a controlled query mixture, but it is not the dataset's natural class prevalence. The companion natural-prevalence task instead makes a stratified 80/20 split, removes original class 0 from training only, and scores **every** test row without rebalancing. Excluded training rows remain listed in the artifact. The two designs change both context size and test composition, so a score difference between them cannot be attributed solely to prevalence.

## Derive what a confidence detector is measuring

> **In plain terms.** A simple novelty detector says "the less confident the model is in any known class, the more likely this row is new." We derive exactly what that heuristic scores, and where the released version throws information away.

**Set up the notation, one symbol at a time.** Let `p[i,c]` be the probability of context class `c` for query `i`. Maximum confidence is `m[i] = max_c p[i,c]`. Our continuous novelty score is `s[i] = 1 − m[i]`: larger means less confident in any supported class. This is a diagnostic heuristic, not a calibrated probability of novelty. A familiar but ambiguous row can also have low confidence.

**Where the released detector loses information.** The source instead scores `b[i] = 1{0.4 ≤ m[i] ≤ 0.6}`, a yes/no flag rather than a ranking. The lower bound is the problem. For three known classes, `[.34,.33,.33]` has lower confidence than `[.50,.25,.25]`, yet the interval flags only the second row. So an interval is not equivalent to “lower confidence means more novel.” Two edge cases matter too: with two known classes maximum confidence cannot fall below 0.5, and with one class a softmax detector is degenerate, which our API rejects.

<!--figure:novelty-->

**Worked example — predict before you compute.** In the six-row example, will using the interval preserve the continuous score's ordering? There are three novel and three known rows. The continuous scores correctly order eight of the nine novel–known pairs, so ROC-AUC is `8/9 = .8889`. The interval instead produces `[1,0,0,1,0,0]`; its novel hit rate is `1/3` and its known rejection specificity is `2/3`. For any binary score, trapezoids through `(0,0)`, `(FPR,TPR)`, `(1,1)` give `AUC = (TPR + 1 − FPR)/2`. Here that is `.5`. Both numbers come from the same underlying probabilities, so the collapse from `.8889` to `.5` is caused purely by discarding the ranking.

**Average precision, and why its baseline moves.** **Average precision (AP)** sums recall increments weighted by precision at each distinct score threshold. It differs from the trapezoidal area under a precision–recall curve; the released API is [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), even though the paper calls the column AUPR. The example's continuous AP is `.9167`, versus `.5` for the binary interval. AP's baseline depends on novel prevalence. For a constant score it equals the positive fraction, so a balanced task's `.5` baseline cannot be carried into a rare-novelty task unchanged.

## Keep the unsupported rows in the loss

> **In plain terms.** When the true answer is a class the model cannot output, it is tempting to drop that row from the score. Doing so hides the failure. We keep the row and give it an explicit, finite penalty.

**Why a naive loss blows up.** For a required target outside the context class map, the correct target probability is zero. Ordinary log loss is then infinite: `−log(0)`. To make a finite diagnostic, we explicitly clip each target probability below at `epsilon = 10⁻¹²`. An unsupported row then contributes `−log(epsilon) = 27.6310` nats, where a **nat** is a log-loss unit measured with the natural logarithm. That number reflects the declared clipping convention; it is not a learned estimate of open-set risk.

**Worked example.** Suppose two known rows have target probabilities `.8` and `.6`. Their mean loss is `(.2231 + .5108)/2 = .3670`. Now add one unsupported row. The all-row diagnostic becomes `(.2231 + .5108 + 27.6310)/3 = 9.4550`. Reporting only `.3670` would hide a third of the required task. Because changing epsilon changes the penalty, report epsilon and the unsupported frequency beside the score.

**What the lab reports, and why half is the ceiling.** The lab reports all-row accuracy, known-only accuracy, known-only loss, the unsupported count and fraction, and the clipped all-row loss. In the balanced leave-one-out design, one half of queries are unsupported by construction, so an ordinary classifier's all-row accuracy is at most one half. This ceiling is built into the task; it is not evidence that the classifier randomly forgot half its classes. A real abstention option would need its own coverage and accepted-risk evaluation, with thresholds fixed from appropriate validation evidence. We do not fit an abstention policy from test labels.

## Feature removal: compare the same row with a declared fallback

> **In plain terms.** If a sensor disappears, the model still needs some number in that column. We fill it with the context average, which keeps the input shape valid while genuinely removing information.

**Schema and the fallback rule.** A **schema** identifies columns by meaning and order. Deleting a required column makes the old input invalid. A mean-imputation fallback restores the original width, but gives the model strictly less information: for a removed numeric column `j`, replace every query value with the mean of context column `j`. Context and weights stay fixed. This corresponds to the numeric path of the released feature-shift experiment. Its categorical path uses the training mode; our measured panel has numeric input only.

<!--figure:features-->

**Worked example.** Context sensor values `[2,4,6]` give mean 4. Query readings 1 and 7 become 4 and 4, while a second column `[10,20]` remains unchanged. The sensor no longer distinguishes these two rows. Notice a testing trap: using their query mean would accidentally give 4 in this example too, so change the second query from 7 to 9 and the error becomes visible. An invariant test must vary exactly the values that would distinguish the competing implementations.

**How removal levels are chosen.** The local run uses six predeclared nominal levels: 0%, 20%, 40%, 60%, 80%, 100%. For `F` features it removes `floor(level × F)` columns from one seeded random permutation. The levels are nested and use no labels. On Iris, 20% means `floor(.2 × 4) = 0`; the actual removed fraction is saved, so the duplicate baseline cannot masquerade as a one-feature intervention. Three seeds vary the real train/test split **and** the mask ordering jointly. They do not separately estimate those two variance components.

> **Scope check.** The released `random/all` path enumerates every subset of each size; our three nested permutations are a sampled-mask diagnostic. All fifteen nonempty Iris subsets are independently checked at the transformation level and separately measured with both complete models at seed 42. The [saved control](../labs/_feature_control_l069_v2_results.json) records every subset and prediction. At 50% removal the all-subset mean accuracy is .6500 for v2 and .6611 for XGBoost. This one-dataset control does not stand in for the full twelve-task feature suite; it makes mask selection inspectable. The released `least/most` alternatives compute feature–label correlation using the full table, including test labels, and reuse mutable frames across levels. We do not use those paths. Selecting a “damaging feature” after seeing test labels changes the scientific claim.

## Absolute performance and degradation answer different questions

> **In plain terms.** "How much did it drop?" and "how good is it now?" are different questions. A model can drop a little and still be worse than one that dropped a lot.

**Two ways to measure change.** For a higher-is-better metric `a`, an absolute change is `a_shift − a_clean`. Relative change is `(a_shift − a_clean)/a_clean`. A fall from `.852` to `.809` is `−.043` on the unit accuracy scale (−4.3 percentage points), or `−5.05%` relative. The paper's Appendix F equation defines the relative form; Table 2 displays numbers resembling the absolute one. Our artifacts store both under different names.

> **Scope check.** Neither number is a confidence interval: the paper's “± gap” annotations must not be read as uncertainty bars.

**Why you must show both scores.** A model beginning at `.50` can decline less than one beginning at `.90` while remaining worse after corruption. So always inspect both the clean and the shifted score. At 100% removal every query has identical features. With our fixed deterministic recipe, predictions should then become identical within a model/context call; any remaining accuracy is explained by the predicted class and the test prevalence, not by retained row-specific information.

**Incremental features — a negative control.** The implemented policy aligns columns by their training names and ignores a newly available sensor. It accepts reordered or additional columns, and rejects missing required names until a fallback is specified. Its aligned input is exactly the baseline input, so invariance here is evidence that the new information was discarded. It is not evidence that v2 learned to use a sensor without context examples. The input equality is measured for each real feature split and retained as a negative control. On the separate Iris control, both predictors are actually called on aligned augmented inputs; the maximum probability change is exactly zero.

## Distribution shift: changing the data-generating law

> **In plain terms.** The data can change in two very different ways: the *features* can move while the labeling rule stays put, or the *rule itself* can change while the features look the same. Telling these apart is the whole game.

**Name the two shifts.** Write the joint distribution as `p(x,y) = p(x)p(y|x)`. **Covariate shift** changes `p(x)` while preserving the conditional label rule `p(y|x)`. **Concept shift** changes that rule. A natural temporal dataset can combine both; changed marginal features alone do not identify which mechanism explains a performance gap.

**The controlled two-coordinate family.** The lab constructs a controlled two-coordinate family. Source rows have independent standard-normal coordinates and label `y = 1{x₀ > 0}`. A covariate intervention moves query coordinate zero by +1.5 and recomputes labels using the same threshold rule. A concept intervention leaves every query feature exactly unchanged and reverses the label rule to `y = 1{x₀ ≤ 0}`. We retain the original query row IDs, use the same frozen context, and run both complete models. Generated tasks are labeled synthetic throughout; their seeds are not additional real datasets.

<!--figure:shift-->

**Worked example — the exact concept-reversal check.** For concept reversal, the frozen predictor must return the same probabilities as on the IID query set: it receives identical context, labels and query features. Only the evaluator's labels changed. For these fixed binary predictions, the new accuracy is `1 − old_accuracy` and the new AUC is `1 − old_AUC`. This is an exact check on the experiment. A collapse here does not reveal a special defect of TabPFN; any unchanged accurate binary predictor must fail when its correct labels are reversed.

**A limitation you can prove.** The paper uses feature-space distances and learned representations to help discuss shift. Here is a limitation you can derive without trusting a model ranking: if source and target have identical `X`, any fixed representation `f(X)` is also identical. A Fréchet distance computed solely from `f(X)` is zero even when every label reverses. Such a distance cannot by itself identify pure conditional shift. This does not invalidate the separate DISDE risk decomposition discussed in Appendix G; it prevents a feature-distance proxy from becoming proof of a changed label law.

**Why an "easier" shift can look better.** Moving a standard-normal coordinate by 1.5 raises positive prevalence from .5 to approximately .9332 under the unchanged threshold rule. A higher covariate-shift accuracy can therefore reflect an easier class mixture. A changed label marginal alone is not proof of pure label shift, which additionally preserves the class-conditional feature distribution. The generated evidence reports realized class counts beside each score.

> **Scope check.** The original nine WhyShift/TableShift tasks are **not run here**. They include geographic and temporal ACS settings plus College Scorecard, BRFSS Diabetes and hospital readmission. Their row universes, domain definitions and large contexts differ from our generated control. The accompanying reproduction document records the specific missing protocol and source discrepancies. Lesson [068](0068-pfns-under-temporal-shift.html) supplies the sequence's distinct real temporal experiment; its evidence must not be counted again as a new 069 run.

## A different objective can expose a different failure

> **In plain terms.** The same predictions can look good or bad depending on which score you use. Fix the score before you look, so you are not tempted to pick the metric that flatters the result.

**One set of predictions, four metrics.** The paper's fourth axis rescored predictions with several classification metrics. It does not, by itself, demonstrate adaptation to a new utility function. Here we compute all four metrics from the same unmodified real query predictions, without choosing a winning metric after the run.

**Define each metric on first use.** **Accuracy** is the fraction correct. **Recall for class c** is the fraction of true-c rows predicted c. **Balanced accuracy** averages these recalls equally across classes. **Precision for class c** is the fraction of predicted-c rows that truly are c. The class F1 is the harmonic mean `2 × precision × recall/(precision + recall)`, or equivalently `2TP/(2TP+FP+FN)`. **Macro F1** averages class F1 values equally; it is not minority-class F1. Our multiclass **ROC-AUC** averages one-versus-rest probability rankings equally over classes; binary AUC uses class-1 probability.

> **Scope check.** Undefined values must be recorded with an explicit missing status; do not substitute zero. The declared sklearn metric adapters ([F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)) average balanced accuracy over true classes present in that test, and macro F1 over the union of true and predicted labels. An absent test class therefore needs particular care when comparing class-averaged metrics across splits; the full context class set is required for our macro OVR AUC.

<!--figure:objectives-->

**Worked example.** Consider ninety class-0 and ten class-1 rows, with every row predicted class 0. Accuracy is `.90`, class-1 recall and F1 are zero, balanced accuracy is `.50`, and macro F1 is `(180/190 + 0)/2 = .4737`. A high accuracy does not prove useful minority recall. Conversely, a lower balanced accuracy than accuracy is not a statistical test of intrinsic model bias; it depends on the class proportions and the confusion matrix.

**A worked contradiction in the paper's own numbers.** The paper's current source uses macro F1, yet the published BRFSS accuracy .875 and F1 .044 cannot be binary macro F1 on the same predictions: binary macro F1 is at least accuracy/(1+accuracy), here .4667. To see that lower bound, write the correct positive and negative fractions as u and v, with u+v=a and error fraction e=1−a. Then macro F1 is u/(2u+e)+v/(2v+e). This concave expression is minimized when one of u,v is zero, giving a/(1+a). We name our averaging explicitly and do not compare incompatible targets.

**Turning probabilities into decisions.** For a calibrated binary probability p, predicting positive has expected cost C_FP × (1−p); predicting negative costs C_FN × p. Comparing them gives the threshold p > C_FP/(C_FP+C_FN). If a false negative costs nine times a false positive, the threshold becomes .1. This derivation assumes those costs and deployment calibration; a changed cost alone does not require a new representation. A ranking can also stay perfect while probability quality differs: probabilities [.1,.9] and [.4,.6] on one negative and one positive row both give AUC 1, but mean log losses .1054 and .5108.

> **Scope check.** Changing a decision threshold can change accuracy/F1 while leaving ranking AUC unchanged. Choosing that threshold against test labels would leak the objective into evaluation. A real cost change may require validation-based calibration or decision selection; post-hoc metric reporting establishes neither. For relational systems, “new objective” might also mean a new prediction horizon, which changes the label itself and needs a new availability audit.

## Read the fresh evidence at its actual unit of analysis

> **In plain terms.** Predict what the results will show before you see them, then read each number at the level it was actually measured — per class, per dataset, per seed — not as a grand claim about all tasks.

**Predict before the reveal.** Before the author panel appears, commit to answers. Will continuous confidence and the interval have the same novelty ranking? Must all feature-removal curves worsen monotonically? Must v2 outperform the fixed tree in every metric? Which generated shift should defeat both unchanged models by construction? Write one reason for each answer.

<!--results-table-->

<!--figure:results-->

<!--figure:feature_results-->

<!--analysis-text-->

**What the panel covers.** The author panel uses all 1,473 CMC rows, all 1,599 red-wine rows, all 4,898 white-wine rows and all 150 Iris rows. CMC is nine numeric-coded features and three classes; the released wine labels have six and seven classes respectively, despite Appendix E's four-level description. CMC/red/white are the novelty panel; Iris/CMC/red are the feature panel. Iris belongs to the paper's feature suite; CMC and red wine extend feature tests to source-prepared novelty datasets. These related Wine tasks are two reported dataset units, not evidence of broad domain diversity.

> **Scope check.** Red wine seed 789 contains no class-0 test row in the feature experiment. Its multiclass AUC is undefined, recorded as `None`; AUC means use only the two available seeds, named in the artifact, while other clean metrics retain three. No missing value is converted to zero.

**How to read the error bars.** Each reported real-task mean first averages held-out classes equally where relevant, then averages split seeds 42, 2023 and 789. Error bars show sample standard deviation across those three split/mask repetitions, conditional on the dataset and fixed checkpoint. They are not confidence intervals over future datasets. Paired comparisons keep the same intervention and query rows; a separate paired dataset bootstrap resamples the three dataset units. With only three related tasks, its interval is descriptive and cannot establish general model-family superiority.

**The comparator.** The comparator is fixed XGBoost 3.3.0, 100 histogram trees, depth 6, learning rate .3, full row/column sampling and one CPU thread. There is no validation search in either arm. This is not Appendix D's five-fold tree search or 100-trial neural optimization. Two-arm average ranks are reported within each dataset. Paired dataset differences are the relevant comparison here; with only three dataset units the uncertainty remains descriptive. We do not add a comparator merely to fit a particular omnibus test.

## Reconstruct a claim before accepting it

> **In plain terms.** A paper's sentences must agree with its own tables. When they do not, the table wins — and you should say so precisely, without overclaiming that the whole paper is wrong.

**When narrative and table disagree.** A paper's table is its primary evidence; its narrative does not outweigh it. Section 5.1 calls v2 consistently better at novelty detection, but Table 1's red-wine MLP `.700` ROC-AUC and `.658` AUPR exceed v2 `.533` and `.521`. Eye Movement `.511` and CMC `.522` are near chance on the reported scale. They do not establish broad reliable new-class detection. Our corrected local scoring experiment is also not a reproduction of those numbers.

**A second internal tension.** The broader “trees remain optimal” wording extends beyond a finite benchmark. Even inside the paper, Table 3 gives RandomForest average rank `3.93` and v2 `3.53`, where smaller is better; it cannot support the accompanying statement that RandomForest consistently beats v2. Neither these inconsistencies nor our local counterexamples prove that the paper's entire empirical direction is false. They require narrower, table-specific claims and reproducible operators.

**The checklist for any claim.** For each claim, ask for: exact source location; data/split identity; estimator and wrapper identity; metric definition; aggregation unit; uncertainty; and a saved result. If any is absent, record the gap. A local code match cannot repair an unavailable historical protocol by assertion.

## Connect the failure to the earlier mechanisms

> **In plain terms.** Each earlier lesson built a mechanism; this lesson is the discipline that tells you when that mechanism actually helps.

Lesson 062 explains a prior fitted network as amortized inference over a training-task distribution. Its predictive behavior inherits that task distribution; a new deployment contract need not match it. Lesson 064 shows how complete v2 weights turn context into probabilities, including preprocessing dependence. A ten-output head does not imply ten named classes of permanent semantic meaning.

Lesson 066's scalable TabICL mechanisms address how to represent and process large tables. More feasible context does not automatically supply a missing class, interpret a new column or fix a reversed label law. Lesson 068 explicitly conditions predictions on time and changes its pretraining tasks to model drift. That is a proposed adaptation mechanism; 069 is the measurement discipline needed to establish when such a mechanism helps. Do not conclude that an architecture is robust merely because it accepts a larger or time-stamped table.

**Bring it back to the relational mission.** Name an operational example: a new merchant category, a dropped transaction attribute, a changed link between activity and fraud, or a higher cost of missing fraud. Relational neighbors might provide the missing signal, but only if their features and labels were available at prediction time. Graph structure can add useful information and additional contracts to audit; it does not excuse a weaker baseline protocol.

## EXIT: explain a failure and a non-failure

> **In plain terms.** Finish by proving you can explain one real failure and one thing that only looks like a failure — and by stating plainly which paper results you did not run.

Run the six live TODOs and their diagnostic CHECKs. The notebook displays the visible complete model and evaluation code, independently checks the source-scoring fragment and complete original network, and executes a fresh CMC/Iris panel plus the controlled shift family. Your functions determine the actual splits, masks, scores, label laws and summaries. The broader gate reruns the full author panel using your current definitions.

Submit the fresh `exit-v2.json` and a written explanation that identifies one failure, one negative control, one metric disagreement, the all-row denominator, and the next discriminating experiment. Explain why a continuous detector may rank novelty usefully while an ordinary classifier still cannot name the held-out class. State precisely which paper results remain unrun.

> **Scope check.** The tutor will assess the reasoning; execution alone does not establish mastery.

Close the lesson and teach the prediction contract back from memory. If a calculation or implementation step is unclear, ask the teacher about that exact row, probability or control. The next lesson's model comparison should preserve this discipline: more compute is useful only when the task and evidence are still identifiable.
