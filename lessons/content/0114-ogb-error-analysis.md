## The question behind one accuracy number

A model can win overall and lose on an identifiable part of the graph. Your task is to turn that statement into a reproducible **error report**: name the population, count its nodes, compare the same questions, explain what the result does and does not establish, and propose the next experiment.

[Lesson 112](0112-ogb-gcn-reproduction.html) reproduced a full-batch Graph Convolutional Network (GCN) on ogbn-arxiv. [Lesson 113](0113-scaling-ogb.html) moved to a different, much larger graph and studied scalable training. Here we return to arxiv because it contains publication years as well as graph structure. The [official OGB documentation](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) defines the task: predict one of 40 subject categories from a paper's features and citation context.

**The win.** By the end, you can defend a sentence of the form: “On this precisely defined set of n nodes, the GCN differs from a feature-only baseline by Δ percentage points; here is the evidence and the next check.” This is the same habit you will need when assessing where relational learning beats feature engineering in Year 4.

**Read first:** [Hu et al., OGB §4.3 and Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3), then the [pinned MLP source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/mlp.py). Trace `train` before `forward`: which rows enter batch normalization?

[[WARMUP]]

## 1 · Freeze the questions before studying the mistakes

A **slice** is a subset of evaluation nodes satisfying a rule. “Test papers with one or two unique neighbors” is a slice. “The hardest papers” is not a reproducible rule until you define how hardness was measured and which data selected them.

**Population.** We use every official validation node (29,799) and test node (48,603), with all 169,343 feature rows and the released citation graph available to the GCN. Validation papers are from 2018; test papers are from 2019–2020. Features have 128 numeric coordinates. Labels are integers 0–39. This is a **transductive** benchmark: the graph and feature rows of evaluation nodes may be visible during training, while their labels are excluded from the training loss.

**Held fixed.** Both models predict the same released nodes with the same labels and feature vectors. Each is selected by its own highest validation accuracy, taking the first epoch in a tie. Each reported run has 500 training epochs. An **epoch** here is one complete pass through the training labels with one optimizer update.

**Varied.** We compare the published GCN and the published multi-layer perceptron (MLP), a network that transforms each node's feature vector without exchanging neighbor messages. Their source protocols differ in normalization population and initialization as well as adjacency. This is a comparison of two released baselines, not an intervention that removes edges from an otherwise identical training process.

**Frozen slice rules.** Degree: 0, 1–2, 3–5, 6–10, 11–20, 21–50, 51+. Homophily: undefined, [0,.25), [.25,.50), [.50,.75), [.75,1]. Class: all 40 categories. Year: each observed publication year. Within a family, every node belongs to exactly one slice. Across families, slices overlap.

> **Scope check.** A year-based label split does not guarantee strict historical forecasting. The released GCN can use later graph context. Recall [Lesson 109](0109-database-timestamp-contracts.html) and the distinction between event time and information availability. We preserve OGB's protocol rather than silently replacing its graph with time-filtered snapshots.

## 2 · Model architecture: what the simple baseline actually does

> **In plain terms.** The MLP sees the words summarized in one paper's vector. The GCN can combine that vector with citation-neighbor information. A strong comparison requires reproducing the MLP carefully too.

The MLP receives a matrix X with one row per paper and 128 columns. A **linear layer** computes XWᵀ+b: W mixes feature coordinates and b adds a learned offset. It produces 256 coordinates in each of the first two layers and 40 class scores in the last layer.

**Hidden blocks.** Each hidden linear output passes through batch normalization, ReLU, and dropout. Batch normalization standardizes each coordinate using the current training batch and keeps running statistics for evaluation. ReLU replaces negative values by zero. During training, dropout randomly removes half the hidden coordinates and rescales those retained; it is disabled at evaluation. The final log-softmax converts scores into log probabilities whose exponentials sum to one per row.

**Objective.** Negative log-likelihood penalizes the log probability of the correct class: a correct-class probability of 0.8 incurs −log(0.8)≈0.223, while 0.2 incurs ≈1.609. Average this loss over training nodes. Adam adjusts weights using gradients, at learning rate 0.01. No label from validation or test participates in that objective.

[[FIG:architecture]]

**The easy-to-miss distinction.** The released MLP calls `model(x[train_idx])` during training. Its batch normalization sees 90,941 training rows. The released GCN calls its model on the complete graph and masks the loss afterward; its normalization sees all 169,343 rows. Changing the MLP to process all nodes before masking the loss changes the released experiment, even if the labels remain correctly masked.

**Selection and inference.** After each epoch, switch to evaluation mode and score the full graph's feature rows. Save the first best validation state, including batch-normalization running statistics. Restore it before archiving predictions. Neither last-epoch accuracy nor the highest test accuracy is a replacement for the selected predictor.

**Predict:** would changing only held-out feature rows alter a single MLP training update?

<details><summary>Check your reasoning</summary><p>No: the released training forward pass includes only training rows. It can change validation predictions and therefore later checkpoint selection, but not that one optimization update. The source audit tests the one-epoch statement using held-out features and labels replaced by extreme values.</p></details>

## 3 · Count neighbors before you count errors

**Degree** is the number of distinct neighbors of a node. We use the binary undirected citation graph without self-loops for this diagnostic. A citation in both directions still creates one neighbor, not two. The self-loop used by GCN computation does not make a paper its own external neighbor.

**Homophily** measures label agreement among those neighbors. For node i with neighbor set N(i), define hᵢ = number of neighbors j with yⱼ=yᵢ, divided by |N(i)|. The bars denote set size. A value of 1 means all neighbors share the label; 0 means none do. An isolated node has degree zero, so the ratio is undefined. We keep it in an explicit “undefined” slice rather than replacing it with zero.

**Worked example.** Node A has label 0, B has label 0, and C has label 1. Edges connect A–B and B–C. For B, degree=2 and homophily=1/2. For A, degree=1 and homophily=1. For C, degree=1 and homophily=0. Repeating the record A→B does not change any answer. Adding a self-loop to C must not turn its diagnostic homophily into 1/2.

[[FIG:neighbors]]

**Task 1 · graph semantics.** Implement `neighborhood_properties`: deduplicate undirected neighbors, exclude self-loops, compute degree and homophily, and retain undefined ratios. Also compute the fraction of neighbors belonging to the training split. That fraction measures access to training nodes; it does not require the focal node's label and is not homophily.

**Information boundary.** True-label homophily requires the focal label and neighbor labels, including held-out labels here. True class also requires the answer we are predicting. They are legitimate retrospective diagnostics after evaluation, but cannot be supplied to a serving-time model selector. Degree and known training-neighbor membership are different: they can be computed without hidden labels when the graph itself is available.

<details><summary>Can we deploy “use MLP when true homophily is low”?</summary><p>No. A future unlabeled paper has no known true class, and many neighbors may also lack labels. The diagnostic suggests a hypothesis about harmful context. A deployable rule would need label-free signals and its own validation protocol. Substituting predicted labels changes the quantity and needs separate testing.</p></details>

## 4 · Compare mistakes on the same nodes

> **In plain terms.** Do not just place two accuracies next to each other. Count where their answers disagree, using the same denominator.

**Accuracy** equals correct predictions divided by evaluated nodes. For slice S with n nodes, accuracy(S)=Σᵢ∈S 1[predᵢ=yᵢ]/n. The indicator 1[statement] is 1 when the statement is true and 0 otherwise. An empty slice has no accuracy; report “no nodes,” not 0%.

**Worked example.** On ten nodes, both models get five right; GCN alone gets two right; MLP alone gets one right; both miss two. GCN accuracy is 7/10=70%, MLP is 6/10=60%, and their difference is +10 **percentage points** (pp). The difference also equals 100×(2−1)/10. This is not a 10% relative improvement, and the two models' mistakes are not independent.

**Task 2 · retain the pairing.** Implement `slice_metrics`. Inputs contain one prediction row per seed, indexed by the same global node IDs. Return n, both accuracies, their pp difference, and the four disagreement counts for each seed. Slice IDs need not be sorted. Do not accidentally index a subset as if it were the whole graph.

**Seed variation.** We report ten individual runs, their mean, and **sample standard deviation**: the square root of the sum of squared deviations divided by 9. This describes variation from repeated training on this fixed graph. It is not a confidence interval for future databases. Nodes connected by citations are dependent; pretending every node is an independent experiment overstates certainty. Indexing MLP seed k beside GCN seed k is a reproducible display convention, not shared initialization across architectures.

**Why slice sizes matter.** Overall accuracy is a weighted average of slice accuracies when the slices form a disjoint partition. The weight for a slice is its n divided by the population size. Averaging slice percentages equally instead answers a different question.

[[FIG:composition]]

**Predict:** a model is correct on 90/100 nodes in one slice and 1/10 in another. Is overall accuracy 50% or about 82.7%?

<details><summary>Check the denominator</summary><p>It is 91/110≈82.7%. The equal-slice mean is (90%+10%)/2=50%; it gives ten difficult nodes the same weight as one hundred easier ones. Both summaries can be useful if named clearly, but only the node-weighted value reconstructs OGB accuracy. The analysis checks this identity for every family, model and seed.</p></details>

## 5 · Full reproduction before interpretation

The fresh MLP experiment is **ten full-data runs × 500 epochs**, following the released architecture and training protocol. The GCN's ten complete L112 training runs are reused, with fresh all-node checkpoint replay in this lesson. The published targets are from OGB Table 6. A predeclared mean tolerance of 0.5 pp gives a descriptive numerical verdict; matching a mean cannot establish historical bitwise identity.

[[RESULTS]]

**Verification chain.** Archive and split identities → source-model output/gradient/optimizer checks → full histories and selected checkpoints → all-node original-model inference replay → independent correct-count reconstruction with the official evaluator. The graph-property implementation is checked separately against SQLite joins and grouping over the complete graph. No failed seed is dropped.

> **Scope check.** We reproduce selected MLP and GCN table entries, not every experiment in the OGB paper. The slice analysis is a new course experiment: the paper does not provide these exact slice targets. Modern libraries, explicit independent seeds, and the GCN sparse backend remain recorded deviations. Original tuning and historical randomness are not recovered.

## 6 · Let validation nominate the failure slice

Looking through many slices and reporting the most dramatic test gap is an **exploratory** discovery. It is worth investigating, but its dramatic size was part of why you chose it. A fresh evaluation must carry more weight than a story selected on that same evaluation.

**Our rule, written before MLP test results.** Among validation degree, homophily and class slices with at least 200 nodes, choose the smallest mean GCN-minus-MLP gap. Break ties by family name, then slice name. Keep year out of this selection because the validation year and test years do not overlap. Apply the chosen slice definition unchanged to test. The minimum support reduces tiny-slice noise but does not make the result causal or eliminate selection effects.

**Task 3 · separate nomination from evaluation.** Implement `choose_failure` and call it with validation rows only. Return no selection when none meets the minimum count. Keep all test rows in the report, but distinguish the validation-nominated slice from a new test-only discovery.

[[SELECTED]]

[[FIG:slices]]

**Explore the fixed results.** Change population and family below. The model weights, checkpoint choices, bin boundaries, and node predictions remain fixed. Controls only change which already-computed results are shown. Bar endpoints display mean accuracy; the table includes sample seed SD, support and pp gap. “Undefined” and empty slices remain explicit.

[[EXPLORER]]

**Do not force the expected story.** Low homophily can coincide with difficult features, uncommon classes, low degree or distribution shift. A low-homophily gap does not prove that message passing caused the errors. A GCN advantage in a slice also does not prove the same advantage for a new dataset.

**A discriminating next experiment.** Keep the architecture, initial parameters, normalization population, split and training budget fixed; vary access to particular neighbor messages, then retrain all declared seeds. Decide the intervention from validation evidence, and protect a new holdout if you have already used test findings to redesign the model. Compare that experiment with the weaker fixed-weight inference intervention: changing edges without retraining also changes the trained model's input distribution.

## 7 · Write the error report

Produce five connected sentences, backed by the saved table:

1. **Population and rule:** name the split, slice definition and count, and state where the rule was chosen.
2. **Measured difference:** report both mean accuracies, seed SDs and the pp gap; identify reused versus fresh training.
3. **Mistake pattern:** use GCN-only and MLP-only correct counts to explain the sign of the gap.
4. **Boundary:** state whether the slice uses hidden labels and why the association does not establish causality.
5. **Next check:** propose one controlled intervention and the evaluation data that remain protected.

[[TEACHBACK]]

**EXIT.** Submit the three live functions, the generated `l114-error-report.json`, and your five-sentence report. Explain why a globally better GCN can lose within a slice, why an isolate is not automatically zero-homophily, and why a true-label rule cannot route predictions at deployment. Author execution is not learner mastery: status remains **PENDING_WRITTEN_DEFENSE**.

[Student notebook](../labs/0114-ogb-error-analysis.ipynb) · [Executed solution](../labs/html/0114-ogb-error-analysis.html) · [Quick reference](../reference/ogb-error-analysis.html) · [Protocol and exact commands](../labs/l114-reproduction.md).

Ask the teaching agent about any unclear step, or paste your error report for feedback. Revisit the report after a few days and reconstruct its denominator and information boundary without looking.
