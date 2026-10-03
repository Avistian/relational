# B15 · Parameter-free encoders: limits and assumptions

<p class="subtitle">Research bridge · elective · one skill: identify what an encoder can know.</p>

[B14](b14-flattening-challenge.html) showed why strong relational features deserve a fair baseline. It left a question open: **if a learned encoder sees neighboring labels, must it gain an advantage?** Today you will construct two indistinguishable worlds and identify the observation that separates them. That is a useful test for your Year 6 proposal: specify what information the learned component adds before paying to train it.

**Route:** read and trace for 20 minutes; implement the lab for 25–40 minutes; return after 1, 7 and 30 days. Prerequisites are a table row, a binary label and the idea of a held-out query; each is restated below. [Student notebook](../labs/b15-parameter-free-encoders.ipynb) · [Executed solution](../labs/html/b15-parameter-free-encoders.html) · [Reference](../reference/b15-parameter-free-encoders.html)

## Recall before reading

<div id="b15-warmup"></div>
<noscript><p>Recall B14: which information must remain fixed when comparing two predictors? Why can an old event have a label that is not yet available?</p></noscript>

## 1 · Draw the information boundary

A **query** is the row whose label we want to predict. A **label** is the outcome, such as whether a study succeeds. **Support** consists of other examples with labels we are allowed to observe. A row may be related to the query yet still have an unavailable label.

An **encoder** turns a database neighborhood into a feature vector. A **parameter-free encoder** uses fixed operations, such as counts, means or column-preserving summaries, without learned encoder weights. The downstream **prediction head** may still be a very large pretrained model. Parameter-free encoding does not mean that the whole system has no learned parameters.

An **H-hop neighborhood** contains rows reachable from the query through at most H key relationships. It is a boundary on access, not a guarantee that every row inside it is usable. A known neighboring label can enter as an encoder feature. Separately, a head can read labeled support examples through **in-context learning**: predictions change with examples supplied at inference while model weights stay fixed. [Paper §§2–3](https://arxiv.org/html/2607.05476v2#S2)

**Worked example.** Predict query Q at day 10. A remote support row happened on day 2 and its outcome was available on day 4. A second row happened on day 3 but its outcome only becomes available on day 11. The first label is usable if the declared context admits that row. The second is unavailable even though its event is older than the query. Q's own label is always hidden.

{{VISIBILITY}}

**Predict:** does widening the context help if the new label arrives tomorrow? Change one control, state which row becomes visible, then inspect the result. The fixed baseline is remote support permitted and cutoff day 10.

> **Scope check.** The lab explicitly models event time and available-at time. This is our teaching contract. It does not establish the publication time of every historical field in the paper's benchmarks.

## 2 · Two worlds, one local observation

Suppose the query's neighborhood contains a label **b = 1**. Two task rules remain possible:

| World | Query rule | Query label when b = 1 |
|---|---|---:|
| Copy | Use b | 1 |
| Flip | Use the opposite of b | 0 |

The local observation is identical. A fixed encoder receives the same input in both worlds, so its representation cannot tell them apart. A fixed prediction head receiving only that representation has the same problem, regardless of how many parameters it has.

The local neighbor label b is **not** a complete training example of this query rule: we have not observed that neighbor's own generating context. Treating b as a pair `(input=b, target=b)` would quietly assume the answer is “Copy.”

We denote the world by **theta**:0 for Copy and1 for Flip. The operation **XOR** returns 1 when its two binary inputs differ. Thus the query rule is `y = b XOR theta`. If both worlds are equally likely, the local head assigns probability 0.5 to label 1.

<div id="b15-predict"></div>

Now permit a separate support example with input 0. Its observed label is theta: 0 in Copy,1in Flip. The head can identify the rule and predict the query. The support label teaches the task relationship; it does not reveal the query label itself.

{{RULES}}

**Fixed in this comparison:** same worlds, query values and rule family. **Changed:** whether the head can observe the distinguishing external support. **Measured:** query probabilities and errors. This isolates an information-access change, not a benefit from learned encoder weights.

### What Proposition 3.1 actually says

> **In plain terms.** Knowing some local labels need not tell you how a new database's task works. Information outside the local view may determine the answer.

Proposition 3.1 constructs distributions over the rest of the database for a given nonempty label-bearing neighborhood. The full database determines the query label, yet any fixed encoder and head restricted to that neighborhood face a chance-level lower bound under the paper's divergence criterion. Its second part establishes the existence of a task-specific predictor using a parameter-free representation and a finite ReLU network. A **ReLU** replaces a negative input with zero. Existence does not guarantee efficient learning from a small sample. [Proposition 3.1 and Appendix D.1](https://arxiv.org/html/2607.05476v2#S3)

The crucial quantifier is **“there exist distributions.”** It rules out a universal guarantee. It does not say every real task is adversarial, every pretrained model performs at chance, or learned encoders are useless. Our four-world lab illustrates indistinguishability; the paper proves its statement with a richer parity-based graph construction. An in-context head given broader support has a different input boundary from the restricted head in the lower bound.

## 3 · Labels can agree while feature relevance stays ambiguous

A label can also be used to decide **which feature column matters**. That is different from using the label itself as a predictive feature.

**Worked example.** Two columns A and B agree on all local labeled examples:

| Row | A | B | Label |
|---|---:|---:|---:|
| Local support 1 | 0 | 0 | 0 |
| Local support 2 | 1 | 1 | 1 |
| Query | 0 | 1 | hidden |

Both rules—“label equals A” and “label equals B”—fit the observed support perfectly. They disagree on the query. More copies of the same local rows cannot decide between them. A permitted external row `(A=0,B=1)` with label 0 identifies A; with label 1it identifies B.

{{COLUMNS}}

Let **S** be the relevant column, either A or B. **Mutual information**, measured in bits, quantifies how much observing one variable reduces uncertainty about another. Under our equal-weight worlds, the local observations provide **0 bits** about S. The distinguishing external label provides **1 bit**, enough to resolve two equally likely alternatives. This says nothing about how many bits a real database needs.

Proposition 4.1 gives an analogous existence result: labels can be deterministic functions of a relevant feature set while the visible neighborhood labels provide no additional information about that set. The paper's construction is broader than the two-column table here. Its statement conditions on an ego-network with nonempty feature columns and observed labels; it does not prohibit informative local labels on other distributions. [Proposition 4.1 and Appendix D.2](https://arxiv.org/html/2607.05476v2#S4)

**Important distinction:** failing to identify S does not always prevent predicting y. When a query has A=B, either rule gives the same answer. Across all eight `(S,A,B)` worlds, the local predictor achieves 75% accuracy even though information about S remains zero.

## 4 · Run the complete finite experiment

The lab enumerates every declared world; there is no stochastic training run. A probability of at least 0.5 is classified as1. The **Brier score** is the average squared difference between predicted probability and binary truth; smaller is better.

{{RESULTS}}

The leaky control looks perfect because it reads the answer. We therefore overwrite both the hidden query label and the future label in all four possible ways per rule world. Legal predictions stay unchanged across all 16 interventions; the leaky output follows the overwritten query label. Correct numbers alone are insufficient: audit the inputs that produced them.

> **Evidence boundary.** These are exact results on a tiny constructed distribution, with no sampling uncertainty to estimate. They are neither a learned-versus-fixed model ranking nor reproduced real-data AUROC. Broad conclusions require representative tasks and a matched information budget.

## 5 · Paper reproduction: inspect the executed system

The primary reading is [Parameter-Free Encoders Remain Viable for RDB Foundation Models, v2](https://arxiv.org/html/2607.05476v2). Read §§3–4 for the argument, then Appendix A and Table 5 to separate it from the empirical pipeline.

Our selected target is **Table 5, RDBLearn v1.1, rel-trial/study-outcome: AUROC 0.7271**. **AUROC** measures how often a randomly chosen positive example receives a higher score than a negative, with half credit for ties. It is not the finite lab's accuracy.

The paper's empirical RDBLearn pipeline creates relational features and passes them to tabular foundation models. Appendix A says its encoder does not yet receive neighborhood labels. This means its strong benchmark performance does not directly measure the benefit of label-aware encoding discussed in the theory. [Appendix A](https://arxiv.org/html/2607.05476v2#A1)

### A concrete source discrepancy

The released README claims target-history augmentation defaults ON. At pinned commit `78561f0`, the actual configuration sets `enable_target_augmentation=False`. The estimator stores and inserts target history only behind this flag. The **code default** agrees with the paper; the README does not. However, a default is not evidence of the exact configuration used for Table 5. [Pinned config](https://github.com/HKUSHXLab/rdblearn/blob/78561f0a9c1dd231d44659e761d5d85e18c82f6e/rdblearn/config.py)

{{PAPER}}

The 25-file authenticated source packet includes the paper and all 24 tracked release files. The released classification example uses F1 defaults. We will not replace its dataset name with “trial,” guess the remaining settings, and call the result Table 5 reproduction. The [contract](../labs/b15-reproduction.md) lists each missing detail; the [audit operator](../labs/_reproduce_b15.py) checks hashes and refuses a paper run while they are unresolved. Full six-benchmark results and backbone pretraining remain NOT_RUN. B14's successful selected TabPFN-Rel experiment remains a different experiment.

## 6 · Your lab and written defense

Open the [student lab](../labs/b15-parameter-free-encoders.ipynb). Implement three mechanisms with immediate checks: filter legal labels using full query identities and clocks; retain compatible task rules; retain compatible feature columns. The solution contains visible implementation and reruns from an empty directory.

**Before revealing an answer:** explain why a fixed encoder cannot distinguish identical local observations. Then identify a legal support observation that changes that conclusion. Finally describe an input intervention that distinguishes the legal predictor from the perfect leaky control.

<div id="b15-teachback"></div>
<noscript><p>Write four sentences: state the theorem's existence quantifier; explain the copy/flip ambiguity; describe legal external support; explain why query-label leakage is invalid.</p></noscript>

**Exit ticket:** submit your three passing implementations, the two finite result tables, and a 150-word defense distinguishing local encoder labels, head support labels and hidden query labels. Passing author checks does not complete your defense: status is **PENDING_WRITTEN_DEFENSE**.

**Return in 1 day:** recreate the copy/flip table from memory. **In 7 days:** build a different ambiguous feature pair. **In 30 days:** audit the information boundary in your proposed learned model. Ask the agent follow-up questions whenever a definition, proof step or implementation is unclear.

**Next:** use [B16's interpretability questions](../plan/year-5-6-bridge.md#b16) or return to the [core bridge route](../reference/curriculum.html#research-bridge). The research consequence is precise: justify the learned encoder using matched inputs, and test whether a simpler representation plus a capable head already has the information needed.
