<div class="package-links"><strong>Your tangible win:</strong> produce an attribution table that distinguishes an architecture difference, an attention intervention and a budget effect.<br><a href="../labs/0099-rgcn-vs-hgt.ipynb">Student notebook</a> · <a href="../labs/solutions/0099-rgcn-vs-hgt.ipynb">Executed solution</a> · <a href="../labs/html/0099-rgcn-vs-hgt.html">Read the solution</a> · <a href="../labs/l099-reproduction.md">Full reproduction contract</a></div>

## 1 · Retrieve before reading — five minutes

[[WARMUP]]

Write your answers before revealing the feedback. From [L091](0091-r-gcn.html): if a receiver has two author neighbors and one subject neighbor, does R-GCN divide all three messages by three? From [L093](0093-hgt.html): which axis does HGT's softmax normalize? From [L098](0098-hetero-mini-batching.html): does matching sampled seed IDs guarantee matching neighborhoods? From [L097](0097-negative-sampling.html): can a different evaluation candidate set change a model ranking?

<details><summary>Check after committing an answer</summary><p>R-GCN takes a separate mean within each relation and then sums those relation contributions. HGT normalizes over all incoming edges for each receiver and head in the implemented release. Identical seeds do not guarantee identical sampled context. Changing evaluation candidates can change the question being scored. A comparison must align the information, selection and evaluation interfaces as well as the model names.</p></details>

**Route:** about 30 minutes for the explanation, then a separate lab and written defense. This is the synthesis between typed message passing and the [Q2 checkpoint](../plan/year-3.md). It prepares the controlled GNN/graph-transformer comparison in L146 and the mission's demand for defensible relational-model evidence.

**Primary reading:** [Schlichtkrull et al., R-GCN, Eq. 2–3 and Table 2](https://arxiv.org/abs/1703.06103v4), followed by [Hu et al., HGT, §3 and Table 2](https://arxiv.org/abs/2003.01332v1). Read the operators first; do not compare the papers' headline scores across datasets. The [pinned DGL loader](../labs/sources/l099/dgl_han_utils.py) defines the ACM venue mapping; our split and training protocol are new.

## 2 · Define the comparison before choosing a winner

The previous lessons established that both families can pass messages on typed graphs. They did not establish that an HGT gain is *caused by attention*: HGT also changes projections, parameter counts, residual paths, nonlinearities and normalization. Comparing two complete systems changes all of these together.

Our central question is narrower: **under one declared data and training budget, how do these implemented families compare, and what changes when we remove HGT's content-dependent edge weights?** A family comparison answers the first part. A within-HGT intervention gives evidence about the second.

| Comparison | What changes | What a result can support |
|---|---|---|
| HGT versus R-GCN | Whole hidden operator, capacity and compute | Conditional family performance under this protocol |
| HGT versus uniform HGT | Learned scoring is removed; shared message/residual parameters remain | Value of the learned scoring mechanism including its capacity and optimization effects |
| Graph models versus MLP | Neighborhood access and graph operators | Whether this fitted graph pipeline improves over the feature-only baseline |
| Same model, another LR | Optimization trajectory | Sensitivity to the declared tuning budget |

None isolates a universal architectural truth. Even the attention ablation removes parameters and changes optimization. “Attention alone caused every point of gain” would still be too strong.

## 3 · One graph and one information contract

We download and SHA256-check the complete released `ACM.mat`, select the five conferences used by the DGL ACM raw example, and retain **all 4,025 selected papers, 7,167 connected authors and 60 connected subjects**. Compact auxiliary IDs preserve their original IDs in the evidence. This is **not** L092's 3,025-paper pickle or its historical split.

| Object | Representation | Use |
|---|---|---|
| Paper | 1,903 bag-of-words columns, normalized by each row's sum | Same inputs for all four arms |
| Author / subject | One constant feature equal to 1 | Learn typed starting states; no target-derived features |
| Paper ↔ author | 13,407 edges in each direction | Two-hop coauthor information |
| Paper ↔ subject | 4,025 edges in each direction | Two-hop subject context |
| Conference | Maps papers to three class labels | **Excluded from graph edges and input features** |
| Split | Seed 99; class-stratified 20% / 10% / remainder | 804 train / 403 validation / 2,818 test papers |

There is one subject incidence per selected paper. A subject can connect many papers: inspect degree before interpreting its influence. Subjects are a legitimate declared input in this static experiment; their relationship to venue categories can make this task easy. They do not establish useful deployment-time availability in a future temporal task.

**Transductive access is explicit.** Every selected paper's text and graph position are visible during training. Only 804 labels enter the training loss. Validation labels choose epochs and learning rates. Test labels enter final scoring after selection. Row normalization needs no corpus fitting; no labels create author or subject features. [L096's typed identity contract](0096-multi-relational-data.html) still applies.

## 4 · Model architecture: follow one paper to its prediction

[[ARCHITECTURE]]

Start with paper *p*. A type-specific affine adapter produces a 32-dimensional state followed by `tanh`. Authors and subjects have their own adapters. At layer one, an author receives information from that author's papers; at layer two, that author sends the resulting state back to *p*. This is how another paper's words can influence *p* through a shared author. The subject route works similarly. The MLP keeps only *p*'s words.

Every graph model has two layers, width 32 and the same four directed input stores. An affine classifier maps the final paper state to three logits. The loss uses only the train slice. Inference evaluates all paper states, then selects the test slice. There is no temporal encoding: this graph has no declared prediction-time interface. There is no neighborhood sampling: full-graph execution removes the sampler as a confound while [L098](0098-hetero-mini-batching.html) explains how to reintroduce it carefully.

### Step A · R-GCN first decides which relation produced a message

For each relation *r*, transform source states with `W_r`, average the incoming messages of that relation, sum the relation means, add one learned self transform and apply ReLU. This is the local mean-normalized Eq. 2 operator. The course model uses unrestricted relation matrices, plus biased adapters/self transforms and a classifier; it is not the featureless AIFB model from L091.

```python
src, dst = edge
message = x_source[src] @ W_relation
aggregate = zeros_for_receivers.index_add(0, dst, message)
mean = aggregate / degree_for_this_relation.clamp_min(1)[:, None]
# Sum one such mean per relation; then add the self path and apply ReLU.
```

The implementation computes the source transform once before indexing edges. Linearity makes this equivalent. An isolated receiver gets a zero neighbor contribution, so its self path remains meaningful. Basis decomposition from Eq. 3 is unnecessary for only four relations and is not introduced silently as another experimental variable.

### Step B · HGT lets the receiver help determine an edge weight

The receiver's type selects a query projection; the source's type selects key and value projections. Each width-32 state becomes four heads of dimension eight. Relation-specific matrices transform keys and values; a per-relation/head prior scales the query–key dot product. The implementation follows the [pinned modern pyHGT operator](https://github.com/acbull/pyHGT/blob/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG/pyHGT/conv.py), with temporal encoding disabled.

```text
score(edge, head) = prior[r, head] * dot(Q_type(dst), K_type(src) @ A_r) / sqrt(8)
message(edge, head) = V_type(src) @ M_r
alpha = softmax(score over every incoming edge of this receiver and head)
aggregate = sum(alpha * message)
```

Concatenate the heads, apply GELU and a destination-type output projection, blend with the old state through a learned sigmoid gate, then LayerNorm. Those operations matter: **R-GCN versus HGT changes much more than a softmax.** The lesson's independent check compares outputs and gradients against L093's archived modern-release port. That establishes operator agreement, not historical training-result reproduction.

### Step C · Remove the scorer without changing the rest of HGT

Uniform HGT deletes Q/K projections, attention matrices and score priors. Every incoming edge receives `1 / total_in_degree`. Value projections, relation message matrices, output projections, residual gates and LayerNorm remain. Every parameter shared with HGT starts with exactly the same tensor for a paired seed. Both models are then trained independently.

R-GCN divides by **relation-specific** degree and sums relation means. Uniform HGT divides by **total** degree. They are different even before considering transforms or residuals. Removing an attention scorer does not turn HGT into R-GCN.

## 5 · Worked trace: three messages, three answers

Freeze scalar messages to author A = 2, author B = 8, subject S = −4. Ignore transforms and self paths for this trace. Give attention logits `0, ln 3, ln 2`.

1. **R-GCN:** author mean `(2 + 8) / 2 = 5`; subject mean `−4`; their sum is **1**.
2. **Uniform HGT:** one mean over all three messages: `(2 + 8 − 4) / 3 = 2`.
3. **Attention:** exponentials are `1, 3, 2`, total 6; weights are `1/6, 3/6, 2/6`; the weighted sum is **3**.

Before using the control, predict what happens when the subject score rises to `ln 8`. Then duplicate author A and explain whether either mean is invariant. Duplicating just one neighbor changes the empirical distribution; invariance to *all* neighbors being duplicated is a different claim.

[[INTERVENTION]]

[[PREDICT]]

## 6 · Freeze the training and selection protocol

We run **24 fits: four arms × two learning rates × three seeds**, each for all 60 epochs. Learning rates are .003 and .01; Adam uses weight decay .001; dropout is zero; width is 32; depth is two. HGT uses four heads. Full-graph CPU execution uses two threads and deterministic algorithms. Every arm gets both learning rates and all three seeds. No early stopping shortens an arm's update budget.

At each epoch, save a checkpoint only if validation cross-entropy strictly improves. Ties keep the earliest epoch. After all fits, choose one learning rate per arm by mean best validation CE across the three seeds; a tie chooses the smaller rate. Only then restore selected checkpoints and evaluate test accuracy and macro F1. Seeds do not get their own test-selected learning rates.

The MLP allocates a paper adapter, two ReLU layers and a classifier. It has no unused auxiliary adapters. Graph models may still have allocated parameters that do not participate in the two-layer paper-only supervised path; the evidence reports both **allocated** and **active** counts. “Active” means autograd reaches the parameter, not that every entry has a nonzero derivative or that all parameters are statistically identifiable.

Equal width and updates do **not** match parameter count, runtime, memory or FLOPs. We report those differences instead of hiding them behind “fair.” Matching parameters or wall-clock budget would answer a different question and requires a separately frozen follow-up. The two-LR search is intentionally modest; weak optimization in one family is a limit on the conclusion.

## 7 · Read the measured evidence before writing the attribution

[[RESULTS]]

Read each seed pair before the mean. The differences are paired by declared seed and shared split; architecture-specific parameters are not magically identical. A sample SD across three seeds describes optimization variability on **one fixed graph and split**. It is not uncertainty across datasets, and 2,818 test papers are not 2,818 independent graph experiments.

**Sanity checks:** if the MLP is competitive, paper text already carries useful signal. If uniform HGT is competitive with HGT, do not claim adaptive scoring is required. If R-GCN's validation curve remains high at epoch 60, this protocol has not demonstrated its best achievable performance. Inspect both learning rates, selection epochs and loss traces before making a mechanism claim.

Your attribution table must contain an observation, the actual intervention, an alternative explanation and the narrowest justified conclusion. Example: “HGT outperformed this R-GCN configuration under the fixed budget; attention, residuals, normalization and optimization all changed; a within-HGT ablation is needed before crediting scoring.” Replace the observation with the measured values, including any result that contradicts your prediction.

## 8 · Full reproduction: three explicit evidence scopes

“Full” needs a target. This lesson's full **course experiment** means every one of the declared 24 fits, all epochs, frozen selection, saved checkpoints, raw predictions and an independent metric audit. It does not mean reproducing either paper's complete benchmark collection.

| Scope | Named target | Evidence boundary |
|---|---|---|
| ACM controlled comparison | The complete protocol in section 6 | New course experiment; no published score assigned |
| R-GCN published experiment | Schlichtkrull Table 2, AIFB, ten runs | Fresh complete release-protocol port; exact historical backend/RNG identity remains INCOMPARABLE |
| HGT published experiment | Hu Table 2, CS Paper–Field L2, five HGT trainings | Runnable full-setting port; execution NOT_RUN; source/protocol gaps remain |

The AIFB replay is separate: 50 updates per run, original 140/36 split, featureless inputs, width16 and unrestricted relation weights. The fresh mean is **95.8333%**, sample SD **1.4640 percentage points**. Agreement with the printed mean does not recover unknown historical seeds, original node ordering or Keras/Theano arithmetic. See the [fresh raw result](../labs/_paper_l099_aifb_results.json) and [L091 protocol audit](../labs/l091-reproduction.md).

The HGT CS archive is already available locally, but L093 recorded a decoding MemoryError under a 10 GiB address-space guard and a GPU-account execution rejection. Those are **prior observations**, not a new capacity test in L099. This lesson does not reclassify the smaller NN run or ACM run as CS. The full CS launcher preserves five seeds, full schedule and CS hash, and also offers the course R-GCN comparator. That extra comparator is not an exact reconstruction of the paper's R-GCN baseline. The [L093 protocol ledger](../labs/l093-reproduction.md) explains modern/publication-era source differences. Full-paper parity remains **NOT_ESTABLISHED**.

From the repository root:

```bash
.venv/bin/python labs/_verify_l099.py
.venv/bin/python labs/_run_l099.py
.venv/bin/python labs/_audit_l099.py
.venv/bin/python labs/_paper_l099.py --target aifb
# Full CS lane requires the hash-verified graph and sufficient host/device resources:
.venv/bin/python labs/_paper_l099.py --target hgt-cs --device cuda
```

The standalone solution reruns all 24 course fits outside the repository with inline loader/model/trainer code and freshly downloaded, verified data bytes. The execution harness uses bounded curl for the download after an initial urllib connection stalled. Browser checks, notebook execution, a clean dependency install, live Colab and deployment are separate claims. Consult the [execution](../labs/_execution_l099_results.json) and [delivery](../labs/_delivery_l099_results.json) records. Running code is not a substitute for your written defense.

## 9 · Lab: predict → implement → check → defend

Implement three live tasks in the student notebook:

1. `relation_mean`: hand-check two receivers and an isolated row; verify source gradients of 1/2, 1/2 and 1.
2. `receiver_softmax`: normalize over incoming edges across relations and heads; check stable scores, empty edges and gradient agreement with a dense oracle.
3. `choose_config`: choose by mean validation CE, not by best test score, best single seed or last epoch.

Then execute all fits, save `l099-fresh.json`, inspect your selected rates and compare per-seed predictions with the frozen author references. Runtime can vary; do not require byte equality for timing fields or serialized checkpoint files. Numerical agreement under the recorded environment is stronger than an unexplained matching aggregate.

**EXIT — PENDING_WRITTEN_DEFENSE:** submit the three implementations, fresh result file and your attribution table. Explain (a) why conference edges are forbidden, (b) why uniform HGT differs from R-GCN, (c) what is and is not held fixed, (d) why validation-only selection matters, (e) why neither ACM nor a matching AIFB mean proves full HGT reproduction. Propose one follow-up with a frozen budget that could overturn your preferred explanation.

[[TEACHBACK]]

Ask the agent follow-up questions about any equation, tensor, experiment record or uncertainty. Return tomorrow to reconstruct the three-message example from memory. Then use [L100's checkpoint specification](../plan/year-3.md) to defend a graph with at least three node types and a controlled architecture comparison.
