<div class="package-links"><strong>Your deliverable:</strong> defend one trained heterogeneous GNN pipeline from table IDs to final held-out predictions.<br><a href="../labs/0100-heterogeneous-gnn-checkpoint.ipynb">Student notebook</a> · <a href="../labs/solutions/0100-heterogeneous-gnn-checkpoint.ipynb">Executed solution</a> · <a href="../labs/html/0100-heterogeneous-gnn-checkpoint.html">Read the lab</a> · <a href="../labs/l100-reproduction.md">Reproduction contract</a></div>

## 1 · Close the notes: retrieve before building

[[WARMUP]]

Write four answers before opening the feedback. From [L091](0091-r-gcn.html), what is the denominator of a relation-wise mean? From [L096](0096-multi-relational-data.html), can author row 4 and paper row 4 be the same node? From [L098](0098-hetero-mini-batching.html), which sampled nodes receive supervised loss? From [L099](0099-rgcn-vs-hgt.html), what extra experiment is needed before attributing an HGT advantage to learned attention?

<details><summary>Feedback — open after answering</summary><p>R-GCN divides by the receiver's degree within that relation, then sums relation contributions. A node identity is (type, row ID). Only the input seed prefix receives the batch loss. A uniform-attention HGT intervention narrows the attention question; comparing two architecture families changes several operators at once.</p></details>

This is the Year 3 Q2 checkpoint. You already have individual operators, typed graph construction, sampling and comparison discipline. The tangible new skill is joining them into one pipeline whose result survives an audit. This supports the [mission](../MISSION.md): explain learned relational systems without hand-waving, and eventually evaluate them on real relational benchmarks.

A suggested route is retrieval and the worked trace first; then the notebook tasks; then the complete experiment and written defense. The experiment can run while you prepare your protocol table. A prepared solution does not demonstrate your mastery.

<!-- depth-walkthrough:start -->
## The big picture · defend the whole prediction procedure

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 100 / INTEGRATION AND DEFENSE</p><p><strong>Bring forward:</strong> L091–L093 supply encoders; L096 supplies identity; L098 supplies batches; L099 supplies comparison discipline. <strong>Follow:</strong> legal graph → paired sampled dependencies → seed-only learning → validation selection → frozen prediction → written claim. <strong>Carry forward:</strong> temporal learning adds availability to every boundary already tested here.</p><p><a href="0096-multi-relational-data.html">L096: typed keys</a> · <a href="0098-hetero-mini-batching.html">L098: seed boundaries</a> · <a href="0099-rgcn-vs-hgt.html">L099: model controls</a> · <a href="0070-foundation-model-checkpoint.html">L070: evidence defense</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

<figure class="model-map"><div class="model-map-scroll" tabindex="0" role="region" aria-label="Scrollable architecture: Checkpoint · make every boundary inspectable"><img src="../assets/architectures/100-checkpoint.svg" alt="Same typed graph → paired batches → trained candidates → frozen evaluation. Audit gradients at fixed weights. Training steps between batches follow a different trajectory from one full-batch update; mastery requires a defense."></div><figcaption>Same typed graph → paired batches → trained candidates → frozen evaluation. Audit gradients at fixed weights. Training steps between batches follow a different trajectory from one full-batch update; mastery requires a defense.</figcaption></figure>

### Build an argument whose links can fail independently

Start with the final claim: one complete procedure obtained a measured score on fixed ACM targets. To defend it, recover the target IDs, reconstruct the allowed graph, identify the sampled inputs, trace the actual encoder, show which labels entered the loss, and reproduce the selected model's predictions. A correct last metric cannot repair an incorrect earlier identity map.

This is the same standard as [L040's exit exam](0040-year-1-exit-exam.html) and L070's evidence chain, now applied to typed message passing. It is valid to conclude that graph context or attention did not help enough under this protocol. The checkpoint rewards a reconstructable claim, not a mandatory neural win.

### Trace one training seed across the interfaces

Select global paper ID 17. In its batch it occupies seed row 0. An author with global ID 2 may occupy author row 1, so a local author→paper edge `(1,0)` refers to `(author 2,paper 17)`. The author map and paper map are different functions. The encoder turns local typed features into hidden states; the classifier produces paper logits; seed row 0 contributes one training loss.

After selection, evaluate the frozen model on the full allowed graph and gather test IDs. Training batch positions are not persistent identities and must not become evaluation keys. The native [NeighborLoader contract](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html) and the source-pinned course checks make these handoffs explicit. The course sampler is not the HGT paper's HGSampling algorithm.

### Understand why a gradient audit can pass while training differs

Use a scalar parameter θ and two losses: `L₁=.5(θ−1)²`, `L₂=.5(θ+1)²`. Their equally weighted mean has gradient θ. At θ=0, one full-mean gradient step leaves θ at zero.

Now apply separate SGD steps of size .1. The first loss moves θ from 0 to .1. The second loss then has gradient 1.1, moving θ to −.01. Both per-example gradients were correct. They were evaluated at different parameters. This tiny teaching example isolates the same distinction as the checkpoint: an accumulated-gradient audit holds weights fixed, whereas ordinary mini-batch training updates between batches. Adam adds its own state to that difference.

The actual epoch also has unequal batch sizes. With 804 seeds and batches 256/256/256/36, the final batch's correct weight in a reported per-seed mean is `36/804≈.0448`, not .25. Correct reporting does not imply that the optimizer used the same trajectory as full-batch training. Keep objective accounting, gradient parity and optimization trajectory as three separate claims.

### Specify what every audit rules out

An ID round trip rules out a particular remapping error. A context-label perturbation rules out direct use of those labels in the seed loss. A full-neighbor forward comparison checks sampled dependency completeness under the audited operators. A fixed-weight gradient comparison checks objective weighting and differentiation. None of them establishes finite-fanout unbiasedness, historical paper reproduction, or general superiority across datasets.

The architecture sources remain [R-GCN](https://arxiv.org/html/1703.06103v4) and [HGT](https://arxiv.org/html/2003.01332v1); their published experiments remain distinct from this combined course comparison. Preserve every `INCOMPARABLE`, `NOT_ESTABLISHED` and `NOT_RUN` boundary in the ledger.

<details><summary>Try first: what would make the final claim too broad?</summary><p>Claiming that HGT is generally better from three seeds on one static graph; calling a uniform-HGT contrast an isolated proof about attention without its optimization limits; or calling this native-sampler course run a reproduction of HGT's full paper protocol. State the observed comparison and its conditions instead.</p></details>

**Your defense handoff.** Submit a trace for one typed edge, a derivation of the seed-weighted objective, and one paired result whose interpretation you could defend if its sign reversed. Then name the additional temporal availability checks needed for the next curriculum stage. The prepared artifacts support that defense; learner mastery remains **PENDING_WRITTEN_DEFENSE** until you write it.

<!-- depth-walkthrough:end -->

## 2 · The integration gap

[L098](0098-hetero-mini-batching.html) checked native sampling on a generated three-type graph. [L099](0099-rgcn-vs-hgt.html) compared models on a real graph using full-graph updates. Neither alone establishes that the comparison remains correct after local node renumbering, seed masking and finite neighborhood sampling are combined.

> **In plain terms.** A sampler makes a small working copy of the neighborhood. Its row numbers change. Its context nodes are useful inputs, but they do not all become training targets.

We use every conference-filtered ACM paper from the [pinned DGL loader](https://github.com/dmlc/dgl/blob/3d16000b4170fa741ed9e9667f22ba84d3493026/examples/pytorch/han/utils.py): 4,025 papers, 7,167 connected authors and 60 connected subjects. Paper features are 1,903 row-normalized word counts. Author and subject features are constant ones. Four directed edge stores represent paper→author, its reverse, paper→subject and its reverse. They contain 34,864 directed edges in total.

The target is a three-class conference grouping. Conference membership supplies labels and is excluded from every input edge store and feature. Otherwise the graph would contain the answer. The fixed course split contains 804 training papers, 403 validation papers and 2,818 test papers.

**Transductive** means all graph features and edges are available when learning, including those of held-out papers. Their labels remain excluded from the training objective. This is a static information contract; it is not a future-prediction claim. [L101 in the roadmap](../plan/year-3.md) adds the harder question of when an edge or feature becomes available.

## 3 · Model architecture: a sampled computation with a fixed target boundary

[[ARCHITECTURE]]

### Follow one seed through the two layers

**Sample backward from the target.** Start with up to 256 training paper IDs. For each relation, the native loader samples at most eight incoming neighbors without replacement at each of two hops. A **fanout** is that per-receiver, per-relation limit. Two hops cover the dependencies of two message-passing layers. See the [NeighborLoader contract](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader).

**Compute forward over local rows.** Each type has its own feature matrix `[N_type_in_batch, input_width]`. A type-specific affine adapter followed by tanh produces width 32. The same learned parameters are reused across batches. Edges use the batch's local row indices. They connect vectors inside this working graph, not arbitrary rows from the full tables.

**R-GCN branch.** For each receiver and relation, transform source vectors with that relation's matrix and average them. Sum the separate relation means, add the learned self transformation and apply ReLU. Repeat for layer two. This is the relation-normalized computation in [Schlichtkrull et al., Eq. 2](https://arxiv.org/html/1703.06103v4). The course wrapper adds feature adapters and a classifier; the named AIFB experiment has its own release architecture.

**HGT branch.** A head is one attention subspace; here four heads each have eight coordinates. Type-specific query, key and value projections produce the receiver query and source key/value. Relation-specific matrices transform keys and values. Dot-product scores, relation priors and division by √8 feed a softmax over all incoming edges for a receiver and head. The weighted message sum passes through GELU and a type-specific output map, then a learned gated residual and node-wise LayerNorm. Repeat twice. [Hu et al., §3](https://arxiv.org/html/2003.01332v1) motivates the typed attention route; the implementation follows the explicitly pinned modern release.

**Two controls.** Uniform HGT removes the score machinery and gives every incoming edge equal weight, retaining the message/output/residual route. Every common initial tensor matches HGT. The MLP uses paper features and ignores edges. It receives the same seed batches so its target exposure and update count match.

**Predict and train.** The classifier produces three logits per local paper. A logit is an unnormalized class score. Cross-entropy is the negative log probability assigned to the correct class after softmax. Only the first `B` paper logits belong to the requested training seeds. Context papers participate in message passing but contribute no direct label loss.

> **Scope check.** Native NeighborLoader is the controlled course sampler. It is not the paper's HGSampling algorithm. Static ACM has no relative temporal encoding. The full HGT CS lane retains its distinct sampler, time encoding, task and protocol.

## 4 · Worked trace: recover identities before trusting a score

**Worked example.** A batch has paper `n_id = [17, 4, 83]` and author `n_id = [9, 2]`. Here `n_id` maps local rows to global IDs within a type. Suppose there are two seed papers. The third paper, global ID 83, is context.

| Local author→paper edge | Global typed edge | Direct label loss? |
|---|---|---|
| source 1 → receiver 0 | author 2 → paper 17 | Paper 17 is a seed |
| source 0 → receiver 2 | author 9 → paper 83 | Paper 83 is context |

The first endpoint must be looked up in the author map. The second must be looked up in the paper map. Applying the paper map to both endpoints creates plausible integers with incorrect meaning. The lab checks reconstructed endpoints against the original edge store indexed by `e_id`, the loader's original-edge map.

[[IDENTITY]]

**Your first two tasks:** implement `global_edges`, then `seed_loss`. The CHECK cells use deliberately different ID maps and change context labels. Correct seed loss remains unchanged; gradients on context output rows are zero. Context *features* can still receive gradients through their messages. No-gradient-on-context-logits is not no-learning-from-context.

## 5 · Three different claims about mini-batching

[[PREDICT]]

### Claim A: the same seed prediction

With all neighbors retained (`fanout=-1`), two sampled hops should reproduce each seed's two-layer full-graph prediction. This relies on having the complete relevant incoming neighborhoods, the same weights, no dropout and node-wise normalization. A graph-wide normalization layer or an insufficient number of sampled hops would require a different argument.

The audit compares float64 computations at a frozen absolute tolerance of `2e-6`. It checks an adversarial fixture with an isolated paper and empty relations, then 23 fixed training seeds on the full real ACM graph at batch sizes 1, 7 and 23. These probes test integration; they are not an exhaustive proof over every possible batch.

### Claim B: the same accumulated gradient

A batch loss is a mean over its `B` seeds. An epoch objective is a mean over `N` seeds. To recover the full objective, multiply each batch mean by `B/N`, add the results, and differentiate while holding the parameters fixed.

**Worked example.** Five individual losses are `[0.2, 0.8, 0.5, 1.1, 0.4]`. A batch size of three gives means 0.50 and 0.75. Their equally weighted mean is 0.625. The correct mean is `(3/5)×0.50 + (2/5)×0.75 = 0.60`. The shorter final batch must not receive the weight of a full batch.

[[WEIGHTING]]

Your third task implements `batch_weight`. It is used by both the gradient audit and the reported epoch loss. For the real training split, the four batches have sizes 256, 256, 256 and 36. Equal averaging would over-weight the final 36 seeds.

### Claim C: the same optimizer trajectory

The audit does **not** step the optimizer between batches. The experiment **does**. Adam changes the weights and its moving state after each batch mean. Later gradients are therefore evaluated at different weights. Even all-neighbor sampling does not make those four steps identical to one full-batch step.

Finite fanout introduces another difference: the model sees a sampled neighborhood. A sampled mean may estimate a linear neighborhood mean under suitable sampling, but a nonlinear multi-layer prediction or attention normalization need not be an unbiased estimate of the full-graph prediction. Neither equal sample counts nor close scores establishes that property.

## 6 · Freeze the comparison before seeing test results

The experiment asks a narrow question: under this graph, split and fixed training budget, what does each complete prediction route achieve? The uniform-HGT intervention asks a narrower mechanism question inside the HGT family.

| Decision | Frozen rule |
|---|---|
| Information and targets | Same complete ACM graph, features and split for all arms |
| Sampling | Fanout 8 at two hops; batch 256; no replacement; directional edges |
| Paired samples | Seed order and sampled-edge fingerprints match across arms and rates for each seed/epoch |
| Model | Width 32, depth 2, HGT heads 4, dropout 0 |
| Training | Adam, weight decay .001, all 40 epochs, four updates per epoch |
| Search | Rates .003 and .01; seeds 0, 1 and 2 for each of four arms; 24 fits |
| Checkpoint | Earliest strictly lowest full-graph validation cross-entropy |
| Rate selection | Lowest mean selected validation loss across three seeds; smaller rate breaks ties |
| Test | Score selected checkpoints on the same complete test set after selection is frozen |
| Reporting | Accuracy, macro F1, individual seeds, paired differences, parameter counts and elapsed time |

**Macro F1** averages per-class F1 scores with equal class weights. For each class, F1 is `2TP/(2TP+FP+FN)`, with zero for an undefined denominator. **Sample SD** describes variation across these three initialization/sampling seeds. It does not describe variation across datasets or unknown future populations.

Sampling uses a separately reset epoch RNG: `100000 + 1000×seed + epoch_index`. Matching a seed label alone would be insufficient if different model initializations consumed different amounts of randomness. We also compare actual sample hashes across all eight fits associated with each seed. Uniform HGT copies common initial weights from HGT before training.

Equal width, update count and search size do not give equal parameter counts, FLOPs or optimization quality. The MLP still pays sampler overhead in this harness, so elapsed times are pipeline measurements, not isolated model-speed benchmarks. Validation and final test inference use the full graph; this course experiment does not establish a scalable all-sampled deployment.

## 7 · Inspect the measured evidence

Before opening the result table, predict which comparison could justify a statement about learned attention. Then explain one alternative explanation for a family-level difference.

[[RESULTS]]

[[RESULT_FIGURE]]

A weak R-GCN result under this fixed budget does not establish that relation-specific models are intrinsically weak. Optimization, the constant auxiliary features, the split and the wrapper all matter. An attention ablation changes parameter count and training dynamics as well as the weighting rule. Carry these limitations into the attribution table, rather than deleting an inconvenient result.

L099 used 60 full-batch steps; L100 uses 160 mini-batch steps and finite fanout. Comparing their scores changes several factors simultaneously. It is not a clean estimate of the causal effect of sampling. A follow-up would hold the optimizer/update schedule fixed and vary the sampler, using new validation-only decisions.

## 8 · Full reproduction: inspect the target, protocol and executed coverage

The [R-GCN paper](https://arxiv.org/abs/1703.06103v4) and [HGT paper](https://arxiv.org/abs/2003.01332v1) introduce methods and multiple experiments. Completing our checkpoint does not reproduce all those experiments.

| Evidence lane | What is provided | What it establishes |
|---|---|---|
| ACM checkpoint | All 24 fresh course fits, 960 epochs, 3,840 updates; traces/checkpoints/predictions | The complete declared course experiment, not a published table |
| R-GCN AIFB | Fresh ten-run full release-protocol port, original 140/36 split and full graph | A named published target comparison; historical identity remains INCOMPARABLE |
| HGT CS Table 2 | Full-setting model/sampler/trainer, pinned 8.1 GiB data identity, five-seed local/Modal/notebook commands | Runnable reconstruction track; full training NOT_RUN |
| Entire original papers | Other tasks/datasets and historical tuning/backend remain outside executed coverage | Full-paper parity NOT_ESTABLISHED |

The fresh AIFB mean is **95.8333%**, with **1.4640 percentage points** sample SD across ten runs. The Table 2 target is 95.83%. Numerical closeness is a useful observation; it does not identify the historical graph bytes, RNG sequence or Keras/Theano execution. The lab appendix visibly includes the complete AIFB implementation and trainer.

For HGT, CS Paper–Field L2 targets NDCG .403 ± .041 and MRR .439 ± .078 across five trainings. The course's full-setting port uses width 256, three layers and eight heads, with the source/deviation ledger linked below. The modern source differs from the archived publication-era code. L093 previously recorded a 10 GiB guarded load failure and a Modal GPU payment-method rejection. These are **prior findings**, not a fresh capacity experiment here. We do not relabel ACM, NN or a single update as the CS result.

The [full contract](../labs/l100-reproduction.md) records hashes, exact commands, model/optimizer/data/split/metric differences and delivery evidence. The standalone notebook contains the full visible CS implementation in a gated appendix. Turning on a flag does not upgrade a verdict; completed evidence must do that.

## 9 · EXIT: defend the pipeline

Submit the three task implementations, your fresh JSON evidence and an attribution table with columns **observation / held fixed / changed / alternative explanation / justified conclusion**. Include one row for HGT versus R-GCN, one for HGT versus uniform HGT and one for the all-neighbor parity audit.

Defend these five questions without reopening the solution:

1. Trace one sampled edge from local indices to global typed IDs, then through both layers to a seed prediction.
2. Explain why a context paper can send useful messages while its label remains excluded from the batch objective.
3. Derive the `B/N` weight for an incomplete final batch. State exactly when accumulated gradients match and when optimizer trajectories can differ.
4. Use the paired results to distinguish an architecture gap from an attention effect. Name one unresolved confound and one useful next experiment.
5. Explain why complete ACM execution, close AIFB scores and an unrun HGT CS operator have different reproduction statuses.

The teacher scores five axes 0–2: identity/graph contract; sampling/loss correctness; reproducible execution; comparison reasoning; reproduction defense. A checkpoint pass requires at least 8/10 and no zero in correctness or reproduction defense. Until your own artifacts and explanation are reviewed, status remains **PENDING_WRITTEN_DEFENSE**.

[[TEACHBACK]]

Ask the agent follow-up questions about any unclear step, or paste your EXIT evidence for a strict defense. Revisit the identity trace tomorrow and the gradient-versus-update distinction in a week. Next, [the temporal quarter](../plan/year-3.md) asks whether each input existed at prediction time—an issue this static checkpoint deliberately leaves open.
