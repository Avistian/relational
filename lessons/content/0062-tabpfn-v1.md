## What you will be able to reconstruct

A new classification table arrives with a few hundred labeled rows. In lessons 042–054, those rows drove an optimizer: initialize a network, compute a supervised loss, update its weights, select a checkpoint, then predict. TabPFN moves most of that learning into an earlier phase. Its weights have already learned from millions of synthetic classification tasks. At prediction time, the labeled table becomes **input to a frozen network**. Replacing context labels changes the prediction without changing a weight.

This lesson's tangible outcome is a working historical TabPFN v1 predictor whose normalization, attention, residual updates and ensemble reduction you can inspect and complete. The lab copies the actual released weights into your visible implementation and measures real predictions. This is **pretrained inference through your implementation**. It is not fresh pretraining and not a reproduction of the paper's aggregate benchmark. The distinction matters: correct randomly initialized attention does not yet implement a useful learned prediction algorithm.

The relational mission requires strong table baselines before claiming that database structure adds value. TabPFN provides a particularly different baseline: it transfers an inference procedure learned across tasks. Its row tokens still represent one flat feature table. They do not create entity relationships, temporal guarantees or database joins. Those remain properties of how we build the prediction task.

**Retrieve first, without scrolling:** What does the label enter in an ordinary MLP: its input or its loss? In lesson 061, why did the GP sampler have to draw context and query targets from one shared function? Could a query attend to its own feature token without seeing its target? Keep the last answer: v1 changes that edge for computation, not because the edge necessarily leaked a label.

## From a PFN objective to a tabular prior

A **prior** is a distribution over possible data-generating mechanisms before observing the current labeled table. Write a mechanism as φ. A draw of φ can specify a graph, its weights, noise levels, which nodes are observed and the classification thresholds. Conditional on this mechanism, draw a whole dataset D. Drawing a new mechanism for every row would destroy much of the shared structure that makes context useful.

The posterior predictive distribution, or **PPD**, averages predictions over mechanisms after conditioning on context Dᶜ:

`p(y* | x*, Dᶜ) = ∫ p(y* | x*, φ) p(φ | x*, Dᶜ) dφ`.

The integral denotes averaging over all allowed mechanisms, weighted by their posterior plausibility. In a joint SCM prior, the observed query features can themselves inform which mechanism is plausible; fixed-design notation often suppresses this additional conditioning. In practice this inference can be expensive. A prior-data fitted network learns an approximation `qθ(y* | x*, Dᶜ)` by repeatedly sampling tasks, hiding some labels and minimizing their negative log probabilities. θ denotes the Transformer weights. A query is simply a row whose label the network must predict. **Cross-entropy** for a class target y is `−log qθ(y)`, measured in natural-log units, or nats.

For a fixed context and query feature vector, expected cross-entropy decomposes into the entropy of the true conditional distribution plus `KL(p || qθ)`. Entropy is fixed when θ changes; KL is a nonnegative measure of distribution mismatch. Therefore the unrestricted optimum is the prior's PPD. The statement is about the selected prior and the ideal optimum. Finite training, finite capacity and a mismatch between synthetic and real data can all leave errors. The GP experiments in lesson 061 made those gaps measurable because their oracle was analytic; the full TabPFN prior has no comparably cheap exact oracle. [Paper §§2–4 and Algorithm 1](https://arxiv.org/html/2207.01848v6#S2).

The original v1 prior mixes two mechanism families with equal probability. A **Bayesian neural network prior** samples an architecture and weights, then produces targets from input features. A **structural causal model**, or SCM, specifies a directed acyclic graph and one noisy assignment for each node. A directed edge indicates that one node's value enters another node's generating function. An acyclic graph has no directed loop, so values can be generated in order. The SCM samples feature nodes and a target node from a sparse layered graph; the target can be a cause or an effect of observed features. Neither family means that the Transformer samples a new graph for each real prediction. Pretraining transfers the consequences of those sampled mechanisms into its weights.

The prior includes distributions over depth, hidden size, weight scale, noise, sparsity and activation. “Simple” here means the particular sampled families favor graphs with relatively few nodes and parameters; it is not a theorem that every simple real task will be easy. Graph sparsification, block sampling of observed nodes and feature-specific scales create differing interactions and correlations. Label generation first creates continuous targets, then samples thresholds, assigns interval IDs and permutes the IDs. This permits multiclass tasks and class imbalance without treating the numerical class ID as an inherently ordered semantic quantity.

For example, continuous targets `[-.4, .2, .8]` and thresholds `[-.1, .5]` produce interval IDs `[0,1,2]`. A sampled relabeling `[2,0,1]` can then turn them into `[2,0,1]`. Bounds are sampled from the generated task's target values; they are a property of the synthetic task construction, not a preprocessing rule fitted to real test labels. Lesson 063 will implement and interrogate that prior. Here we implement the learned predictor that consumes its consequences. [Paper §§4.1–4.5; Appendices C and E.4](https://arxiv.org/html/2207.01848v6#S4).

**CHECK your claim:** An SCM-based prior does not identify a causal graph from your table. The paper explicitly predicts associations directly without producing a graph. Observational prediction, an intervention such as setting a treatment and a counterfactual for one person are different targets. High classification accuracy establishes only the first of these unless the task and evidence establish more.

## Model architecture

Let C be the number of context rows, Q the number of queries, F the number of active numeric features, B the number of independent tasks or views processed together, d the hidden width and H the number of attention heads. One **token** is the vector representation that attention treats as an element of its sequence. In v1, one token is a **whole row**, not an individual feature. The released model has d=512, H=4, 12 Transformer blocks, feed-forward width 1024 and a maximum of 10 output classes. Thus one head has 512/4=128 coordinates. The feature input has a fixed maximum width of 100.

Before tokenization, context-fitted transformations standardize features, optionally apply a power transform and compress outliers. After scaling and zero padding, the input is `B×(C+Q)×100`. A linear feature encoder maps each row to 512 coordinates. A separate linear label encoder maps a scalar context class ID to 512 coordinates. The context token is their sum:

`hᵢ⁰ = Wₓ xᵢ + bₓ + Wᵧ yᵢ + bᵧ`, for `i < C`.

The query token is only `hⱼ⁰ = Wₓ xⱼ + bₓ`. **There is no learned “unknown label” embedding on this released path.** A class-0 context row still has the label encoder's bias, while a query does not receive that encoder at all. No positional encoding is active: row order is not intended to carry meaning. The model signature accepts `x` and context `y`; it never needs a query target placeholder.

<!--figure:architecture-->

Every block first makes query, key and value projections from the current row states. A **query vector** asks which context information matters for the receiving token; keys determine compatibility; values carry the information to combine. These uses of “query” differ slightly: all rows produce attention query vectors, while only the unlabeled rows are prediction queries. Every receiving row can read context keys and values. No receiving row can read query keys or values.

The 12 blocks use the same kind of operation but have different weights. Their attention and feed-forward sublayers each have a residual addition followed by LayerNorm. **LayerNorm** normalizes across the hidden coordinates of each row and applies learned coordinate-wise scale and offset. It does not estimate population moments from the set of query rows. The feed-forward sublayer is Linear 512→1024, GELU, Linear 1024→512. GELU is a smooth nonlinear activation. The query states after block 12 enter Linear 512→1024, GELU, Linear 1024→10. Only the first K logits are used for a task with K observed classes.

The model has 25,821,706 learned parameters in the copied checkpoint. Pretraining updates these weights using held-out synthetic labels. Real-table inference keeps them fixed and changes only input features and context labels. The checkpoint stores additional training configuration, but serialized callable descriptions and settings such as local batch size do not establish that its metadata independently reconstructs the paper's distributed training run. [Paper E.1–E.3](https://arxiv.org/html/2207.01848v6#A5); released [`TransformerModel.forward`, `TransformerEncoderLayer.forward`](https://github.com/PriorLabs/TabPFN/tree/v1.0.0/tabpfn).

## Trace information access

For one head, let `U` contain the projected query vectors for all C+Q receivers and let `Kᶜ,Vᶜ` contain the projected context keys and values. The attention output is

`A = softmax(U (Kᶜ)ᵀ / √128) Vᶜ`.

The softmax acts over the **sending context row** dimension: each receiver gets weights summing to one. Scaling by the square root of the head width controls how dot-product magnitude grows with dimension. Splitting the projection into four heads lets each head combine its own value coordinates; concatenate the resulting heads and apply an output projection before the residual addition.

A rectangular matrix of `(C+Q)×C` scores has exactly `C²+CQ` entries per head. The original lesson-061 PFN could attend to each query's own feature token as well. Appendix E.2 explicitly removes these query self edges in v1. For C=3,Q=2, the old conceptual 5×5 mask allowed 17 entries; the v1 mask allows 15. The efficient implementation avoids forming blocked query columns, so it allocates 15 score entries rather than a full 25-entry matrix. This saving does not remove the C² context cost.

<!--figure:attention-->

A small fixture exposes the arithmetic. Take one head of width 2 with receiver `[1,0]`, context keys `[0,0]` and `[√2,0]`, and context values `[1,0]` and `[0,2]`. The scaled scores are `[0,1]`. Softmax gives approximately `[.268941,.731059]`. The weighted value is therefore `[.268941,1.462117]`. These are synthetic projected vectors, not recorded activations from the checkpoint. The figure holds scores and context fixed while changing an unrelated query's value; that value has no sending column, so the existing query's output cannot change.

Why can the model still distinguish two query rows if neither query can read itself? Its own features determine the query projection, and the residual preserves its state. With attention output a, the block does

`u = LayerNorm(h + a)` and `h′ = LayerNorm(u + W₂ GELU(W₁u+b₁)+b₂)`.

Removing self attention is not removing the row. Moving LayerNorm before these additions implements a different computation; copying pretrained weights into that different model is invalid even when every shape matches.

The independence argument must cover every layer. Context states only depend on context at layer 1, and this remains true inductively. Each prediction query combines those context states with its own residual path. A second query cannot change the first query through attention, through the row-wise feed-forward network or through LayerNorm. A whole-pipeline check is still needed: a transform fitted on all query rows would create a separate dependency before tokenization. Even context-only fitting is insufficient if transforming a new query triggers a fallback that changes previously transformed rows; the counterexample below exposes precisely that failure. Dropout must also be off for the deterministic comparison.

**Predict before CHECK:** Permute context rows and labels together. The context keys, values and attention weights permute together, so their weighted sum should be preserved up to floating-point error. Now exchange two feature columns without changing the encoder. There is no equivalent architectural guarantee: each input coordinate has its own encoder weights. Permuting row order, rotating feature indices and rotating the geometric feature space are three different transformations.

## Preprocessing is part of the predictor

Pretrained weights expect a particular distribution of inputs. “I copied the Transformer” is insufficient if I change the feature scaling or label mapping. On the audited numeric v1 path, each view starts with context-only sample means and standard deviations. The denominator uses sample standard deviation, with divisor C−1, plus 10⁻⁶. Standardized values are clipped to [−100,100]. Features constant in context are removed. Query rows do not decide which columns survive.

For a one-feature context `[1,3]`, mean=2 and sample standard deviation=√2. A query value 101 becomes approximately 70.7106 after standardization. A much larger query may hit the clamp, but it must not move the context's standardized values. Using `[1,3,101]` to fit the moments would change the existing context representation and make the prediction depend on which other queries happened to be submitted.

The optional Yeo–Johnson power transform estimates one transformation per feature on context rows and applies it to context and query. It can reduce skew and handles negative as well as positive values. Its parameter is not estimated from the query table. The released wrapper keeps the existing column when fitting or transforming it raises an exception; the lab preserves this behavior for the supported numeric path and checks it against the original implementation. Power transforms are view choices, not gradient updates to the Transformer.

**A measured failure of whole-wrapper isolation.** The audited historical wrapper catches an exception around both fitting the power transform and transforming the combined rows. An extreme extra query can make transformation overflow. The fallback then retains the untransformed column for the *entire* combined table, including context and the original query. Context-only fitting did not prevent this control-flow dependency. A reproducible 12-row one-feature context with values around 10⁻⁹ fits a Yeo–Johnson exponent about −219.54. Adding a query of −100 triggers an `expm1` overflow in the recorded environment. The original query's class 1 probability changes from .382355 to .484686 with four views; the released reference agrees within 1.4×10⁻⁶. The one-view none-transform control has zero observed difference. [Saved source-checked counterexample](../labs/_query_fallback_l062_results.json).

This is a historical wrapper behavior, not an attention-mask failure and not a silently repaired implementation. The notebook recreates it using your live predictor. The exact numerical trigger depends on the recorded library versions, but the structural lesson is broader: prove that preprocessing **including its failure path** stays fixed when proving pipeline independence. Ordinary batch-isolation fixtures pass; they do not establish a universal property.

Next comes **outlier softening**. The wrapper estimates context mean and standard deviation, identifies context values within four standard deviations, re-estimates moments on these retained values and uses the resulting lower/upper bounds for logarithmic tail compression. This is not simply deleting query rows or clipping all values to ±4. The implementation is visible because the exact transformation affects copied-weight predictions. Missing values survive these moment calculations until the feature encoder replaces them with zero. In this representation, zero does not retain an explicit missingness indicator; that limitation is consistent with the paper's weaker emphasis on missing data.

<!--figure:preprocessing-->

Finally, with k surviving features and maximum K=100, multiply values by `K/k`, then append K−k zero columns. The paper calls this a flexible encoder: one fixed linear layer can accept tables with different feature counts. For the miniature K=6 fixture, `[1,−2]` becomes `[3,−6,0,0,0,0]`. The actual checkpoint uses K=100. Multiplication by √(K/k) is an available source option but is **not active in this checkpoint**. Nor does multiplying by K/k guarantee equal variance for every possible covariance structure; it is the specific magnitude convention learned during pretraining.

The supported lesson path handles numeric columns, context-fitted transformations, at most 100 input columns and 2–10 contiguous class IDs. It excludes ranking transforms, feature subsampling beyond 100, explicit categorical-column handling and query-fitted normalization. The official source contains these optional paths; passing our supported-path tests does not certify all of them. All-constant context features are rejected explicitly in this lab rather than silently producing a degenerate feature-count division.

## Ensembling means aligning the answer space

The released predictor can form several **views** of the same table. A view rotates feature-column indices, rotates class IDs and selects raw-standardized or power-transformed features. A column-index rotation by two changes `[x₁,x₂,x₃,x₄]` to `[x₃,x₄,x₁,x₂]`; it does not multiply features by a dense orthogonal rotation matrix. With F original columns and K classes there are at most 2FK rotation/transform combinations. Requesting more views cannot create more unique combinations in this procedure.

Class rotation is `y′=(y+s) mod K`. The predicted coordinate for original class j is therefore the rotated model's coordinate `(j+s) mod K`. Reverse this mapping **before** combining views. Otherwise one view's “class 0” means another event than a second view's “class 0”. The feature and label transformations apply consistently to context and query; only context labels are available.

<!--figure:ensemble-->

In the fixture, view A has logits `[3,0,0]` with shift 0. View B has logits `[1,0,0]` with shift 1. Rotate view B's answers back to `[0,0,1]`, then average to `[1.5,0,.5]`. With illustrative temperature 1, softmax gives about `[.6285,.1402,.2312]`. Averaging each view's probabilities instead produces a different distribution. The historical 0.1.11 wrapper defaults to **averaging aligned logits**, then dividing by temperature .8 and applying softmax. Temperature smaller than one makes a fixed unequal logit vector sharper; it does not change which coordinate is largest.

This detail is derived from `transformer_predict`, not inferred from the paper's generic word “ensemble”. Its default ordering alternates none/power views within a sampled pair of rotations. A single requested view therefore uses the none transformation; it is not a random 50/50 choice between none and power. The random seed chooses view rotations; it does not retrain the checkpoint. Constant-feature removal occurs after the source draws feature shifts from the original column count. The lesson preserves this slightly awkward slicing behavior so its wrapper actually matches the released predictor.

The ensemble encourages robustness to arbitrary column and class indexing, but a finite collection of cyclic rotations is not all possible permutations. Pretraining rotations can encourage learned invariance without proving exact invariance. Keep the exact context-row permutation argument separate from these empirical transformations.

## Run the historical model

The original reference is the `tabpfn==0.1.11` wheel, identified by a SHA-256 digest, and the released `prior_diff_real_checkpoint_n_0_epoch_42.cpkt` checkpoint with SHA-256 beginning `3c9aadae`. A digest identifies exact bytes. The lab independently implements the inference mechanism, copies every model weight with a strict key/shape check and excludes only the criterion's class-weight buffer from the mapping. It imports the released package only for validation. You can read the complete model and inference wrapper in the notebook.

Validation has three layers. Small exact fixtures expose the intended arithmetic. Full copied-weight forward and gradient comparisons establish parity with the named source computation. Finally, complete wrapper comparisons cover binary and multiclass data, a constant feature, missing numeric entries and an extreme query. Context permutation, context-label sensitivity and query batching probe complementary properties. A predictor returning the same constant distribution for everything could pass a query-isolation check, which is why isolation alone is weak evidence.

The actual executed teacher run completes your five live functions and passes them into a real inference experiment. The functions are not decorative exercises followed by an imported completed model. Call counters verify that preprocessing, attention, residual updates and ensemble aggregation execute downstream. Kernel identity records the actual visible functions, class methods, defaults, keyword defaults, configs and dependency versions. Repository source hashes are reference identity; they do not certify an edited notebook kernel. A mutation CHECK changes a helper, method, default and preset and requires their identities to change. No checkpoint resume is offered.

## What was measured here

**Prediction before results:** Will four views improve every split over one view? Does giving correct context labels outperform shuffling them? Write the direction you expect and one reason it might fail on an individual small test set. The first intervention changes a fixed inference recipe; the second removes much of the legitimate feature-label relationship while preserving label counts. Neither is new pretraining.

The new author panel uses diabetes (OpenML 37), blood transfusion (1464) and Wisconsin diagnostic breast cancer (the same UCI dataset available as sklearn `load_breast_cancer`, OpenML 1510 in the paper). All rows are retained. For each dataset, three stratified 50/50 splits use seeds 0,1,2; both view budgets share the exact rows and labels. We compare one and four views of the same frozen checkpoint. **AUC**, area under the receiver operating characteristic curve, measures ranking of positive relative to negative examples. Binary **log loss** scores the probabilities assigned to actual labels; lower is better. Both are recorded with individual predictions and view logits. The raw features and labels have separate hashes, and row indices are retained.

<!--figure:results-->

Read the points within each dataset first. Means and sample standard deviations summarize the three split/view-seed combinations. They are descriptive variation over overlapping resamples of the same dataset, not independent-dataset confidence intervals. No Friedman test or critical-difference claim is appropriate for this two-budget mechanism comparison. The three datasets are not enough to recover the paper's full model ranking, and repeated splits do not create new datasets.

The paper's main result evaluates 18 numeric datasets without missing values, drawn from a 30-dataset OpenML-CC18 subset with at most 2,000 rows, 100 features and 10 classes. It uses five 50/50 splits and normally 32 views. Baselines receive declared tuning budgets and use their own prescribed preprocessing. The local panel aligns the dataset identities, full-row small-data regime, 50/50 split proportion and pretrained model family. It differs in dataset count, view budget, number of splits, exact split construction/row order and absence of tuned competing baselines. WDBC comes from sklearn's UCI copy rather than a proven byte-and-order match to the paper's OpenML representation. Therefore every local score is **INCOMPARABLE to aggregate paper-result reproduction**, even when numerically close to an individual published score. [Paper §5.2, Tables 1–2, F.3](https://arxiv.org/html/2207.01848v6#S5.SS2).

Appendix F.3 constructs 150 meta-validation datasets; Appendix B.5 drops flags because enough valid splits could not be generated, leaving 149 evaluated datasets. These datasets informed method development; they are not 149 untouched meta-test datasets. Its prior ablation uses smaller training budgets and compares BNN, SCM and their mixture; those results do not isolate a particular encoder or prove that causal priors dominate all other priors. Our pretrained ablations cannot re-estimate that prior-training comparison. The older local three-task/180-context result remains available under its original filenames and operator identity. It has not been relabeled as this full-data visible-model run.

## What “a second” measures

The title's speed claim has a denominator. The paper measures downstream preparation and prediction after the model is already loaded, with CPU and GPU figures reported separately. It excludes one-time prior-fitting, just as its comparisons exclude baseline algorithm-development and meta-learning costs. The paper reports approximately .62 seconds on its GPU for the 32-view procedure; a CPU measurement, a cold download or another number of views is a different workload. Quoting .62 seconds beside a locally measured CPU quality result would splice together two systems. [Paper Table 1 and F.1/F.5](https://arxiv.org/html/2207.01848v6#A6).

In the local runner, the timer includes numeric preprocessing and model inference, but excludes data loading and checkpoint loading. It uses one CPU thread. The wrapper recomputes context states for each query chunk; it does not claim a context-cache speedup. Batching changes how often this context computation is repeated and how much memory is allocated. The rectangular attention still has C²+CQ score work per head per layer, while feature projections and feed-forward layers add substantial dense-matrix work. A score-count diagram is not a measured runtime model.

Original prior-fitting used 18,000 batches of 512 synthetic datasets, or 9,216,000 datasets, with sequences of 1,024 rows, a random context/query split, 12 layers and Adam with warm-up/cosine scheduling. The paper reports about 20 hours on eight RTX 2080 Ti GPUs and selects among three learning rates by final training loss. This course has not repeated that training. A pretrained checkpoint cannot be counted as newly reproduced prior fitting. The supplied closer track raises the local inference protocol to five splits and up to 32 views using the same visible implementation; it remains a subset reproduction attempt, with no cloud job launched by default.

## Where this architecture sits

**Versus FT-Transformer:** FT-Transformer makes feature tokens within one row and learns task-specific weights from that table. TabPFN v1 compresses the entire row into one token and communicates across labeled rows. A numeric feature's influence in v1 starts in its input projection coordinate; there is no explicit feature-token attention map to inspect. More context can change v1's answer while leaving all pretrained weights fixed.

**Versus TabTransformer:** TabTransformer contextualizes categorical embeddings within a row before its supervised prediction network. Its unlabeled-data pretraining and task-specific supervised fine-tuning are different from repeatedly learning to infer labels across synthetic supervised tasks. Calling both “pretrained Transformers” misses what the training target and inference input are.

**Versus SAINT:** SAINT's row attention also allows examples to interact, but its training recipe and batch semantics differ. A row-attention axis alone does not make a model a PFN. The defining extra ingredients are prior-generated shared tasks, a held-out query objective and the context/query information boundary carried into inference.

**Versus TabR:** TabR retrieves training examples as task-specific memory and learns a retrieval/prediction mechanism on the downstream table. TabPFN's full context is memory for a mechanism already learned elsewhere. Both can use labels from legitimate training rows, and both need the task's temporal or group boundary to decide which rows are legitimate. Neither can make a random split valid for predicting future customer behavior.

**Versus lesson 061 and the next lessons:** We keep the PFN objective but replace the fixed GP family and scalar density head with a rich tabular classification prior and class logits. The released v1 architecture removes query self edges, supports a fixed padded feature budget and combines inference views. Lesson 063 examines what the SCM generator assumes; lesson 064 changes the representation and scaling story. Learning these distinctions prepares you to diagnose which part of a foundation model needs to change when a database task exceeds its context, feature, class or prior assumptions.

## Exit and teach-back

The runnable notebook contains five substantive live tasks: context normalization; feature scaling/padding; rectangular attention; postnorm residual updates; and class-aligned logit aggregation. Each has an immediate diagnostic CHECK. The default experiment then runs the full pretrained model using your functions, compares one and four views, and shuffles context labels while keeping its query targets hidden until scoring. The EXIT saves your kernel identity, source parity checks, model/checkpoint identity, exact rows, probabilities and your written interpretation.

**Explain without looking:** A colleague says, “No optimizer runs on my table, so context labels cannot affect the answer.” Trace the first operation where that claim fails. Then explain why a second query cannot affect an existing query inside the Transformer, and how the measured preprocessing fallback invalidates a universal whole-wrapper claim. Finally distinguish code parity, pretrained inference and original paper-result reproduction in three sentences.

Bring the EXIT artifact and any failed CHECK back to the teacher. A notebook that runs after importing a hidden completed replacement is not evidence that you can reconstruct the mechanism. The next useful step is to predict and inspect one mismatch: change a live operator, observe the source-parity failure, and explain the pathway that changed before restoring it.
