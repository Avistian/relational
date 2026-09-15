## What you will be able to build

By the end of this lesson you will have a working copy of the historical **TabPFN v1** classifier whose every step — normalization, attention, residual updates, and the way it combines several predictions — you can read and complete yourself.

Start from something familiar. In lessons 042–054, a new table with a few hundred labeled rows drove an optimizer: you initialized a network, computed a supervised loss, updated the weights, picked a checkpoint, and then predicted. Training happened *on your table*.

TabPFN moves almost all of that learning to an earlier phase, done once by its authors. Its weights were trained on millions of small synthetic classification tasks. By the time the model reaches your table, it never trains again. Instead, your labeled rows are fed in **as input to a frozen network**, alongside the rows you want to predict. Change the labels you feed in and the prediction changes — with no weight ever updated.

> **In plain terms.** A normal model learns your task by adjusting weights. TabPFN already learned *how to learn small tabular tasks* in general, so it reads your labeled rows the way a trained reader reads a new example: recognizing the pattern and answering, without going back to school.

This lesson's tangible outcome is a predictor you can inspect end to end. The lab copies the *actual released weights* into your visible implementation and measures real predictions on real data.

> **Scope check.** What you build is **pretrained inference running through your own implementation**. It is not fresh pretraining, and it is not a reproduction of the paper's aggregate benchmark. Copying weights into correctly-shaped attention is also not enough on its own: randomly initialized attention with the right shapes does not yet compute a useful prediction algorithm — the learned weights are what make it work.

**Why this is on the syllabus.** The relational mission is to show whether database *structure* (entities, joins, time) adds value over a strong flat-table baseline. TabPFN is an unusually strong and unusually *different* baseline: it transfers an inference procedure learned across many tasks. But its row tokens still describe one flat feature table. They do not, by themselves, create entity relationships, temporal guarantees, or database joins — those remain properties of how *you* set up the prediction task.

**Retrieve first, without scrolling.** In an ordinary MLP, does the label enter the *input* or the *loss*? In lesson 061, why did the Gaussian-process sampler have to draw the context targets and the query targets from one shared function? Could a query row attend to its own feature token without seeing its own target? Hold onto that last answer: v1 changes that edge for a *computational* reason, not because the edge necessarily leaked a label.

## The idea behind a prior-fitted network

This section builds up, one term at a time, to a single equation that says what TabPFN is trained to approximate. Read it slowly; each new word gets its own sentence.

**A prior.** A prior is a distribution over the possible *data-generating mechanisms* you might face, written down *before* you see the current table. Call one mechanism φ (phi). One draw of φ fixes a whole recipe: a graph, its weights, its noise levels, which nodes you get to observe, and the thresholds that turn numbers into class labels. Given that recipe φ, you can draw an entire dataset D.

**One mechanism per dataset, not per row.** A single φ generates a whole table. If you redrew φ for every row, the rows would share no structure, and the labeled rows could tell you nothing about the unlabeled ones. Shared structure across the table is exactly what makes context useful.

**The posterior predictive distribution (PPD).** This is the quantity we ultimately want. Given the labeled context Dᶜ, the best possible probability for a new label y\* averages the predictions of *every* mechanism that could have produced the context, each weighted by how plausible it is:

`p(y* | x*, Dᶜ) = ∫ p(y* | x*, φ) p(φ | x*, Dᶜ) dφ`.

Read the integral as "average over all allowed mechanisms φ, weighting each by its posterior plausibility given what we have seen." (In a joint model, the query's own features x\* also help say which φ is plausible; fixed-design notation often hides this extra conditioning.)

**Why we approximate it.** Computing that integral exactly is expensive. A **prior-data fitted network** (PFN) instead *learns* an approximation `qθ(y* | x*, Dᶜ)`, where θ are the Transformer's weights. Training is a loop: sample a task from the prior, hide some of its labels, ask the network to predict them, and adjust θ to raise the probability it assigned to the true labels. A **query** is simply a row whose label the network must predict. The training loss on one label y is the **cross-entropy** `−log qθ(y)`, measured in natural-log units called **nats**.

**Where the loop lands.** For a fixed context and query, the expected cross-entropy splits into two pieces: the entropy of the true conditional distribution (fixed, independent of θ) plus `KL(p || qθ)`, a distance-like quantity that is zero only when qθ matches the truth exactly. Lowering the loss can only shrink the KL term, so the best possible network reproduces the prior's PPD.

> **Scope check.** That "best possible" statement is about the chosen prior and the *ideal* optimum. Finite training, finite model capacity, and any mismatch between the synthetic prior and real data all leave real error. In lesson 061 those gaps were measurable because the Gaussian-process oracle was analytic; the full TabPFN prior has no comparably cheap exact oracle. [Paper §§2–4 and Algorithm 1](https://arxiv.org/html/2207.01848v6#S2).

### What v1's prior actually samples

v1's prior mixes two families of mechanisms with equal probability.

- A **Bayesian neural network (BNN) prior** samples an architecture and random weights, then pushes input features through that random network to produce targets.
- A **structural causal model (SCM)** specifies a directed acyclic graph plus one noisy assignment per node. A *directed edge* means one node's value feeds another node's generating function. *Acyclic* means no directed loops, so the nodes can be generated in order. The SCM draws feature nodes and a target node from a sparse layered graph; the target may be a *cause* or an *effect* of the observed features.

Both families are used only *during pretraining*. The Transformer does not sample a new graph for each real prediction; pretraining has already baked the consequences of these sampled mechanisms into its fixed weights.

The prior also has distributions over depth, hidden size, weight scale, noise, sparsity, and activation function. "Simple," in the paper's sense, means these sampled families tend to favor graphs with relatively few nodes and parameters — it is not a claim that every simple-looking real task is easy.

**How labels are made (worked example).** The generator first produces continuous targets, then samples thresholds, assigns an interval ID to each row, and finally permutes those IDs. For continuous targets `[-.4, .2, .8]` and thresholds `[-.1, .5]`, the interval IDs are `[0, 1, 2]`. A sampled relabeling `[2, 0, 1]` turns them into `[2, 0, 1]`. Permuting the IDs is what lets the same machinery produce multiclass tasks and class imbalance without treating the class number as an ordered quantity. The thresholds come from the *generated* task's own values.

> **Scope check.** An SCM-based prior does not recover a causal graph from your table. The paper predicts associations directly, without producing a graph. Observational prediction, an intervention (forcing a treatment), and a counterfactual for one person are three different targets; high classification accuracy establishes only the first unless your task and evidence establish more. Lesson 063 implements and interrogates this prior; here we implement the predictor that consumes its consequences. [Paper §§4.1–4.5; Appendices C and E.4](https://arxiv.org/html/2207.01848v6#S4).

## Model architecture

**The symbols, once.** Let C be the number of context (labeled) rows, Q the number of queries, F the number of active numeric features, B the number of independent tasks processed together, d the hidden width, and H the number of attention heads.

**The one idea that shapes everything.** A **token** is the vector that attention treats as one element of its sequence. In v1, one token is a **whole row** — not a single feature. Everything downstream follows from that choice: attention mixes information *between rows*, not between columns.

The released model uses d = 512, H = 4, 12 Transformer blocks, feed-forward width 1024, and at most 10 output classes. So each head works in 512 / 4 = 128 coordinates. The feature input has a fixed maximum width of 100 columns.

**From a row to a token.** Before tokenization, context-only transformations standardize the features, optionally apply a power transform, and compress outliers (all detailed in the next section). After scaling and zero-padding, the batch of rows has shape `B × (C+Q) × 100`. A linear **feature encoder** maps each 100-wide row to 512 coordinates. A separate linear **label encoder** maps a single context class ID to its own 512 coordinates. A context token is the sum of the two:

`hᵢ⁰ = Wₓ xᵢ + bₓ + Wᵧ yᵢ + bᵧ`, for `i < C`.

A query token uses the feature encoder only:

`hⱼ⁰ = Wₓ xⱼ + bₓ`.

There is no learned "unknown label" vector on this released path. A context row whose label is class 0 still carries the label encoder's bias term; a query row never passes through the label encoder at all. No positional encoding is used, so row order carries no meaning. The model's forward signature takes features `x` and context labels `y`; it never needs a placeholder target for a query.

<!--figure:architecture-->

### Inside one block

Every block starts by projecting each current row state into three vectors: a **query vector** (what information this row is looking for), a **key** (how well another row matches that request), and a **value** (the information a matching row contributes). Watch the word "query": *all* rows produce attention query vectors, but only the *unlabeled* rows are prediction queries. Every receiving row may read the context rows' keys and values. No receiving row may read any query row's keys or values.

The 12 blocks perform the same operation with different learned weights. Each block has an attention sublayer and a feed-forward sublayer, and each sublayer does a residual addition followed by LayerNorm. **LayerNorm** normalizes across the hidden coordinates *within a single row* and then applies a learned per-coordinate scale and offset; it never pools statistics across the query rows. The feed-forward sublayer is Linear 512→1024, then **GELU** (a smooth nonlinear activation), then Linear 1024→512. After block 12, the query states pass through Linear 512→1024, GELU, Linear 1024→10, and only the first K logits are used for a task with K classes.

The copied checkpoint holds 25,821,706 learned parameters. Pretraining set them using held-out synthetic labels; real-table inference keeps them fixed and changes only the input features and context labels.

> **Scope check.** The checkpoint also stores training configuration, but serialized settings such as local batch size do not prove that the metadata alone reconstructs the paper's distributed training run. [Paper E.1–E.3](https://arxiv.org/html/2207.01848v6#A5); released [`TransformerModel.forward`, `TransformerEncoderLayer.forward`](https://github.com/PriorLabs/TabPFN/tree/v1.0.0/tabpfn).

## How information moves between rows

This is the mechanical heart of the model. We will state the attention formula, count exactly which connections exist, then walk one tiny numeric example.

**The formula.** For one head, let `U` hold the query vectors of all C+Q receivers, and let `Kᶜ, Vᶜ` hold the keys and values of the C context rows only. The attention output is

`A = softmax(U (Kᶜ)ᵀ / √128) Vᶜ`.

The softmax runs over the **sending context-row dimension**, so each receiver's weights over the context rows sum to one. Dividing by √128 (the square root of the head width) keeps the dot products from growing too large as the dimension grows. Splitting into four heads lets each head combine its own slice of the value coordinates; the heads are concatenated and passed through an output projection before the residual addition.

**Counting the connections.** A score matrix of shape `(C+Q) × C` has exactly `C² + CQ` entries per head. The lesson-061 PFN also let each query attend to its *own* feature token; v1 removes those query self-edges (Appendix E.2). For C = 3, Q = 2, the old conceptual 5×5 mask allowed 17 entries, while v1's mask allows 15. The efficient implementation never builds the query-column block, so it stores 15 score entries instead of a full 25-entry matrix. Removing self-edges does not remove the `C²` cost of context attending to context.

<!--figure:attention-->

**Worked example (one head, width 2).** Take a receiver query vector `[1, 0]`, two context keys `[0, 0]` and `[√2, 0]`, and two context values `[1, 0]` and `[0, 2]`. The scaled scores are `[0, 1]`. Softmax gives approximately `[.268941, .731059]`. The weighted value is therefore `[.268941, 1.462117]`. (These are synthetic projected vectors, not activations pulled from the checkpoint.) In the figure, changing an *unrelated* query's value leaves this output unchanged, because that other query contributes no sending column.

### Why two queries stay independent

If a query cannot read itself, how can the model tell two query rows apart? Two ways: its own features still determine its query vector, and the residual connection carries its own state forward. With attention output `a`, a block computes

`u = LayerNorm(h + a)` and `h′ = LayerNorm(u + W₂ GELU(W₁u + b₁) + b₂)`.

Removing self-attention removes an *edge*, not the *row*. (Moving LayerNorm to *before* these additions would be a different computation; copying pretrained weights into that variant is invalid even when every shape matches.)

The independence holds layer by layer. At layer 1 the context states depend only on context, and by induction that stays true at every layer. Each query then combines those context states with its own residual path, so a second query can never change the first through attention, through the row-wise feed-forward network, or through LayerNorm.

> **Scope check.** Independence *inside the Transformer* is not the same as independence of the *whole pipeline*. A preprocessing transform fitted on all query rows would create a dependency *before* tokenization. Even context-only fitting is not enough if transforming a new query triggers a fallback that rewrites already-transformed rows — exactly the failure documented in the next section. Dropout must also be off for a deterministic comparison.

**Predict before you check.** Permute the context rows and their labels *together*: the keys, values, and attention weights permute with them, so the weighted sum is preserved up to floating-point error. Now swap two feature columns without changing the encoder: there is no matching guarantee, because each input coordinate has its own encoder weights. Permuting rows, rotating feature indices, and rotating the geometric feature space are three genuinely different operations.

## Preprocessing is part of the predictor

Pretrained weights expect inputs in a particular distribution, so "I copied the Transformer" is not enough if you change the feature scaling or the label mapping. Every scaling rule below is fitted on context rows only, so a query can never move the numbers the context sees.

**Standardization.** Each view starts from context-only sample means and standard deviations. The denominator is the sample standard deviation (divisor C − 1) plus 10⁻⁶. Standardized values are clipped to [−100, 100]. Any feature that is constant across the context is dropped, and query rows never decide which columns survive.

**Worked example.** For a one-feature context `[1, 3]`, the mean is 2 and the sample standard deviation is √2. A query value of 101 becomes about 70.7106 after standardization; a much larger query may hit the clamp. Crucially, feeding `[1, 3, 101]` into the moment estimate would change the context's own standardized values and make the prediction depend on which other queries happened to arrive in the same batch.

**Power transform.** The optional Yeo–Johnson transform estimates one transformation per feature *on the context rows*, then applies it to both context and query. It reduces skew and handles negative as well as positive values, and its parameter never comes from the query table. The released wrapper keeps a column unchanged if fitting or transforming it raises an exception. These are *view choices*, not gradient updates to the Transformer.

> **Scope check — a measured failure of whole-wrapper isolation.** The historical wrapper wraps *both* the fit and the transform in one exception handler. An extreme extra query can make the transform overflow, and the fallback then keeps the *untransformed* column for the **entire** combined table — context and original query included. Context-only fitting did not prevent this control-flow dependency. Concretely: a 12-row, one-feature context with values around 10⁻⁹ fits a Yeo–Johnson exponent of about −219.54; adding a query of −100 triggers an `expm1` overflow in the recorded environment. The original query's class-1 probability then shifts from .382355 to .484686 with four views, and the released reference agrees within 1.4×10⁻⁶; the one-view, no-transform control shows zero difference. This is a real wrapper behavior, not an attention-mask bug and not something silently patched. The notebook reproduces it with your own predictor. The exact trigger depends on the recorded library versions, but the lesson is general: to prove pipeline independence, you must prove that preprocessing stays fixed **including its failure path**. [Saved counterexample](../labs/_query_fallback_l062_results.json).

**Outlier softening.** The wrapper estimates the context mean and standard deviation, keeps the context values within four standard deviations, re-estimates the mean and standard deviation on those kept values, and uses the resulting bounds for logarithmic tail compression. It compresses extremes rather than deleting query rows or clipping everything to ±4. Missing values pass through these moment calculations untouched until the feature encoder replaces them with zero; that zero carries no separate missingness flag, which matches the paper's light treatment of missing data.

<!--figure:preprocessing-->

**Padding to a fixed width (worked example).** With k surviving features and a maximum of K = 100, the wrapper multiplies the values by K / k and appends K − k zero columns, so one fixed linear layer can accept tables with different feature counts. For a miniature K = 6, `[1, −2]` becomes `[3, −6, 0, 0, 0, 0]`; the real checkpoint uses K = 100. A √(K / k) variant exists in the source but is **not active** in this checkpoint. Multiplying by K / k does not guarantee equal variance for every covariance structure — it is the specific magnitude convention the weights were pretrained with.

> **Scope check.** The supported lesson path covers numeric columns, context-fitted transforms, at most 100 columns, and 2–10 contiguous class IDs. It excludes ranking transforms, feature subsampling beyond 100, explicit categorical handling, and query-fitted normalization. The official source has these optional paths; passing our supported-path tests does not certify them. All-constant context features are rejected outright rather than silently dividing by a zero feature count.

## Combining several views

The released predictor can build several **views** of the same table and average them. One view rotates the feature-column indices, rotates the class IDs, and picks either the raw-standardized or the power-transformed features.

**Feature rotation is a reordering, not a matrix.** A column-index rotation by two turns `[x₁, x₂, x₃, x₄]` into `[x₃, x₄, x₁, x₂]`. It does not multiply the features by a dense orthogonal matrix. With F columns and K classes there are at most 2FK rotation/transform combinations, so asking for more views than that cannot create new ones.

**Class rotation, and undoing it.** A class rotation is `y′ = (y + s) mod K`. So the coordinate the rotated model predicts for original class j sits at position `(j + s) mod K`. You must reverse this mapping *before* averaging views, or one view's "class 0" would be averaged against a different event in another view. The feature and label transforms apply identically to context and query, and only context labels are ever revealed.

<!--figure:ensemble-->

**Worked example.** View A has logits `[3, 0, 0]` with shift 0. View B has logits `[1, 0, 0]` with shift 1. Rotate view B's answers back to `[0, 0, 1]`, then average to `[1.5, 0, .5]`. With an illustrative temperature of 1, softmax gives about `[.6285, .1402, .2312]`. Averaging each view's *probabilities* instead would give a different distribution. The historical 0.1.11 wrapper averages the *aligned logits*, divides by a temperature of .8, and then applies softmax. A temperature below one sharpens a fixed logit vector without changing which coordinate is largest.

> **Scope check.** This ordering is read from `transformer_predict`, not inferred from the paper's generic word "ensemble." The default alternates the none/power transform within a sampled pair of rotations, so a single requested view uses the *none* transform rather than a random 50/50 choice. The random seed picks view rotations; it never retrains the checkpoint. Constant-feature removal happens after the source draws feature shifts from the original column count, and the lesson keeps that slightly awkward slicing so its wrapper matches the released predictor exactly. A finite set of cyclic rotations is not all possible permutations, so keep the exact context-row permutation guarantee separate from these empirical view rotations.

## Running the historical model

The reference is the `tabpfn==0.1.11` wheel (identified by a SHA-256 digest) and the released `prior_diff_real_checkpoint_n_0_epoch_42.cpkt` checkpoint (SHA-256 beginning `3c9aadae`); a digest names exact bytes. The lab implements the inference mechanism independently, copies every model weight with a strict key/shape check (excluding only the criterion's class-weight buffer), and imports the released package *only* for validation. You can read the complete model and inference wrapper in the notebook.

**Validation has three layers.** Small exact fixtures expose the intended arithmetic. Full copied-weight forward and gradient comparisons establish parity with the named source computation. Complete wrapper comparisons then cover binary and multiclass data, a constant feature, missing entries, and an extreme query. Context permutation, context-label sensitivity, and query batching probe complementary properties. A predictor that returned the same constant distribution for everything could still pass a query-isolation check, which is why isolation alone is weak evidence.

The executed teacher run completes your five live functions and passes them into a real inference experiment, so the functions are load-bearing rather than decorative. Call counters confirm that preprocessing, attention, residual updates, and ensemble aggregation all run downstream. The recorded kernel identity captures the visible functions, methods, defaults, configs, and dependency versions; repository source hashes are reference identity and do not certify an edited notebook kernel. A mutation CHECK alters a helper, method, default, and preset and requires their identities to change. No checkpoint resume is offered.

## What the experiment measured

**Predict before the results.** Will four views beat one view on *every* split? Does feeding correct context labels beat feeding shuffled ones? Write down the direction you expect and one reason it might fail on a single small test set. The first intervention changes a fixed inference recipe; the second removes most of the genuine feature–label relationship while keeping the label counts. Neither is new pretraining.

The author panel uses diabetes (OpenML 37), blood transfusion (1464), and Wisconsin diagnostic breast cancer (the UCI dataset available as sklearn `load_breast_cancer`, OpenML 1510 in the paper). All rows are kept. Each dataset gets three stratified 50/50 splits with seeds 0, 1, 2, and both view budgets share the exact rows and labels, so one and four views of the same frozen checkpoint are compared on identical data. **AUC** (area under the ROC curve) measures how well positives are ranked above negatives; binary **log loss** scores the probabilities assigned to the true labels, lower being better. Both are recorded with individual predictions and view logits, the raw features and labels are hashed, and row indices are kept.

<!--figure:results-->

Read the points *within* each dataset first. The means and sample standard deviations summarize the three split/seed combinations; they describe variation over overlapping resamples of one dataset, so they are not independent-dataset confidence intervals. No Friedman test or critical-difference claim fits a two-budget mechanism comparison, three datasets cannot recover the paper's full model ranking, and repeated splits do not create new datasets.

> **Scope check — why the local numbers are INCOMPARABLE to the paper.** The paper's main result evaluates 18 numeric, no-missing datasets drawn from a 30-dataset OpenML-CC18 subset (at most 2,000 rows, 100 features, 10 classes), using five 50/50 splits and normally 32 views, with baselines tuned under declared budgets and their own preprocessing. Our local panel matches the dataset identities, the full-row small-data regime, the 50/50 split proportion, and the pretrained model family. It differs in dataset count, view budget, number of splits, exact split construction and row order, and the absence of tuned competing baselines; WDBC comes from sklearn's UCI copy rather than a proven byte-and-order match to the paper's OpenML version. So every local score is **INCOMPARABLE to aggregate paper-result reproduction**, even when it lands close to an individual published number. [Paper §5.2, Tables 1–2, F.3](https://arxiv.org/html/2207.01848v6#S5.SS2). For completeness: Appendix F.3 builds 150 meta-validation datasets and Appendix B.5 drops one for lack of valid splits, leaving 149; those datasets informed method development and are not untouched meta-test data. The paper's prior ablation uses smaller budgets to compare BNN, SCM, and their mixture, which does not isolate a particular encoder or prove causal priors dominate; our pretrained ablations cannot re-estimate that comparison. The older local three-task / 180-context result still exists under its original filenames and operator identity and has not been relabeled as this full-data run.

### What "a second" actually times

The title's speed claim has a denominator. The paper times downstream preparation and prediction *after* the model is loaded, reporting CPU and GPU separately and excluding one-time prior-fitting — just as its baselines' development and meta-learning costs are excluded. It reports about .62 seconds on its GPU for the 32-view procedure. A CPU measurement, a cold download, or a different view count is a different workload, so quoting .62 seconds beside a locally measured CPU quality result would splice two systems together. [Paper Table 1 and F.1/F.5](https://arxiv.org/html/2207.01848v6#A6).

In our runner, the timer includes numeric preprocessing and model inference but excludes data and checkpoint loading, and it uses one CPU thread. The wrapper recomputes the context states for each query chunk (no context cache), so batching changes how often that context work repeats and how much memory is used. Rectangular attention still costs `C² + CQ` score work per head per layer, and the feature projections and feed-forward layers add substantial dense-matrix work; a score-count diagram is not a runtime model.

> **Scope check.** Original prior-fitting used 18,000 batches of 512 synthetic datasets (9,216,000 datasets) with sequences of 1,024 rows, a random context/query split, 12 layers, and Adam with warm-up/cosine scheduling — about 20 hours on eight RTX 2080 Ti GPUs, selecting among three learning rates by final training loss. This course has not repeated that training, so the checkpoint cannot be counted as newly reproduced prior-fitting. The supplied "closer" track raises the local *inference* protocol to five splits and up to 32 views with the same visible implementation; it is still a subset reproduction attempt, with no cloud job launched by default.

## Where this architecture sits

- **Versus FT-Transformer.** FT-Transformer makes one token *per feature* within a row and learns task-specific weights on your table. TabPFN v1 compresses the whole row into one token and communicates *across* labeled rows. A feature's influence in v1 begins in its input-projection coordinate; there is no per-feature attention map to inspect, and adding context can change v1's answer while every weight stays fixed.
- **Versus TabTransformer.** TabTransformer contextualizes categorical embeddings within a row before a supervised head, using unlabeled-data pretraining and task-specific fine-tuning. That is a different target from repeatedly learning to infer labels across synthetic supervised tasks; calling both "pretrained Transformers" hides what the training target and inference input are.
- **Versus SAINT.** SAINT's row attention also lets examples interact, but its training recipe and batch semantics differ. A row-attention axis alone does not make a PFN; the defining ingredients are prior-generated shared tasks, a held-out query objective, and the context/query information boundary carried into inference.
- **Versus TabR.** TabR retrieves training examples as task-specific memory and learns its retrieval and prediction mechanism on your table. TabPFN's whole context is memory for a mechanism learned elsewhere. Both use labels only from legitimate training rows, and both need the task's temporal or group boundary to decide which rows are legitimate; neither makes a random split valid for predicting future customer behavior.
- **Versus lesson 061 and what comes next.** We keep the PFN objective but swap the fixed GP family and scalar density head for a rich tabular classification prior and class logits. v1 removes query self-edges, uses a fixed padded feature budget, and combines inference views. Lesson 063 examines what the SCM generator assumes; lesson 064 changes the representation and scaling story. Knowing these distinctions is what lets you diagnose *which* part of a foundation model must change when a database task outgrows its context, feature, class, or prior assumptions.

## Exit and teach-back

The notebook has five substantive live tasks: context normalization; feature scaling and padding; rectangular attention; postnorm residual updates; and class-aligned logit aggregation. Each has an immediate diagnostic CHECK. The default experiment then runs the full pretrained model through *your* functions, compares one and four views, and shuffles context labels while keeping the query targets hidden until scoring. The EXIT saves your kernel identity, source-parity checks, model/checkpoint identity, exact rows, probabilities, and your written interpretation.

**Explain without looking.** A colleague says, "No optimizer runs on my table, so the context labels cannot affect the answer." Trace the first operation where that claim fails. Then explain why a second query cannot affect an existing query inside the Transformer, and how the measured preprocessing fallback breaks a universal whole-wrapper independence claim. Finally, distinguish code parity, pretrained inference, and original paper-result reproduction in three sentences.

Bring the EXIT artifact and any failed CHECK back to the teacher. A notebook that only runs after importing a hidden finished model is not evidence that you can rebuild the mechanism. A good next step is to change one live operator, watch the source-parity check fail, explain exactly which pathway changed, and then restore it.
