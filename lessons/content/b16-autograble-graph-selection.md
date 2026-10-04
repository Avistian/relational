# B16 · AutoGrable: selecting or declining a graph

<p class="subtitle">Research bridge · elective · choose a construction before training its GNN.</p>

[B15](b15-parameter-free-encoders.html) asked what information an encoder can see. [B11](b11-supervised-relational-baselines.html) compared ways to process relational structure. Today we ask a question that comes first: **which structure should the model receive at all?** A schema supplies possible relationships; it does not guarantee that every relationship helps the task.

Your tangible win: take eight rows, score every candidate column subset, and explain why a valid search can return **no graph edges**. This gives your Year 6 proposal a stronger baseline: the relational model must earn its communication channels.

**Route:** 20–30 minutes to trace; 30–45 minutes for the lab; revisit after 1, 7 and 30 days. Prerequisites: binary labels, held-out validation, and message passing. Each is restated below. [Student notebook](../labs/b16-autograble-graph-selection.ipynb) · [Executed solution](../labs/html/b16-autograble-graph-selection.html) · [Printable reference](../reference/b16-autograble-graph-selection.html)

## Recall before reading

<div id="b16-warmup"></div>
<noscript><p>Recall: why must test labels stay out of model selection? Can two different graphs look identical to a message-passing model? What does an encoder's information boundary constrain?</p></noscript>

## 1 · A graph is a choice about access

A **row** represents an example. Its **label** is the outcome to predict: here either 0 or 1. A **candidate column** is a feature we permit the algorithm to turn into shared graph structure. The candidate list is fixed before looking at validation outcomes. Labels are never candidate features.

An **incidence graph** has row nodes and value nodes. Selecting column A creates one value node for each distinct A value. Each row connects to its own A value node. Two rows with A=0 now share a communication route. A **grable** is the paper's name for a graph constructed from a table. Selecting nothing yields the **trivial grable**: row nodes with no cross-row edges. A row-local predictor can still use each row's features. Abstention does not mean refusing to predict. [Paper §4](https://arxiv.org/html/2608.11431v1#S4)

**Message passing** updates a node using its current features and aggregated messages from its neighbors. Shared neighbors let a prediction depend on other rows. But they can also introduce noise, cost or unsupported distinctions. AutoGrable searches for useful columns before paying to train the downstream model.

{{ARCHITECTURE}}

The upper path is the load-bearing algorithm in this lab. With N training rows, m eligible columns and k selected columns, its inputs are N×m features and N labels. Each candidate produces one probability per validation row. The graph contains N row nodes, V distinct typed value nodes and Nk undirected incidence edges. A downstream model would map node features to hidden vectors and then to one output per row. **Table 1 studies column recovery only: no GNN is trained there or in our finite experiment.** The lower model path is shown to place selection in the whole system, not as evidence of training. [Paper §5.1](https://arxiv.org/html/2608.11431v1#S5.SS1)

## 2 · Partitions make the construction measurable

A **partition** divides rows into nonoverlapping blocks. For a selected subset S, put rows together exactly when they agree on every column in S. Write this as `r ~S r′` when `r[S] = r′[S]`. Empty S gives one block; a unique key K gives eight singleton blocks. A singleton is a block containing just one row.

Our eight training rows enumerate A/B pairs twice. A is 0 for rows 0–3 and 1 for rows 4–7; B is 0,0,1,1,0,0,1,1. K is unique. In the **signal world**, the label equals A. Validation and test repeat these A/B patterns with disjoint IDs and fresh K values. This controlled setup is deliberately constructed; it is not a sample of Adult.

The **block predictor** estimates P(label=1) as the proportion of training labels equal to 1 inside a block. An unseen validation projection has no training block, so it falls back to the overall training proportion. It fits counts, even though the paper calls the scoring stage “training-free”: that phrase means no trained graph model, not no use of training labels.

**Worked example.** For S={A}, the A=0 block has four zeros and the A=1 block four ones. The estimated probabilities are 0 and 1. For S={K}, every training label can be memorized, but every validation key is unseen. All validation probabilities become 0.5. Perfect training purity alone does not establish generalization. [Paper §2](https://arxiv.org/html/2608.11431v1#S2)

<div id="b16-predict"></div>

{{PARTITIONS}}

**Try it:** hold the labels and split fixed. Switch A → K → A+B, then increase the penalty. State which change affects prediction and which only fragments the training sample. The board always preserves the empty-set baseline. The course scorer uses binary **Brier loss**, `(p−y)²`: the squared gap between a predicted probability p and the actual binary label y. Smaller is better; 0 is perfect and probabilities of 0.5 give loss 0.25.

## 3 · Charge for fragile distinctions

Let `n_b` be the number of training rows in block b and `n` the total number of training rows. The **occupancy penalty** is

`Ω = (Σ_b √n_b) / n`.

The symbol Σ means “sum over the occupied blocks.” Dividing by n makes the scale comparable across partitions of the same training sample. With eight rows: one block gives √8/8≈0.354; two equal blocks give (2+2)/8=0.5; eight singletons give 8/8=1. Splitting a positive block increases the numerator because √a+√b≥√(a+b). Finer partitions therefore incur at least as much occupancy cost.

AutoGrable scores a candidate by `J(S) = validation risk + λΩ`. The nonnegative **penalty weight λ** states how much predictive improvement is needed to justify fragmentation. The course default is λ=0.5. In the signal world, empty S scores 0.25+0.5×0.354≈0.427; A scores 0+0.5×0.5=0.250; K scores 0.25+0.5×1=0.750. A earns its extra blocks; K does not. These are exact finite diagnostics, with displayed decimals rounded. [Paper Eq.1 and Appendix C](https://arxiv.org/html/2608.11431v1#S2)

The paper gives a bound for a **penalized-risk comparator**, assuming independent validation draws, a bounded loss and candidates fixed independently of validation. It does not guarantee that every selected graph improves a GNN. Adaptive search over a fixed candidate family is covered by its uniform bound, but greedy search can add an optimization gap. Frequency encodings derived from validation features require care before invoking that conditional argument; a theorem's assumptions must match the implemented pipeline. Our repeated finite grid illustrates arithmetic, not a random-sampling generalization theorem. [Appendix C.5–C.7](https://arxiv.org/html/2608.11431v1#A3.SS5)

## 4 · A good objective can have a bad search path

There are 2^m subsets of m candidate columns. For three columns, all eight are cheap to enumerate. With many columns, **greedy forward search** starts empty and accepts the best single-column addition only if J improves by more than a tolerance τ. **Greedy backward search** starts with all candidates and considers single-column removals. Neither is guaranteed to find the global minimum. A tolerance is a stopping threshold, not a confidence interval.

Now change the world: the label is **XOR**, equal to 1 exactly when A and B differ. Each column alone has balanced labels; together they determine the outcome. At λ=0.5, empty J≈0.427, either single column J=0.5, and A+B J≈0.354. The pair is better, but forward search cannot cross the worse single-column step. Backward search first removes K and reaches A+B. [Algorithm 1 and §3](https://arxiv.org/html/2608.11431v1#S3)

**Predict before changing the control:** does increasing λ to 1 rescue forward search, or make the pair too expensive?

{{SEARCH}}

An **objective failure** means the scoring rule itself prefers an unwanted construction. A **search failure** means the procedure misses a better value of its own score. At λ=0.5 the XOR forward result is a search failure. At λ=1, empty J≈0.604 beats A+B J≈0.707; abstention is now the exact objective's preference. Backward search still gets stuck at A+B: either intermediate singleton scores 0.750, worse than the pair. Both search directions can fail, even when they fail at different penalty settings. Neither result alone answers whether a separately trained GNN would help.

The **null world** alternates labels within each A/B pair. None of A, B or their combinations predicts these held-out labels better than the marginal. K still memorizes training rows but never repeats. With λ>0, empty is the best subset. Our implementation allows backward search to reach it. This is a finite example of justified abstention relative to the declared features and score; it is not a universal theorem that the task has no learnable rule.

## 5 · Check what the graph actually exposes

The **one-dimensional Weisfeiler–Leman procedure**, or **1-WL / colour refinement**, repeatedly replaces each node's colour by a signature of its old colour and the multiset of neighbor colours and edge types. A multiset records repeats. Colours are equality labels, not numerical features with distances. If two nodes retain the same colour, a standard deterministic message-passing model with those initial features cannot distinguish them.

The paper's Lemma 1 equates the projection partition with the stable row colours under two precise conditions: **all row features are erased**, and **value nodes expose their typed identities `(column, value)`**. The actual downstream graph keeps the unexpanded row features. Its rows may therefore be more distinguishable than the structural partition used for scoring. Structural blocks are also not connected components: with A+B selected, row (0,0) can share A with (0,1) and B with (1,0), connecting different blocks. [Paper §4, Lemma 1](https://arxiv.org/html/2608.11431v1#S4)

{{GRAPH}}

In this fixed balanced graph, typed A identities give two structural row blocks. Hide the identities and both four-row stars are symmetric: all eight rows have one colour. Keep row-local feature B and the typed graph distinguishes four groups. These interventions alter what the model can observe; adding parameters cannot repair an identity that is never exposed. Our lab constructs the edges and executes refinement rather than assigning colours from the desired answer.

## 6 · Reproduce the mechanism; audit the publication separately

{{RESULTS}}

The complete finite run covers 72 subset scores and 27 selections. Test probabilities are frozen after validation selection, and all 256 possible test-label vectors leave selection and probabilities unchanged in the intervention case. A separate scorer checks all 216 saved test predictions. The graph check covers 24 graphs and 1,536 pairwise same-block relations. These are author verification results, not your completion score.

The primary reproduction target is **B16-AUTOGRABLE-TABLE1**: five Adult-derived families × two signatures × two search directions × two λ values × ten seeds = **400 selection runs**. The targets are exact selected-column recovery and recall of the planted columns, not test accuracy. No fresh Table 1 runs have executed. The source release lacks the required synthetic-task driver, seed/split recipe and enough generation settings to reconstruct it honestly. [Table 1 and Appendix D](https://arxiv.org/html/2608.11431v1#S5.SS1)

The release at commit `1cfad48ae8d8362ad7eca10f9fe6db115004975e` also differs materially from the paper:

| Audit | Paper / declared interpretation | Measured pinned-release behavior |
|---|---|---|
| Backward abstention | Algorithm 1 can remove the last column | Stops with one; null fixture keeps B at J=0.750 although empty scores 0.677 using 0–1 loss |
| Frequency population | Eq.2 recodes T=train∪validation | Train codes use train counts; validation uses union counts: the same a receives 2 versus 4 |
| Value-node features | Lemma 1 exposes typed value identity | Builder emits zero vectors; balanced A graph has one structural row class |

The graph counterexample concerns the actual builder features with row features erased. The TabArena encoder also offers per-ID embeddings and random value features; both are off by default, but enabling them changes the information exposed. Builder features alone do not certify every downstream configuration.

The 0–1 source-audit scores above differ from the course Brier scores because the loss differs. We matched **144 scoring-primitive comparisons** using aligned 0–1 and clipped log losses. This narrow parity check does not resolve the three discrepancies or establish whole-source parity. In particular, our backward-to-empty implementation follows the paper and deliberately differs from the pinned release. [Pinned source](https://github.com/TamaraCucumides/autoGrable/tree/1cfad48ae8d8362ad7eca10f9fe6db115004975e)

**Selected paper verdict: `INCOMPLETE_SOURCE_PROTOCOL_GATE`; execution `NOT_RUN`.** The full source packet, discrepancies and guarded operator ship with the notebook. It authenticates sources and refuses a paper run until an actual experiment driver/configuration is recovered. Inventing a driver with convenient defaults would make a new course experiment. The complete paper's other benchmarks are `NOT_RUN`. Cloud spend is $0. [Reproduction contract](../labs/b16-reproduction.md) · [Source audit](../labs/evidence/b16/source-audit.json) · [Source manifest](../labs/sources/b16/manifest.json)

## Lab · implement the decision

Use the [portable notebook](../labs/b16-autograble-graph-selection.ipynb). Three live tasks implement training-only block probabilities with unseen fallback, the occupancy penalty, and exhaustive/greedy selection including abstention. The full subset scorer, fixture generator, typed graph constructor and colour-refinement algorithm are visible. Immediate checks distinguish bad arithmetic, invalid information access and a genuine local-search trap.

Before opening the [executed solution](../labs/html/b16-autograble-graph-selection.html), predict the selected set in signal/XOR/null at λ=0.5. Run the complete grid, then explain why a pure key block loses on unseen keys. The notebook embeds figures and the source packet; Python's standard library suffices for the learning lane. A local notebook pass does not establish live Colab execution.

## EXIT · defend your construction

<div id="b16-teachback"></div>
<noscript><p>Write a short defense: why does forward search miss A+B for XOR? When does empty become the exact optimum? Which graph features does Lemma 1 require? What remains unrun?</p></noscript>

Your answer must separate validation selection from test evaluation, objective failure from search failure, and projection blocks from full-feature graph distinguishability. Learner status remains **PENDING_WRITTEN_DEFENSE** until you submit your work. Ask follow-up questions here about any formula, code path or source discrepancy.

**Spacing:** in 1 day, derive Ω for one block versus eight singletons from memory. In 7 days, reconstruct the XOR search trap. In 30 days, audit the feature assumptions of a graph in your own project. [B17's plan](../plan/year-5-6-bridge.md#b17) then asks when representations can be reused across tasks.

**Primary reading:** [AutoGrable v1](https://arxiv.org/html/2608.11431v1), §§2–4, Table 1 and Appendices C–D. Read Algorithm 1 beside the pinned `selection.py`; explain the one-column stopping difference before treating a library call as a reproduction.
