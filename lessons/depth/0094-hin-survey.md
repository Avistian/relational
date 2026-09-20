## The big picture · classify assumptions before comparing results

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 094 / A MAP OF DESIGN CHOICES</p><p><strong>Bring forward:</strong> R-GCN, HAN and HGT made different choices about transformations and routes. <strong>Follow:</strong> schema → supplied structure → learned representation → prediction task → evidence contract. <strong>Carry forward:</strong> the bipartite and database lessons make the input assumptions concrete before the later controlled comparison.</p><p><a href="0091-r-gcn.html">L091: relations</a> · <a href="0092-meta-paths.html">L092: paths</a> · <a href="0093-hgt.html">L093: attention</a> · <a href="0060-broad-model-comparison.html">L060: comparison protocol</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:094-taxonomy]]

### Use the survey to generate a hypothesis

The primary survey is **Dong, Hu, Wang, Sun and Tang (2020)**, identified precisely in the reading contract below. Read its schema vocabulary before its representation families. A family label should help you predict a behavior: what inputs are required, which distinctions survive, and what happens to an unseen entity. It cannot rank models on its own.

Try the same question for three methods: “What must I provide to represent a new paper?” A node-ID lookup needs a new vector or an explicit inference procedure. An attributed encoder needs suitable features and the required neighbors. A featureless R-GCN that uses identity coordinates still needs a policy for new identities. Calling all three “graph embeddings” conceals this operational difference.

### Locate the lookup model in the same map

[[DEPTH_MAP:094-metapath2vec]]

The survey mentions metapath2vec, so its architecture deserves a visible handoff too. A designer provides a typed walk template. The walker produces node sequences whose steps respect that template. Context windows turn those sequences into center/context pairs. A skip-gram objective learns lookup vectors that make sampled contexts predictable. The resulting vectors feed a later classifier or clustering procedure. The [authors' project](https://ericdongyx.github.io/metapath2vec/m2v.html) separates walk generation, learning code, saved representations and downstream labels; follow that order when reading it.

This contrasts with HAN even though both use meta-paths. HAN constructs endpoint neighborhoods and computes representations from features with attention. metapath2vec optimizes stored vectors from walk contexts. The enhanced metapath2vec++ objective additionally distinguishes context types; do not transfer a ++ detail silently to the base method. This is a conceptual source map, not an added implementation or fresh reproduction lane.

### A concrete classification exercise

Consider a hypothetical method: choose PAP and PSP, calculate binary endpoint graphs, compute one mean keyword vector per route, concatenate the results, and fit logistic regression. Its route policy is hand-specified; its neighborhood aggregation is fixed; its classifier is learned; its inputs include attributes. Changing only the classifier to a neural head does not make route selection learned. Changing the endpoint graphs into walk counts changes the representation even if the classifier stays identical.

Now replace the mean with a learned GAT operation and add semantic fusion. You can explain why the result resembles HAN by identifying changed operations, rather than by spotting the word “attention.” That vocabulary will let you specify an ablation in L099 precisely.

### Count what the graph actually stores

Suppose a release stores 100 forward facts, their 100 reverse entries, and 12 hierarchy entries outside the listed fact families. “100 edges,” “200 directed entries,” and “212 stored entries” can all describe different legitimate counting conventions. None establishes that two releases contain the same facts. Match the convention, then inspect identities and provenance. The complete NN audit below applies this reasoning to actual released data without repairing a mismatch merely to reach a printed total.

<details><summary>Try first: are “meta-path method” and “GNN” mutually exclusive categories?</summary><p>No. One names supplied route structure and the other names a representation mechanism. HAN belongs to both descriptions. Use independent axes for route selection, learned object, attributes, supervision and prediction target.</p></details>

**Forward connection.** L095 preserves user–item identities that a same-type projection can discard. L096 preserves database roles and event multiplicity. Those are representation decisions made before fitting any encoder. Write one hypothesis linking a retained distinction to a target, plus one intervention that could falsify it. That is the useful output of the survey; a list of impressive model names is not enough.
