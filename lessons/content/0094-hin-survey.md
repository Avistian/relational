<div class="package-links"><strong>Lesson + lab package</strong><br><a href="../labs/html/0094-hin-survey.html">Read the executed lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/0094-hin-survey.ipynb">Run in Colab</a> · <a href="../labs/0094-hin-survey.ipynb">Download student notebook</a> · <a href="../labs/solutions/0094-hin-survey.ipynb">Teacher solution</a><br><a href="#lab">Go to exercises</a> · <a href="../reference/hin-taxonomy.html">Reference map</a> · <a href="../labs/l094-reproduction.md">Reproduction contract and commands</a></div>

## One tangible win

Build a defensible map of heterogeneous graph methods. Given a method description, identify **what structure it uses, how routes are chosen, what it learns, and what evidence could test it**. This is the organizing lesson between the three heterogeneous encoders and the coming database/recommendation lessons.

Our mission is to establish when relational structure adds predictive value. A taxonomy helps choose a testable hypothesis. It does not establish that more elaborate structure is better.

**Reading route:** first complete the cold questions; then trace the small graph; finally fill the taxonomy and audit a real released graph in the notebook. You can complete the conceptual core in one sitting and return for the data audit.

## Retrieve before reading

[[WARMUP]]

Write three short answers from memory before opening the model answers:

1. In L091, what chooses a message's weight matrix?
2. In L092, what does the designer choose before attention is learned?
3. In L093, why does adding relative time not guarantee leakage-free prediction?

<details><summary>Check after attempting</summary><p>R-GCN uses the relation type. HAN starts with selected meta-path neighbor graphs. HGT's time representation changes the computation; availability cutoffs and removal of target edges still have to be enforced by the data pipeline.</p></details>

## 1 · Repair the reading contract

The roadmap called this “Sun & Han 2020,” without an identifiable title. This package uses **Dong, Hu, Wang, Sun and Tang (2020), Heterogeneous Network Representation Learning**. Read §1 for schema vocabulary, §3.1–3.2 for the two representation families, and §4 for open questions and OAG statistics. Sun and Han's 2012 work is historical context, not the author list of this 2020 paper. [Primary survey](https://web.cs.ucla.edu/~yzsun/papers/2020_IJCAI_HIN_Survey.pdf).

> **Scope check.** This survey does not introduce a new predictive architecture or a table of its own model accuracy. Its concrete numeric target here is **Table 1's complete NN statistics row**. We audit the entire released NN graph and the arithmetic of all three printed rows. Retraining the preceding models belongs to their named experiment tracks, linked below. A statistics audit is not model-result reproduction.

## 2 · Start with the schema, then choose a question

> **In plain terms.** The schema tells you which kinds of things can connect. A route tells you what relationship you want a sequence of connections to express.

A **heterogeneous information network (HIN)** is a graph whose nodes or relations have multiple types. A node is one entity instance, such as Ada or paper P0. A node type is a category, such as author or paper. An edge relation gives the connection a meaning, such as “writes.” The **schema** is the graph of these types and permitted relations. A **meta-relation** is a source type, relation, target type triple: `(author, writes, paper)`.

The **meta-path** `author → paper → author` is a schema-level sequence. Ada → P1 → Bo is one concrete path instance. The reversed “written by” link must actually be available in the representation; a directed graph does not imply it automatically. These distinctions follow the survey's §1 definitions. [Survey §1](https://web.cs.ucla.edu/~yzsun/papers/2020_IJCAI_HIN_Survey.pdf).

**Worked example.** Four authors write four papers. Ada writes P0 and P1; Bo writes P1 and P2; Cy writes P3; Dee has no papers. P0 and P3 appear at venue V0, while P1 and P2 appear at V1.

[[ROUTE_FIG]]

**Step 1: store a relation.** Let `A_AP` be a 4 × 4 matrix: its author row and paper column contain 1 when that authorship exists, otherwise 0. Its transpose `A_PA` reverses that relation.

**Step 2: compose relations.** Matrix multiplication `A_AP @ A_PA` sums over intermediate papers. Entry `(Ada, Bo)` is 1 because they share P1. Entry `(Ada, Ada)` is 2 because Ada has two papers. The diagonal is not automatically an error or a self-loop added by a neural layer; it counts actual returning walks.

**Step 3: change the meaning.** Let `A_PV` have four paper rows and two venue columns. The product `A_AP @ A_PV @ A_PV.T @ A_AP.T` counts shared-venue walks. Ada–Cy changes from 0 to 1. They have no common paper, but both have a paper at V0. Ada–Bo changes from 1 to 2 because two paper-pair combinations pass through V1.

**Step 4: distinguish counts from reachability.** Replacing every positive entry by 1 asks whether a route exists. Keeping the integer asks how many instances exist. Those are different input features. A sparse implementation must preserve the chosen meaning; building these dense products on OAG would be impractical.

### A meta-graph is not simply a longer path

A **meta-graph** is a typed structural pattern that can branch and combine constraints. Here is one explicit teaching example: require both a shared-paper route and a shared-venue route between the same two authors. Define its Boolean result as `(paper_count > 0) AND (venue_count > 0)`. Ada–Cy fails this conjunction despite their shared venue. This example is a chosen pattern semantics; it is not a universal formula for every meta-graph method.

[[ROUTE_WIDGET]]

**Predict, intervene, explain.** Remove Bo's link to P1. Ada–Bo shared-paper count becomes 0; the shared-venue count becomes 1 because Bo still has P2 at V1. Dee's row remains zero. Explain why changing the route can create connectivity without adding an authorship edge.

## 3 · Use independent classification axes

[[PREDICTION]]

[[FAMILY_FIG]]

A useful taxonomy should survive a method that belongs to two categories. “Meta-path” describes route construction; “GNN” describes a learned representation mechanism. They are not competing labels.

| Method | Structure or route supplied | Learned object | Prediction handoff |
|---|---|---|---|
| Fixed route counts | Explicit typed path | Nothing in the counting operator | Counts become task features |
| metapath2vec | Typed walk template and sampled contexts | One embedding vector per node ID | Embeddings feed a downstream task |
| R-GCN · L091 | One-hop relation-labeled edges | Relation transforms, optionally shared bases | Repeated updates feed a task head |
| HAN · L092 | Chosen meta-path neighbor graphs | Neighbor attention and semantic fusion | Fused node representations feed a task head |
| HGT · L093 | Typed edges and relative time contexts | Node-type projections and relation attention/message transforms | Repeated updates feed a task head |

**Read the map’s symbols.** In R-GCN, `h_source` is a source-node vector and `Wᵣ` is the learned linear map for relation `r`; the sum combines relation-specific neighbor means. In HGT, a query (Q) represents what the receiver attends to, a key (K) represents how a source matches that query, and a value (V) is the payload to aggregate. A task head converts the resulting representation into predictions.

**Supervision is another axis.** A supervised objective uses task labels; an unsupervised objective uses another signal, such as node co-occurrences. Route choice alone determines neither. Likewise, node attributes are information supplied to an encoder, while learned node-ID vectors are parameters; they are not interchangeable assumptions.

**Lookup versus encoder.** A lookup table stores a learned vector for each known node ID. A message-passing encoder computes a representation from inputs and neighbors using shared parameters. For a newly arriving node, a lookup table needs an extra rule or optimization. An encoder can compute a new representation when the required features, types, and neighbors are available. [metapath2vec, author project](https://ericdongyx.github.io/metapath2vec/m2v.html); [R-GCN paper](https://arxiv.org/abs/1703.06103).

**Do not collapse route design into attention.** In HAN, the designer supplies the possible routes. Learning semantic weights chooses how to combine representations from those routes. It does not make an omitted route appear. In HGT, multiple layers compose one-hop typed updates, but that is not an exhaustive search over every possible relational query. [HAN §4](https://arxiv.org/abs/1903.07293); [HGT §3](https://arxiv.org/abs/2003.01332).

**Induction needs an input contract.** A GNN is not automatically an inductive experiment. If its only input is a learned embedding tied to training IDs, new IDs still require a policy. A transductive experiment may expose all graph nodes during fitting while hiding test labels. An inductive experiment withholds the target nodes or graphs from fitting. For forecasting, the time at which each feature and edge becomes available adds another constraint. See [L083 GraphSAGE](0083-graphsage.html) and [L093 HGT](0093-hgt.html).

### Bridge the three preceding lessons

R-GCN addresses the loss of relation meaning caused by a shared transform. HAN addresses the need to distinguish selected multi-hop meanings. HGT addresses typed attention and composition with a scalable sampling strategy. These are different design choices, not three rungs on an accuracy ladder. Revisit their full computation maps: [L091](0091-r-gcn.html), [L092](0092-meta-paths.html), [L093](0093-hgt.html).

## 4 · A taxonomy is not a leaderboard

**Worked evidence trap.** AIFB entity-classification accuracy, ACM embedding KNN F1, and CS paper-field ranking NDCG measure different tasks on different graphs. Sorting their numeric values would produce an ordering with no statistical interpretation. Repeated seeds cannot repair that mismatch. Accuracy is the fraction of correct class predictions. F1 combines precision and recall; KNN here is a k-nearest-neighbor classifier applied to embeddings. NDCG scores the ordering of relevant items with a rank discount and normalizes by an ideal ranking. These quantities have different denominators and prediction units.

An **evaluation protocol** specifies what the model could see, how it was chosen, and how predictions are scored. Compare data hashes, target task, split identities, feature access, target-edge removal, selection rule, compute/search budget, metric definition, and aggregation. A missing field is unknown, not agreement.

The notebook's `compare_protocols` returns `INCOMPARABLE` for a known conflict, `NOT_ESTABLISHED` when required information is missing, and `COMPARABLE` only when every required field is recorded and agrees. That last state is eligibility for a controlled comparison. It is not statistical significance, fairness of every implementation detail, or proof of reproduction.

**A fair next experiment.** Choose one graph, one prediction target, fixed availability rules, a declared split, and a validation-only selection budget. Include a simple baseline that tests whether the task is trivial. Train the candidate encoders with comparable opportunities and report paired outcomes. For a broad family claim, repeat across datasets and aggregate at the dataset level, as in [L055](0055-tabred-temporal-splits.html) and [L060](0060-broad-model-comparison.html).

## 5 · Reproduce the accounting before trusting the benchmark

**Held fixed:** the SHA-256-pinned complete NN archive and a source-visible read-only loader. **Measured:** every node count and every stored typed adjacency. **Varied:** the edge-count convention. There are no sampled nodes, random seeds, train/validation choices, or fitted models in this audit.

The released structure stores both original and `rev_` relations. It also stores field-to-field hierarchy links, which are absent from the table's five listed edge families. The audit counts original entries, verifies their reverse content, and reports both totals. An edge entry represents a source/target pair within one relation map; repeated events overwritten during preprocessing cannot be recovered from it.

[[MEASURED_TABLE]]

**Explain the discrepancy before fixing anything.** Matching all node counts is useful evidence, but it does not establish identical edges or features. An apparent printing error, different preprocessing, or another snapshot could explain some gaps. This audit does not identify which explanation is true. Keep the printed targets unchanged and preserve the mismatch.

[[ARITHMETIC_TABLE]]

> **Scope check.** The arithmetic check uses all three printed rows; it does not load CS or OAG. Fresh graph coverage is NN only. Historical snapshot identity and full-paper reproduction remain **NOT_ESTABLISHED**; historical comparison is **INCOMPARABLE**. The counting implementation passes when it accurately reports mismatches. It must not turn a mismatch into a successful reproduction label.

### Follow the existing model evidence without silently rerunning it

[[PRIOR_EVIDENCE]]

These are hash-pinned **prior author-run artifacts**, inspected in this lesson. They are not new training. The complete models and trainers remain visible in their originating notebooks. The [reproduction contract](../labs/l094-reproduction.md) gives their named commands and remaining gaps. A source operator check, a nearby score, an audit pass, and a complete historical reproduction are four different claims.

## 6 · Open questions become testable choices

**Route selection:** would a missing relation path remove necessary information, or merely reduce noise? Compare route interventions while holding prediction examples fixed.

**Multiple meanings:** are “works with” and “publishes at similar venues” useful for the same target? Do not assume one embedding distance should answer both questions.

**Scalability:** can the sampler reach the relationships that the target needs within the available budget? An elegant full-graph formula does not settle this.

**Availability and transfer:** can the same input contract serve a new entity or a new time period? Relative time encoding does not create historical feature snapshots. These operational questions turn the survey's research agenda into experiments relevant to our mission.

<a id="lab"></a>
## 7 · Lab: build and defend the map

The standalone notebook includes the explanation, portable diagrams, visible audit implementation, frozen evidence, and three live tasks. The default portable exercise uses synthetic matrices (Tier C) and recorded author evidence; an explicit full-graph switch runs the real released NN audit (Tier B). The executed teacher solution runs that full graph locally.

1. **TODO — compose a typed route.** Validate type continuity and compute path counts. CHECK tests multiplicity, an isolated author, and an invalid path.
2. **TODO — classify on two axes.** Represent HAN, HGT, R-GCN, and a lookup model without forcing disjoint categories. CHECK distinguishes route choice from encoder type.
3. **TODO — reject unsupported comparisons.** Make missing protocol fields fail closed. CHECK changes a metric while keeping all other fields fixed.

**EXIT artifact:** produce one row per method with route, encoder, learned parameters, supervision/feature assumptions, and source evidence. Add a 150–250 word defense: why HAN has two labels; why your proposed comparison is eligible; and what the NN mismatch prevents you from claiming. Do not submit only a passing notebook.

[[TEACHBACK]]

**Strict mastery status: `PENDING_WRITTEN_DEFENSE`.** Teacher execution prepares material; it is not evidence that you can retrieve or defend it. Tomorrow, redraw the map from memory before reopening it. Ask the agent follow-up questions and paste your EXIT defense for critique.

## What comes next

L095 uses a user–item graph to turn the taxonomy into a bipartite prediction problem. L096 maps database keys to typed graph relations. Carry forward the key distinction: choosing a schema, choosing a route, learning an encoder, and validating a prediction are separate decisions.
