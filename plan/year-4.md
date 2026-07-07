# Year 4 — Relational Deep Learning · Lesson Decomposition

**Year goal:** master RDL end to end — construct the relational entity graph (REG), run and reproduce
RelBench baselines, reach near-SOTA with next-gen architectures, and **fairly beat manual feature
engineering** on ≥3 tasks with honest failure cases. This is where the thesis is first tested with real
numbers. Labs are Tier B (RelBench via PyTorch Frame + PyG) on the `relkit` harness, now extended with
RelBench loaders and the REG builder.

The `CURRICULUM.md` range `151–154 · Multi-task RelBench portfolio` is decomposed into four lessons below.

---

## Q1 · RDL foundations (121–130)

**Papers (chronological):** Cvitkovic 2019 `2002.02046` · Fey et al. 2024 (RDL, PMLR v235) ★ · RelBench
beta `2312.04615` · Robinson et al. 2024 RelBench v1 `2407.20060` ★ · Hu 2024 PyTorch Frame `2404.00776`
★ · Peleška & Šír 2026 Universal Row Encoder `2606.21434` ★.

### 121 · History of relational ML — *Cvitkovic 2019, `2002.02046`*
- **Skill** — draw the prior-art map from ILP / propositionalization / AutoFE to GNN-on-RDB, and state what
  each era got wrong.
- **Teach** — classical relational learning, Deep Feature Synthesis / Featuretools, Cvitkovic's GNN-on-RDB
  result (beat AutoFE on 2/3 datasets), the gaps (no benchmark, no temporal standard).
- **Lab** — Tier C · crucial fragment: build the timeline map + locate the thesis on it. Deliverable: the
  prior-art map.
- **Viz** — reuse `arch-family-viz.js` (relational branch, historical).
- **Bridge** — callback Y3 L118; grounds the thesis historically; forward to L122 (Fey's modern REG).

### 122 · REG construction — *Fey 2024 §3, ★*
- **Skill** — construct a relational entity graph from a database schema: nodes = rows, edges = PK/FK links,
  node features from row encoders.
- **Teach** — the REG definition, node/edge typing from schema, the mapping rules (callback Y3 L096/L117);
  building it in PyG `HeteroData`.
- **Lab** — Tier B · crucial fragment (paper-repro): build a RelBench database's REG from schema.
  Deliverable: the constructed `HeteroData` REG + node/edge-type counts.
- **Viz** — reuse `hetero-graph-viz.js` + `rdl-stack-viz.js`.
- **Bridge** — the central construction of the year; callback Y3 L117; forward to L123 (add time).

### 123 · Temporal heterogeneous graphs — *Fey 2024 §3–4, ★*
- **Skill** — timestamp every node and edge in a REG and enforce time-respecting neighbor sampling for a
  prediction time.
- **Teach** — entity time vs event time, seed-time subgraph sampling, the "no future" guarantee (callback
  Y3 L101/L109), heterogeneous + temporal combined.
- **Lab** — Tier B · crucial fragment: add timestamps + a time-respecting sampler to the L122 REG.
  Deliverable: a leak-free temporal subgraph for a seed node at time t.
- **Viz** — reuse `temporal-graph-viz.js` + `hetero-graph-viz.js`.
- **Bridge** — the temporal correctness the thesis lives or dies on; callback Y3 L104; forward to L156 audit.

### 124 · Entity vs task table — *Fey 2024, ★*
- **Skill** — distinguish entity tables from task tables and select the prediction node + label + seed time
  for a task.
- **Teach** — task tables as (entity, time, label) triples, how a task defines the training signal on top of
  a fixed REG, entity-/link-/temporal-task types.
- **Lab** — Tier B · crucial fragment: define a task table over the L123 REG (entity, seed time, label).
  Deliverable: the task-table spec + one training example subgraph.
- **Viz** — reuse `rdl-stack-viz.js` (task node highlighted).
- **Bridge** — separates the reusable REG from the per-task signal; callback Y3 L087 link setup; forward to L128.

### 125 · PyTorch Frame deep dive — *Hu 2024, ★ `2404.00776`*
- **Skill** — configure PyTorch Frame stype encoders for every column type in a RelBench database and
  materialize node features for the REG.
- **Teach** — stype system, per-stype encoders (numeric/categorical/text/timestamp/embedding), the
  encoder→GNN interface RelBench uses (deepens Y2 L075).
- **Lab** — Tier B · crucial fragment (paper-repro): materialize a RelBench table's `TensorFrame`, attach
  encoders, feed the L122 REG. Deliverable: encoded REG node features.
- **Viz** — reuse `tokenizer-viz.js` + `cell-graph-viz.js`.
- **Bridge** — callback Y2 L075; the row-encoder half of the RDL stack; forward to L125b (make it modular).

### 125b · Universal Row Encoder — *Peleška & Šír 2026, ★ `2606.21434`*
- **Skill** — explain the modular decoupling of row encoding from graph message passing, and how a
  transformer row encoder fed global column statistics produces table-width-invariant embeddings that
  transfer across databases.
- **Teach** — the four pillars (semantic granularity, structural topology, temporal causality, unified
  optimization); intra-row self-attention over cells + schema metadata + global stats; a pretrained
  "backend" for any downstream GNN; cross-database transfer + faster convergence.
- **Lab** — Tier B · crucial fragment (paper-repro): swap the L125 encoder for a modular row encoder;
  measure convergence/memory + transfer to a second database. Deliverable: modular vs monolithic table.
- **Viz** — reuse `tokenizer-viz.js` (global-stats-conditioned) + `rdl-stack-viz.js` (decoupled block).
- **Bridge** — the encoder pillar of the *foundation* relational model (Y5); callback Y2 L076 decoupling;
  forward to Y5 L163/L167.

### 126 · RelBench beta paper — *RelBench beta `2312.04615`*
- **Skill** — navigate the RelBench package (datasets, tasks, loaders, evaluators) and run its tour.
- **Teach** — package layout, the RDL lineage from the beta to v1, how tasks/DBs are exposed in code.
- **Lab** — Tier B · crucial fragment: load a RelBench DB + task via the API; inspect the schema.
  Deliverable: a working RelBench load + schema summary.
- **Viz** — reuse `hetero-graph-viz.js` (RelBench schema).
- **Bridge** — tooling for the whole year; callback Y3 L111 OGB literacy; forward to L127.

### 127 · RelBench v1 — *Robinson 2024, ★ `2407.20060`*
- **Skill** — describe RelBench v1's 7 databases / diverse tasks and the standardized RDL evaluation, and
  run one task end to end.
- **Teach** — the databases/domains/scales, the RDL pipeline (row encoder + GNN), the standardized splits/
  metrics; the "first comprehensive RDL study."
- **Lab** — Tier B · crucial fragment (paper-repro): run the provided RDL baseline on one task.
  Deliverable: matched baseline metric.
- **Viz** — reuse `rdl-stack-viz.js`.
- **Bridge** — the benchmark the thesis is measured on; callback Y2 L056 benchmark literacy; forward to L130.

### 128 · Task taxonomy — *RelBench docs*
- **Skill** — classify RelBench tasks (entity classification/regression, recommendation/link, temporal) and
  pick the right head + metric for each.
- **Teach** — the task-type taxonomy, per-type heads and metrics (AUROC, MAE, MAP/Hits@k), how the seed time
  shapes each.
- **Lab** — Tier B · crucial fragment: for three tasks, specify head + metric + split. Deliverable: the
  task-type table.
- **Viz** — reuse `rdl-stack-viz.js` (head variants).
- **Bridge** — callback L124 task tables, Y3 L087/L106 link/temporal; forward to the multi-task portfolio (Q4).

### 129 · Manual FE study — *Robinson 2024 user study, ★*
- **Skill** — replicate the manual-feature-engineering baseline mindset and quantify the human effort RDL
  is competing against.
- **Teach** — the user-study protocol (expert hand-features per task), the ">order-of-magnitude less human
  work" result, why FE is the gold standard to beat.
- **Lab** — Tier B · crucial fragment: hand-engineer FE features for one task (as the study's data scientist
  would) + train a tuned tree. Deliverable: the FE baseline + effort log (time spent).
- **Viz** — reuse `feature-viz.js` + `flatten-loss-viz.js`.
- **Bridge** — the human-effort ratio at the heart of the thesis; callback Y1 L009/L033; forward to L155.

### 130 · **Q1 checkpoint** — *Fey 2024 + Robinson 2024 · Deliverable-based*
- **Deliverable** — run **one RelBench task end to end** (build REG → encode rows → train RDL GNN →
  evaluate with the official evaluator), matching the published baseline within tolerance.
- **Bridge** — proves the full RDL pipeline works before optimizing it; callback L122–L127.

---

## Q2 · RDL architectures & baselines (131–140)

### 131 · GNN + tabular encoder stack — *Fey 2024 §5, ★*
- **Skill** — trace a full RDL forward pass: row encoder → hetero temporal message passing → task head, with
  shapes at each stage.
- **Teach** — the complete stack (deepens Y2 L076 / Y3 L115), where encoder gradients meet GNN gradients,
  end-to-end training.
- **Lab** — Tier B · crucial fragment: instrument the L130 model to print shapes/activations per stage.
  Deliverable: annotated forward-pass trace.
- **Viz** — reuse `rdl-stack-viz.js` (shape annotations).
- **Bridge** — the mental model for every later architecture; callback Y2 L076, Y3 L115.

### 132 · ID-GNN / IDMP variants — *RelBench baselines*
- **Skill** — compare the RelBench baseline message-passing variants (e.g. ID-aware GNN) and explain what
  identity-awareness adds.
- **Teach** — why vanilla message passing can't distinguish some structures (expressiveness, callback Y3
  L088), ID-GNN's fix, the RelBench baseline family.
- **Lab** — Tier B · crucial fragment: run 2 baseline variants on one task. Deliverable: variant comparison.
- **Viz** — reuse `message-passing-viz.js` + `wl-viz.js`.
- **Bridge** — expressiveness limits that RelGNN later attacks (L141); callback Y3 L088.

### 133 · Hetero conv on REG — *PyG + RelBench*
- **Skill** — design the per-edge-type convolution and aggregation across node types for a REG.
- **Teach** — `HeteroConv`, per-relation transforms (callback Y3 L091 R-GCN), aggregation across relations,
  layer design choices.
- **Lab** — Tier B · crucial fragment: build a `HeteroConv` layer for the REG's relation set. Deliverable:
  the hetero conv + its effect on the metric.
- **Viz** — reuse `hetero-graph-viz.js` + `rdl-stack-viz.js`.
- **Bridge** — callback Y3 L091/L098; the layer the whole portfolio uses.

### 134 · Training at scale — *RelBench sampling*
- **Skill** — train an RDL model over millions of nodes with time-respecting hetero neighbor sampling and a
  realistic compute budget.
- **Teach** — subgraph sampling per seed, batch construction, memory/throughput at RelBench scale, the
  baseline-vs-extended-session compute reality.
- **Lab** — Tier B · crucial fragment: configure the hetero temporal sampler + a mini-batch loop at scale.
  Deliverable: a scaled training run + compute note.
- **Viz** — reuse `temporal-graph-viz.js` + `group-viz.js`.
- **Bridge** — callback Y3 L108/L113; scale is a first-class thesis constraint; forward to Y5 FM scale.

### 135 · Hyperparameter tuning on REG — *— (fair-tuning unit)*
- **Skill** — tune an RDL model under a **fixed, documented budget** equal to the tree baseline's, so
  comparisons stay fair.
- **Teach** — the RDL search space (encoder dims, GNN depth/width, sampling), the equal-budget rule
  (callback Y1 L017, Y2 L050), variance reporting.
- **Lab** — Tier B · crucial fragment: run a budgeted search; log the budget spent. Deliverable: tuned model
  + budget ledger.
- **Viz** — reuse `search-viz.js` + `checklist.js`.
- **Bridge** — fairness discipline the thesis needs; callback Y1 L017/Y2 L060; forward to L140.

### 136 · Leaderboard literacy — *relbench.stanford.edu*
- **Skill** — read the RelBench leaderboard and reproduce a top entry's reported setup.
- **Teach** — the leaderboard protocol, what counts as a valid entry, reading top-entry configs.
- **Lab** — Tier B · crucial fragment: reproduce a leaderboard entry's number. Deliverable: matched entry +
  config diff.
- **Viz** — reuse `checklist.js`.
- **Bridge** — callback Y3 L111; forward to Y5 RelBench v2 leaderboard, Y6 submission.

### 137 · Error analysis on REG — *— (analysis unit)*
- **Skill** — locate where the RDL GNN beats and loses to manual FE (task type, entity degree, temporal
  regime, cold start).
- **Teach** — slice-based error analysis on REG (callback Y3 L114), the honest "where FE still wins" map.
- **Lab** — Tier B · crucial fragment: slice one task's errors by entity property + time. Deliverable:
  win/lose map GNN vs FE.
- **Viz** — reuse `checklist.js` + a slice bar chart.
- **Bridge** — feeds the thesis's honest limitations; callback L129; forward to L149/L155.

### 138 · Domain: e-commerce task — *rel-amazon*
- **Skill** — deep-dive one e-commerce task (e.g. churn / LTV / recommendation) end to end and interpret the
  result in domain terms.
- **Teach** — the rel-amazon schema, the business meaning of the label, recsys-as-link (callback Y3 L095/
  L106), domain-specific pitfalls.
- **Lab** — Tier B · crucial fragment: full pipeline on one rel-amazon task + a domain-readable result.
  Deliverable: metric + a business interpretation.
- **Viz** — reuse `hetero-graph-viz.js` (rel-amazon schema) + `cross-viz.js` for recsys.
- **Bridge** — the flagship thesis domain; callback Y2 L048 recsys; forward to L144 ContextGNN.

### 139 · Domain: healthcare / social — *RelBench v1 tasks*
- **Skill** — run a second-domain task (healthcare/social) and show the RDL pipeline transfers across domains
  with only schema/task changes.
- **Teach** — a contrasting schema/label, domain-specific temporal semantics + privacy sensitivity (preview
  Y5 L187), generalization of the pipeline.
- **Lab** — Tier B · crucial fragment: adapt the L138 pipeline to a new domain by changing only the REG/task
  spec. Deliverable: second-domain result + what had to change.
- **Viz** — reuse `hetero-graph-viz.js`.
- **Bridge** — evidence RDL is a general method, not a per-dataset hack; callback L124; forward to Y5 cross-DB.

### 140 · **Q2 checkpoint** — *Deliverable-based*
- **Deliverable** — match the **published GNN baseline on 2 tasks** under a fair, budgeted, temporally
  correct protocol, with variance reported.
- **Bridge** — baseline RDL competence locked before the next-gen architectures; callback L131–L137.

---

## Q3 · Advanced RDL methods (141–150)

**Papers (chronological):** ContextGNN `2411.19513` ★ · RelGNN `2502.06784` ★ (ICML 2025) · RelGT
`2505.10960` ★ (ICLR 2026) · RDL survey `2506.16654`.

### 141 · Composite message passing — *RelGNN `2502.06784`, ★*
- **Skill** — implement RelGNN's **atomic routes** — simple paths enabling direct single-hop interaction
  between source and destination nodes — and the composite message passing built on them.
- **Teach** — atomic routes as a structural primitive for REG, composite message passing + graph attention
  that reduce redundancy and highlight key signals; up to 25% gains on 30 RelBench tasks.
- **Lab** — Tier B · crucial fragment (paper-repro): implement atomic-route construction + one composite
  message-passing layer. Deliverable: RelGNN layer + gain on one task.
- **Viz** — new `atomic-route-viz.js`: source→hub→destination route highlighted on a REG (reused L142/L145).
- **Bridge** — the current best *open supervised* RDL model; callback Y3 L092 meta-paths; forward to L143.

### 142 · Many-to-many edge pathology — *RelGNN §motivation, ★*
- **Skill** — demonstrate *why* vanilla GNNs lose signal on many-to-many relationships (redundant/averaged
  messages through high-degree hub nodes).
- **Teach** — the hub-node bottleneck, signal dilution through many-to-many joins, how atomic routes bypass
  it; the structural-property argument.
- **Lab** — Tier B/C · crucial fragment: build a many-to-many toy REG; show vanilla message passing dilutes
  a signal that atomic routes preserve. Deliverable: the pathology demonstration.
- **Viz** — reuse `atomic-route-viz.js` + `flatten-loss-viz.js`.
- **Bridge** — the "why" behind L141; callback Y3 L085 over-smoothing / over-squashing; grounds the fix.

### 143 · RelGNN reproduction — *RelGNN `2502.06784`, ★*
- **Skill** — reproduce a RelGNN result on a RelBench subset within tolerance and write a reproduction report.
- **Teach** — the reproduction workflow at RDL scale (config, seeds, variance, honest fail), reading the
  paper's exact settings.
- **Lab** — Tier B · crucial fragment (paper-repro): run RelGNN on ≥1 task; match the paper or document an
  honest fail. Deliverable: reproduction report.
- **Viz** — reuse `atomic-route-viz.js` + `checklist.js`.
- **Bridge** — first frontier-model reproduction; callback Y3 L112; forward to L150 checkpoint, Y6 baselines.

### 144 · ContextGNN — beyond two-tower recsys — *ContextGNN `2411.19513`, ★*
- **Skill** — implement ContextGNN's fusion of **pair-wise** representations (for familiar items in a user's
  local subgraph) with **two-tower** representations (for exploratory items).
- **Teach** — two-tower limits (pair-agnostic), pair-wise scaling limits, the fusion network that ranks both;
  ~20% avg gains on recommendation.
- **Lab** — Tier B · crucial fragment (paper-repro): implement the pair-wise + two-tower fusion on a
  RelBench recsys task. Deliverable: ContextGNN vs two-tower baseline.
- **Viz** — reuse `cross-viz.js` + `hetero-graph-viz.js` (bipartite) + `retrieval-viz.js`.
- **Bridge** — the recsys/link-prediction frontier; callback Y3 L095/L106, Y2 L048; forward to Y5 recsys FMs.

### 145 · Relational Graph Transformer — *RelGT `2505.10960`, ★*
- **Skill** — implement RelGT's **multi-element tokenization** (each node → features, type, hop distance,
  time, local structure) with local attention over sampled subgraphs + global attention to learnable
  centroids.
- **Teach** — why standard positional encodings fail on massive hetero graphs, the 5-element token, local+
  global attention, matching/beating GNNs by up to 18% on 21 RelBench tasks.
- **Lab** — Tier B · crucial fragment (paper-repro): implement the 5-element tokenizer + local attention.
  Deliverable: RelGT vs GNN baseline on one task.
- **Viz** — reuse `tokenizer-viz.js` (5-element) + `atomic-route-viz.js`.
- **Bridge** — Graph Transformers for RDL; callback Y3 Graphormer/GraphGPS (◆), L093 HGT; forward to Y5 RelGT-AC/GelGT.

### 146 · GNN vs Graph-Transformer on REG — *RelGT §results, ★*
- **Skill** — run RelGNN (GNN paradigm) and RelGT (transformer paradigm) on the *same* tasks and attribute
  differences (long-range vs local, compute).
- **Teach** — the controlled two-paradigm comparison (callback Y3 L099), where global attention beats
  message passing and vice versa, compute trade-offs.
- **Lab** — Tier B · crucial fragment: matched RelGNN-vs-RelGT run + attribution. Deliverable: paradigm table.
- **Viz** — reuse `arch-family-viz.js` (relational branch) + `atomic-route-viz.js`.
- **Bridge** — the architecture-research gap that Y5 L183 revisits; callback Y3 L099.

### 147 · Survey: next-gen architectures — *RDL survey `2506.16654`*
- **Skill** — map RDL's open problems (large-scale multi-table integration, temporal dynamics, heterogeneity,
  the convergence toward foundation models) from the survey.
- **Teach** — the survey's REG framing, benchmark review, foundational + next-gen method taxonomy, the
  "converging toward foundation models" thesis.
- **Lab** — Tier C · crucial fragment: build an open-problems map ranked by tractability. Deliverable: the map.
- **Viz** — reuse `arch-family-viz.js`.
- **Bridge** — the on-ramp to Year 5 (foundation models); callback L141/L145; forward to Y5 L169/L189.

### 148 · Ablation discipline — *— (methodology unit)*
- **Skill** — design an ablation that separates the contribution of the **encoder** vs **message passing**
  vs **data/graph construction** to an RDL result.
- **Teach** — the three-way attribution (why a gain must be sourced), controlled ablations, the trap of
  attributing to architecture what came from data.
- **Lab** — Tier B · crucial fragment: ablate encoder/MP/graph on one task. Deliverable: attribution table.
- **Viz** — reuse `checklist.js` + `rdl-stack-viz.js`.
- **Bridge** — the rigor for L157b (graph construction) and the Y6 research project; callback Y2 L059.

### 149 · Identify weakest RelBench tasks — *— (thesis stress unit)*
- **Skill** — find the tasks where RDL *underperforms* manual FE / trees and hypothesize why (cold start,
  low relational signal, rule-governed labels).
- **Teach** — where the thesis is weakest, honest failure cataloguing, connecting to Y2 L069b operational
  barrier + L056b non-IID.
- **Lab** — Tier B · crucial fragment: rank tasks by RDL-minus-FE gap; profile the worst. Deliverable:
  weakness catalog.
- **Viz** — reuse `checklist.js` + slice bars.
- **Bridge** — the falsification discipline the thesis requires; callback L137, Y2 L069b; forward to Y6 L195.

### 150 · **Q3 checkpoint** — *RelGNN or RelGT · Deliverable-based*
- **Deliverable** — reach **SOTA or near-SOTA on 1 task** with RelGNN or RelGT, plus a written reproduction
  report (config, variance, honest gaps).
- **Bridge** — frontier-model competence locked; callback L143/L145; forward to the Q4 portfolio.

---

## Q4 · RDL expertise (151–160)

The `CURRICULUM.md` range `151–154 · Multi-task RelBench portfolio` is decomposed into four lessons below.

### 151 · Portfolio task 1 — entity classification — *— (portfolio unit)*
- **Skill** — deliver a complete, documented result on an entity-classification task (model, tuning, metric,
  variance) as one entry in a reusable portfolio.
- **Teach** — the portfolio-entry contract (reproducible config, fair budget, variance), consistent reporting.
- **Lab** — Tier B · crucial fragment: full pipeline + portfolio entry. Deliverable: entry 1 (result + config).
- **Viz** — reuse `rdl-stack-viz.js` + `checklist.js`.
- **Bridge** — starts the ≥3-task exit portfolio; callback L128/L130.

### 152 · Portfolio task 2 — entity regression — *— (portfolio unit)*
- **Skill** — deliver a regression-task entry with correct metric (MAE/RMSE) and calibration check.
- **Teach** — regression heads/metrics on REG, calibration (callback Y1 L008), variance.
- **Lab** — Tier B · crucial fragment: regression pipeline + entry. Deliverable: entry 2.
- **Viz** — reuse `reliability-viz.js` (regression calibration).
- **Bridge** — broadens the portfolio; callback Y1 L008.

### 153 · Portfolio task 3 — recommendation / link — *— (portfolio unit)*
- **Skill** — deliver a recommendation/link-prediction entry with ranking metrics + temporal negatives.
- **Teach** — recsys metrics (MAP/Hits@k), temporal negatives (callback Y3 L097/L106), ContextGNN as the
  strong option (L144).
- **Lab** — Tier B · crucial fragment: recsys pipeline + entry. Deliverable: entry 3.
- **Viz** — reuse `cross-viz.js` + `hetero-graph-viz.js`.
- **Bridge** — completes the ≥3-task requirement; callback L144.

### 154 · Portfolio synthesis report — *— (portfolio unit)*
- **Skill** — synthesize the 3+ entries into one report with a consistent fair-comparison table (RDL vs
  FE vs trees) and per-task verdicts.
- **Teach** — cross-task reporting, when to aggregate vs report per-task, single-benchmark-claim caveats.
- **Lab** — Tier B · crucial fragment: assemble the unified results table. Deliverable: the portfolio report.
- **Viz** — reuse `checklist.js` + a results table.
- **Bridge** — the artifact the Y4 exit exam grades; callback L151–L153; forward to L155.

### 155 · Compare to manual FE — *Robinson user-study protocol, ★*
- **Skill** — quantify the **human-effort ratio** (hours of FE vs RDL setup) alongside accuracy across the
  portfolio, replicating the RelBench user-study logic.
- **Teach** — measuring human effort, the ">order-of-magnitude" claim tested on your own work, accuracy-per-
  effort as the real thesis metric.
- **Lab** — Tier B · crucial fragment: log FE hours vs RDL hours across tasks + accuracy. Deliverable:
  effort-vs-accuracy table.
- **Viz** — reuse `feature-viz.js` + a effort/accuracy scatter.
- **Bridge** — the core of the Y4 exit criterion; callback L129, Y1 L033; forward to Y6 thesis.

### 156 · Temporal leakage audit — *— (audit unit)*
- **Skill** — run a full temporal-leakage audit on the REG pipeline (seed times, neighbor sampling, label
  windows, feature timestamps) with a reusable checklist.
- **Teach** — the complete REG audit checklist (callback Y3 L104, L123), common RDL time-travel bugs, sign-off.
- **Lab** — Tier B · crucial fragment: audit the portfolio pipeline; find/fix any leak; re-measure.
  Deliverable: signed audit + any corrected numbers.
- **Viz** — reuse `leakage-viz.js` (temporal REG) + `checklist.js`.
- **Bridge** — makes the portfolio numbers trustworthy; callback Y3 L104/Y4 L123; the discipline for Y6.

### 157 · Open-source contribution — *— (contribution unit)*
- **Skill** — package a reproduction as a public artifact (PR to RelBench/PyG or a reproducibility repo) with
  docs + fixed env.
- **Teach** — contribution mechanics (issue → PR, reproducibility standards, licensing), community norms.
- **Lab** — Tier B · crucial fragment: open a PR or publish a reproduction repo. Deliverable: the public link.
- **Viz** — none (code/PR artifact).
- **Bridge** — first public artifact toward the Y6 launch; callback L143; forward to Y5 L196, Y6 L222.

### 157b · When schema graphs fail — *Desired graph `2606.08491`, ★ (ICML 2026)*
- **Skill** — apply controlled structural adaptation to a REG — **filtering** (mitigate information overload)
  and **injection** (repair semantic fragmentation) — and explain why the raw schema graph is often not the
  desired graph.
- **Teach** — the two systematic failures (information overload, semantic fragmentation), filtering as a
  non-monotonic bias-variance knob, injection that restores missing relational dependencies, the end-to-end
  structural optimizer; gains + lower inference cost across 26 tasks.
- **Lab** — Tier B · crucial fragment (paper-repro): apply a filtering + injection step to a portfolio task's
  REG; measure accuracy + inference cost. Deliverable: raw-vs-optimized-graph table.
- **Viz** — reuse `hetero-graph-viz.js` (before/after adaptation) + `flatten-loss-viz.js`.
- **Bridge** — an optimizer *on top of* the REG vocabulary (must understand Fey's default REG first, L122);
  callback L142 pathology; forward to Y6 graph-construction experiments.

### 158 · Year 4 synthesis essay — *— (writing unit)*
- **Skill** — write the evidence for/against the thesis from the portfolio: where RDL beat FE, by how much,
  at what effort, and where it honestly lost.
- **Teach** — synthesis of L155 (effort), L149/L137 (weaknesses), L157b (graph matters); a falsifiable
  interim verdict.
- **Lab** — writing deliverable: the evidence essay with the portfolio table as backbone.
- **Viz** — reuse the L154/L155 tables.
- **Bridge** — the honest mid-thesis verdict; callback L149/L155; forward to Y6 hypothesis.

### 159 · Foundation model preview — *Zahradník 2023 full, `2305.15321`*
- **Skill** — state the relational foundation-model vision (learn from the *full* relational structure +
  scale to real DBs) and why it follows from RDL.
- **Teach** — Zahradník's vision paper (LM/GNN pretraining over full DBs, scale limits of single-table
  models), the convergence the survey (L147) predicted; the Year-5 north star.
- **Lab** — Tier C · crucial fragment: write a one-page FM vision brief grounded in the portfolio's limits.
  Deliverable: the brief.
- **Viz** — reuse `rdl-stack-viz.js` + `arch-family-viz.js`.
- **Bridge** — the on-ramp to Year 5; callback L147; forward to Y5 L162.

### 160 · **Year 4 exit exam** — *all Y4 papers · Deliverable-based*
- **Deliverable** — a **RelBench portfolio (≥3 tasks)** with documented comparison to manual FE (accuracy +
  human-effort ratio), a passed temporal-leakage audit, and the evidence essay.
- **Exit criterion (from CURRICULUM)** — reproduced RelBench baselines + documented FE comparison on ≥3 tasks.
- **Bridge** — gate to Year 5. Artifacts: the RelBench portfolio harness, the audit checklist, the FM brief.

**Optional (◆, after exit):** 4DBInfer `2404.18209` (second graph-centric RDB benchmark — stress-test
whether conclusions are method/benchmark-dependent); DBFormer `2412.05218` (SQL-native transformer,
contrast to graph-native RDL); ReDeLEx `2506.22199` (70+ CTU DBs); FROG `2605.21475`; DHN expressivity
`2605.22852`.
