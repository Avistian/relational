# Year 3 — Graph Machine Learning · Lesson Decomposition

**Year goal:** implement GNNs from message-passing first principles through heterogeneous and temporal
graphs — the exact graph types a relational database becomes. Reproduction-dominated: each unit is a
PyG implementation, not a reading. Labs are Tier B/C (graph benchmarks + toy REG-shaped graphs) and
build on the `labs/relkit/` harness (now extended with graph loaders).

The collapsed row `111–114` in `CURRICULUM.md` is decomposed below into four concrete OGB-reproduction
lessons.

---

## Q1 · Message passing foundations (081–090)

**Papers (chronological):** Gilmer 2017 MPNN `1704.01212` ★ · Kipf & Welling 2017 GCN `1609.02907` ★ ·
Hamilton 2017 GraphSAGE `1706.02216` ★ · Veličković 2018 GAT `1710.10903` · Fey & Lenssen 2019 PyG ·
Xu 2019 GIN `1810.00826` · Chiang 2019 Cluster-GCN `1905.07953` · Li 2018 over-smoothing `1801.07606`.

### 081 · MPNN framework — *Gilmer 2017, ★ `1704.01212`*
- **Skill** — implement a generic MPNN (message → aggregate → update → readout) and instantiate GCN/GAT as
  special cases of it.
- **Teach** — the message/aggregate/update/readout abstraction, permutation invariance, why aggregation
  must be order-invariant (callback Y2 L078).
- **Lab** — Tier C · crucial fragment: write the generic `message`/`aggregate`/`update` functions; recover a
  mean-aggregator GNN. Deliverable: one-layer MPNN forward pass on a toy graph.
- **Viz** — reuse `message-passing-viz.js` (from Y2 L078).
- **Bridge** — the unifying frame for all of Y3; a GNN layer = learned relational aggregation; callback L078.

### 082 · GCN — *Kipf & Welling 2017, ★ `1609.02907`*
- **Skill** — implement the GCN propagation rule (symmetric-normalized adjacency) and train node
  classification on Cora.
- **Teach** — `H' = σ(D̃^{-1/2}Ã D̃^{-1/2} H W)`, the spectral→spatial simplification, self-loops,
  transductive setup.
- **Lab** — Tier B (Cora, PyG) · crucial fragment (paper-repro): implement the normalized propagation;
  reproduce Cora accuracy within tolerance. Deliverable: matched accuracy.
- **Viz** — reuse `message-passing-viz.js` (normalized weights) + a small Cora embedding scatter.
- **Bridge** — the canonical GNN; callback L081 (GCN as an MPNN); forward to L083 inductive limits.

### 083 · GraphSAGE — *Hamilton 2017, ★ `1706.02216`*
- **Skill** — implement neighbor-sampling + aggregator (mean/pool/LSTM) and train *inductively* with
  mini-batches on unseen nodes.
- **Teach** — inductive vs transductive, neighbor sampling for scale, the aggregator choice; why induction
  matters for growing databases (new rows).
- **Lab** — Tier B · crucial fragment (paper-repro): implement mean-aggregator SAGE + a `NeighborLoader`
  mini-batch loop; train inductively. Deliverable: inductive test accuracy on held-out nodes.
- **Viz** — reuse `message-passing-viz.js` (sampled neighborhood) + `group-viz.js` analogy for batches.
- **Bridge** — the scalable, inductive GNN RelBench actually uses; callback L082; forward to L089 sampling, Y4.

### 084 · GAT — *Veličković 2018, `1710.10903`*
- **Skill** — implement multi-head graph attention and interpret learned edge attention weights.
- **Teach** — attention coefficients over neighbors, multi-head concat/average, vs GCN's fixed weights;
  when learned edge weights help.
- **Lab** — Tier B · crucial fragment: implement the attention coefficient + softmax over neighbors.
  Deliverable: Cora accuracy + an attention-weight visualization.
- **Viz** — reuse `mask-viz.js` (edge-attention heatmap) + `message-passing-viz.js`.
- **Bridge** — attention on graphs → HGT (L093), Graph Transformers (Y4 RelGT); callback Y2 attention lessons.

### 085 · Over-smoothing — *Li et al. 2018, ◆ `1801.07606`*
- **Skill** — demonstrate that stacking many GNN layers collapses node representations toward
  indistinguishability, and state the depth/performance trade-off.
- **Teach** — repeated averaging → low-pass filter → node features converge; why GNNs are shallow; remedies
  preview (residuals, DropEdge, PNA).
- **Lab** — Tier B/C · crucial fragment: measure representation similarity vs depth; find the collapse point.
  Deliverable: depth-vs-accuracy + similarity curve.
- **Viz** — new `oversmoothing-viz.js`: node colors converge as layers increase (reused Y4 deep-REG pathology).
- **Bridge** — the depth failure mode that predicts where deep REG message passing breaks (Y4); ◆ DropEdge/
  PNA/over-squashing read here.

### 086 · PyG fundamentals — *Fey & Lenssen 2019*
- **Skill** — use PyG's `Data`/`HeteroData`, `NeighborLoader`, and message-passing base class fluently.
- **Teach** — the `MessagePassing` class (`propagate`/`message`/`aggregate`/`update`), data objects, mini-
  batch loaders; the library the rest of the curriculum runs on.
- **Lab** — Tier B · crucial fragment: reimplement L082 GCN using PyG's `MessagePassing`. Deliverable: PyG
  GCN matching the from-scratch one.
- **Viz** — reuse `message-passing-viz.js`.
- **Bridge** — tooling backbone for Y3–Y5; callback L081; forward to L098 HeteroData, Y4 RelBench.

### 087 · Link prediction — *Zhang & Chen 2018 (skim)*
- **Skill** — set up link prediction (edge scoring, negative sampling, edge-level train/val/test split) and
  evaluate with ranking metrics.
- **Teach** — encoder→edge-decoder, why edge splits leak differently than node splits, MRR/Hits@k.
- **Lab** — Tier B · crucial fragment: implement a dot-product edge decoder + edge split. Deliverable: link-
  pred MRR/Hits@k.
- **Viz** — reuse `split-viz.js` (edge-level split) + `leakage-viz.js`.
- **Bridge** — link prediction *is* RelBench recommendation (Y4 L144 ContextGNN); callback Y1 split hygiene.

### 088 · Graph classification — *Xu 2019 (GIN), `1810.00826`*
- **Skill** — implement GIN and explain the Weisfeiler-Lehman expressiveness bound (why sum-aggregation is
  maximally expressive among MPNNs).
- **Teach** — WL test, injective aggregation, sum vs mean vs max readout, expressiveness hierarchy.
- **Lab** — Tier B · crucial fragment: implement GIN's sum-aggregation + MLP; whole-graph readout on a
  graph-classification set. Deliverable: accuracy + which readout wins.
- **Viz** — new `wl-viz.js`: WL color-refinement rounds (reused Y4 expressiveness discussions).
- **Bridge** — expressiveness theory underpinning Y4 RelGNN (why vanilla MPNNs lose many-to-many signal);
  callback L081 aggregation.

### 089 · Sampling at scale — *Chiang 2019 (Cluster-GCN), ◆ `1905.07953`*
- **Skill** — train a GNN on a graph too large for full-batch using clustering-based subgraph sampling.
- **Teach** — Cluster-GCN partitioning, memory/accuracy trade-offs vs neighbor sampling; scale is a
  first-class RDL concern (millions of nodes).
- **Lab** — Tier B · crucial fragment: partition a large graph + train on cluster batches. Deliverable:
  scale-vs-accuracy vs full-batch.
- **Viz** — reuse `group-viz.js` (partitioned clusters).
- **Bridge** — callback L083 neighbor sampling; forward to Y4 L134 (mini-batch over millions of REG nodes).

### 090 · **Q1 checkpoint** — *Kipf + Hamilton · Deliverable-based*
- **Deliverable** — a GNN implemented **from scratch** (message passing, not just `import`) reproducing a
  node-classification benchmark within tolerance, with an inductive mini-batch variant.
- **Bridge** — proves message-passing fluency before heterogeneity; callback L082/L083.

---

## Q2 · Heterogeneous graphs (091–100)

**Papers:** Schlichtkrull 2018 R-GCN `1703.06103` ★ · Wang 2019 HAN `1903.07293` · Hu 2020 HGT `2003.01332`
★ · Sun & Han 2020 HIN survey.

### 091 · R-GCN — *Schlichtkrull 2018, ★ `1703.06103`*
- **Skill** — implement relation-specific weight matrices (one per edge type) with basis decomposition and
  explain why relation typing matters.
- **Teach** — per-relation transforms, basis/block-diagonal decomposition to control parameters, the direct
  analogy to SQL foreign-key types.
- **Lab** — Tier B · crucial fragment (paper-repro): implement per-relation message weights + basis decomp.
  Deliverable: entity-classification accuracy on a hetero benchmark.
- **Viz** — new `hetero-graph-viz.js`: typed nodes/edges with per-relation colors (anchor asset for Q2 + Y4 REG).
- **Bridge** — the first "edges have types" model — directly maps to PK/FK relations (Y4); callback L082.

### 092 · Meta-paths — *Wang 2019 (HAN), `1903.07293`*
- **Skill** — define meta-paths and implement hierarchical (node-level + semantic-level) attention over them.
- **Teach** — meta-path as a typed path template (e.g. author→paper→author), node vs semantic attention;
  hand-crafted vs learned relation composition.
- **Lab** — Tier B · crucial fragment: enumerate meta-paths + implement semantic attention. Deliverable:
  which meta-path the model weights highest.
- **Viz** — reuse `hetero-graph-viz.js` (highlight a meta-path) + `mask-viz.js` (semantic attention).
- **Bridge** — meta-paths = multi-hop relational routes → RelGNN atomic routes (Y4 L141); callback L084 attention.

### 093 · HGT — *Hu 2020, ★ `2003.01332`*
- **Skill** — implement heterogeneous graph transformer (type-specific attention + relative temporal
  encoding) and contrast with R-GCN.
- **Teach** — per-type Q/K/V projections, meta-relation attention, relative temporal encoding; scalable
  hetero attention.
- **Lab** — Tier B · crucial fragment (paper-repro): implement type-specific attention heads. Deliverable:
  HGT vs R-GCN on the same hetero graph.
- **Viz** — reuse `hetero-graph-viz.js` + `mask-viz.js`.
- **Bridge** — the strongest classical hetero model + temporal encoding preview; direct ancestor of RelGT
  (Y4 L145); callback L091.

### 094 · HIN survey — *Sun & Han 2020*
- **Skill** — classify heterogeneous-information-network methods and place R-GCN/HAN/HGT in a taxonomy.
- **Teach** — HIN taxonomy (meta-path, meta-graph, message-passing families), evaluation conventions,
  open problems.
- **Lab** — Tier C · crucial fragment: build a taxonomy table locating each Q2 model. Deliverable: the map.
- **Viz** — reuse `arch-family-viz.js` (hetero branch).
- **Bridge** — organizing map before applying to databases; callback L091–L093.

### 095 · Bipartite graphs — *— (user-item as hetero graph)*
- **Skill** — model a user–item interaction table as a bipartite hetero graph and set up recommendation as
  bipartite link prediction.
- **Teach** — bipartite structure, why recsys is a two-node-type link-pred problem, projection pitfalls.
- **Lab** — Tier B · crucial fragment: build a bipartite `HeteroData` + bipartite link split. Deliverable:
  recsys link-pred baseline.
- **Viz** — reuse `hetero-graph-viz.js` (two-column bipartite layout).
- **Bridge** — sets up Y4 e-commerce/recsys (L138, L144); callback L087 link prediction.

### 096 · Multi-relational data → SQL FK semantics — *— (bridge unit)*
- **Skill** — translate a relational schema (tables, PK/FK) into a heterogeneous graph specification (node
  types = tables, edge types = FK links).
- **Teach** — the schema→graph mapping rules, one-to-many vs many-to-many FK, direction of edges; the exact
  construction Fey formalizes in Y4.
- **Lab** — Tier C · crucial fragment: given a 4-table schema, write the node-type/edge-type spec + build the
  `HeteroData`. Deliverable: the hetero graph spec.
- **Viz** — reuse `hetero-graph-viz.js` (schema → graph overlay) + `cell-graph-viz.js`.
- **Bridge** — the literal schema→REG bridge; callback Y1 L034 star schema; forward to Y4 L122 REG construction.

### 097 · Negative sampling — *— (training unit)*
- **Skill** — implement correct negative sampling for hetero link prediction and avoid false negatives /
  train-test contamination.
- **Teach** — uniform vs degree-based negatives, hard negatives, why bad negatives inflate metrics.
- **Lab** — Tier B · crucial fragment: implement a negative sampler + ablate negative strategies.
  Deliverable: metric sensitivity to negatives.
- **Viz** — reuse `hetero-graph-viz.js` (positive vs sampled-negative edges).
- **Bridge** — evaluation hygiene for recsys (Y4); callback L087, Y1 leakage discipline.

### 098 · Hetero mini-batching — *PyG HeteroData*
- **Skill** — build a RelBench-shaped toy hetero graph and train with `HeteroData` + hetero `NeighborLoader`.
- **Teach** — hetero neighbor sampling, per-type feature dicts, batching typed subgraphs; the exact loader
  RelBench uses.
- **Lab** — Tier B · crucial fragment: construct a 3-type toy graph + hetero mini-batch loop. Deliverable:
  trained hetero GNN on the toy graph.
- **Viz** — reuse `hetero-graph-viz.js` (sampled typed subgraph).
- **Bridge** — the training mechanics for Y4 RelBench; callback L083/L089; forward to L100.

### 099 · Compare R-GCN vs HGT — *— (analysis unit)*
- **Skill** — run R-GCN and HGT on the *same* hetero graph and attribute performance differences to
  architecture (typing vs attention).
- **Teach** — controlled comparison discipline (same data/split/budget), where attention helps hetero,
  parameter/compute trade-offs.
- **Lab** — Tier B · crucial fragment: matched R-GCN vs HGT run + ablation. Deliverable: attribution table.
- **Viz** — reuse `arch-family-viz.js` + `hetero-graph-viz.js`.
- **Bridge** — the "same graph, two paradigms" pattern reused for GNN-vs-Graph-Transformer in Y4 L146.

### 100 · **Q2 checkpoint** — *Schlichtkrull + Hu 2020 · Deliverable-based*
- **Deliverable** — a heterogeneous GNN trained on a graph with **3+ node types**, with correct hetero
  mini-batching and a controlled R-GCN-vs-HGT comparison.
- **Bridge** — heterogeneity mastered; only temporality remains before RDL; callback L091/L093.

---

## Q3 · Temporal & dynamic graphs (101–110)

**Papers:** Rossi 2020 TGN `2006.10637` ★ · Xu 2020 TGAT `2002.07962` · Kazemi/Jin 2020 RE-Net `1904.05530`.

### 101 · Static vs temporal — *Fey 2024 temporal REG preview*
- **Skill** — explain why static graph splits leak in temporal data and set up time-respecting splits.
- **Teach** — timestamped nodes/edges, the "no future information" rule, time-respecting neighbor sampling;
  the temporal analogue of Y1 PIT correctness.
- **Lab** — Tier C · crucial fragment: build a timestamped toy graph + a time-respecting split. Deliverable:
  a split where random-split leakage is demonstrable.
- **Viz** — reuse `split-viz.js` (temporal) + `leakage-viz.js` (time-travel edge).
- **Bridge** — callback Y1 L021/L002, Y2 L055 temporal split; forward to Y4 temporal REG audit (L156).

### 102 · Temporal Graph Networks — *Rossi 2020, ★ `2006.10637`*
- **Skill** — implement TGN's memory module + message function and explain how per-node memory carries
  history.
- **Teach** — node memory, message aggregation over events, memory update, embedding module; continuous-time
  events.
- **Lab** — Tier B · crucial fragment (paper-repro): implement the memory update + message aggregation.
  Deliverable: temporal link-pred on a dynamic graph within tolerance.
- **Viz** — new `temporal-graph-viz.js`: event stream updating per-node memory over time (reused Y4 temporal).
- **Bridge** — the canonical temporal GNN; the memory idea recurs in relational temporal modeling; callback L098.

### 103 · TGAT — *Xu 2020, ◆ `2002.07962`*
- **Skill** — implement time-encoded attention (functional time encoding) and contrast with TGN's memory.
- **Teach** — Bochner/functional time encoding, self-attention over temporal neighborhoods, memoryless vs
  memory-based temporal models.
- **Lab** — Tier B · crucial fragment: implement the time-encoding kernel + temporal attention. Deliverable:
  TGAT vs TGN on the same dynamic graph.
- **Viz** — reuse `temporal-graph-viz.js` + `temporal-embed-viz.js` (from Y2 L055b).
- **Bridge** — time encoding → RelGT's time token (Y4 L145); callback L084/L093 attention.

### 104 · Information leakage in time — *Kapoor 2022 + Fey 2024*
- **Skill** — audit a temporal GNN pipeline for time-travel bugs (using an edge's future to predict its own
  past) and fix them.
- **Teach** — common temporal leaks (label from future neighbors, non-causal features, val after test time),
  the audit checklist.
- **Lab** — Tier B · crucial fragment: inject then detect a time-travel leak; measure the inflated metric.
  Deliverable: leak found + corrected number.
- **Viz** — reuse `leakage-viz.js` (temporal edges) + `checklist.js`.
- **Bridge** — the discipline the whole thesis rests on (temporal correctness); callback Y1 L022, Y2 L055;
  forward to Y4 L156 full REG audit.

### 105 · Continuous time — *— (event streams)*
- **Skill** — model interactions as a continuous-time event stream vs discrete snapshots and choose between
  them.
- **Teach** — continuous-time dynamic graphs (CTDG) vs discrete-time (DTDG), event-based vs snapshot-based
  trade-offs.
- **Lab** — Tier C · crucial fragment: represent the same data as a CTDG and a DTDG. Deliverable: the two
  representations + when each is right.
- **Viz** — reuse `temporal-graph-viz.js` (continuous vs snapshot toggle).
- **Bridge** — informs REG design choices in Y4; callback L102/L103.

### 106 · Temporal link prediction — *— (future edge prediction)*
- **Skill** — set up and evaluate future-edge prediction with strictly causal splits and temporal negatives.
- **Teach** — predict edges at t+1 from ≤t, temporal negative sampling, evaluation windows.
- **Lab** — Tier B · crucial fragment: build a future-edge task + temporal negatives. Deliverable: temporal
  link-pred metric with a causal split.
- **Viz** — reuse `temporal-graph-viz.js` + `split-viz.js`.
- **Bridge** — RelBench recommendation is temporal link prediction; callback L087/L097; forward to Y4 L144.

### 107 · Snapshot methods — *— (discrete time slices)*
- **Skill** — implement discrete-time snapshot GNNs (per-slice GNN + sequence model) and compare to
  continuous-time.
- **Teach** — snapshot encoding + RNN/attention over slices, when discretization is adequate, granularity
  choice.
- **Lab** — Tier B · crucial fragment: build snapshots + a sequence model over slices. Deliverable: snapshot
  vs TGN on the same data.
- **Viz** — reuse `temporal-graph-viz.js` (snapshot mode).
- **Bridge** — callback L102/L105; a practical baseline for temporal REG (Y4).

### 108 · Scale: neighbor sampling over time — *— (efficiency unit)*
- **Skill** — implement efficient time-respecting neighbor sampling for large dynamic graphs.
- **Teach** — temporal neighbor sampling, recency windows, memory/throughput trade-offs at scale.
- **Lab** — Tier B · crucial fragment: implement a time-windowed neighbor sampler. Deliverable: throughput
  vs accuracy at scale.
- **Viz** — reuse `temporal-graph-viz.js` + `group-viz.js`.
- **Bridge** — callback L089/L101; the sampler Y4 RelBench training relies on.

### 109 · Connect to databases — *— (`observed_at` semantics)*
- **Skill** — map database timestamp columns (`created_at`, `observed_at`) to node/edge times and enforce
  point-in-time correctness at the schema level.
- **Teach** — which timestamp defines an entity's time, as-of joins, the DB→temporal-graph timestamp
  contract.
- **Lab** — Tier C · crucial fragment: assign timestamps to a schema's nodes/edges leak-free. Deliverable:
  a timestamped REG-shaped graph spec.
- **Viz** — reuse `hetero-graph-viz.js` (timestamped) + `leakage-viz.js`.
- **Bridge** — the literal DB→temporal-REG hand-off; callback L096/L101; forward to Y4 L123 (timestamp every node).

### 110 · **Q3 checkpoint** — *Rossi 2020 · Deliverable-based*
- **Deliverable** — a temporal GNN trained with a **clean time-respecting split** on a dynamic graph,
  passing a time-travel-leak audit.
- **Bridge** — temporality mastered; the student can now build the exact graph type a database becomes;
  callback L102/L104.

---

## Q4 · Graph ML mastery & the RDL bridge (111–120)

The `CURRICULUM.md` range `111–114 · OGB benchmark task` is decomposed into four lessons below.

### 111 · OGB setup & leaderboard literacy — *Hu 2020 OGB `2005.00687`*
- **Skill** — load an OGB dataset with the official splits/evaluator and read the leaderboard protocol.
- **Teach** — OGB datasets/tasks, standardized evaluators, why fixed splits matter for honest comparison.
- **Lab** — Tier B · crucial fragment: load `ogbn-*` with the official evaluator; run a provided baseline.
  Deliverable: baseline number matching the leaderboard protocol.
- **Viz** — reuse `checklist.js` (leaderboard rubric).
- **Bridge** — benchmark discipline transferring to RelBench (Y4); callback Y2 L056 TabArena literacy.

### 112 · OGB node/link reproduction — *Hu 2020 OGB*
- **Skill** — reproduce a published OGB node- or link-prediction number within tolerance using a Q1–Q3 model.
- **Teach** — the reproduction workflow (config, seeds, variance), reading a paper's exact settings.
- **Lab** — Tier B · crucial fragment (paper-repro): reproduce one OGB entry. Deliverable: matched metric +
  variance bars.
- **Viz** — reuse `message-passing-viz.js` / `hetero-graph-viz.js` as fits the task.
- **Bridge** — the reproduction muscle Y4/Y5 depend on; callback L090/L100/L110.

### 113 · Scaling the OGB run — *Hu 2020 OGB + Cluster-GCN/SAGE*
- **Skill** — scale the L112 model to a large OGB graph with sampling and hit a compute/accuracy target.
- **Teach** — sampling choice (neighbor vs cluster), memory budgeting, throughput; realistic-compute honesty
  (baseline vs extended sessions).
- **Lab** — Tier B · crucial fragment: convert the L112 model to mini-batch sampling on a large graph.
  Deliverable: scaled result + compute note.
- **Viz** — reuse `group-viz.js` (sampling) + `temporal-graph-viz.js` if temporal.
- **Bridge** — callback L089/L108; the scale skills for Y4 L134 (millions of REG nodes).

### 114 · OGB error analysis — *— (analysis unit)*
- **Skill** — analyze where the OGB model fails (degree, homophily, class, time) and write a short error
  report.
- **Teach** — slice-based error analysis, homophily/degree effects, when a GNN underperforms a simple baseline.
- **Lab** — Tier B · crucial fragment: slice metrics by node property; find the failure slice. Deliverable:
  error-analysis note.
- **Viz** — reuse `checklist.js` + a slice-metric bar chart.
- **Bridge** — the error-analysis habit for Y4 L137 (where GNN fails vs FE); callback Y2 L069.

### 115 · Graph ML design patterns — *— (synthesis unit)*
- **Skill** — apply the reusable "encoder → message passing → head" pattern and choose components per task
  type (node/link/graph, homogeneous/hetero/temporal).
- **Teach** — the design-pattern catalog, component-choice heuristics; the pattern that becomes the RDL stack.
- **Lab** — Tier C · crucial fragment: given three task specs, write the component choices. Deliverable: the
  design table.
- **Viz** — reuse `rdl-stack-viz.js` (from Y2 L076).
- **Bridge** — callback Y2 L076; the direct template for Y4 L131 RDL stack.

### 116 · Debug GNN training — *— (debugging unit)*
- **Skill** — diagnose the common GNN failure modes (over-smoothing, exploding/vanishing, bad sampling,
  label leakage) from symptoms.
- **Teach** — a symptom→cause table (flat loss, collapsed embeddings, train/val gap), using the
  `diagnosing-bugs` skill loop.
- **Lab** — Tier B · crucial fragment: fix a provided broken GNN training script. Deliverable: the bug found
  + fixed curve.
- **Viz** — reuse `oversmoothing-viz.js` + loss-curve readout.
- **Bridge** — callback L085; the debugging discipline for every Y4/Y5 reproduction.

### 117 · RDL bridge lecture — *Fey et al. 2024 full paper (PMLR v235), ★*
- **Skill** — read the RDL position paper end to end and walk through relational-entity-graph (REG)
  construction from a database.
- **Teach** — the full REG blueprint (nodes=rows, edges=PK/FK, node features via row encoders, temporal +
  heterogeneous), the end-to-end learning argument vs manual FE.
- **Lab** — Tier C · crucial fragment: construct a REG by hand from a small database (nodes, typed edges,
  timestamps, encoder assignment). Deliverable: the complete REG spec.
- **Viz** — reuse `hetero-graph-viz.js` + `rdl-stack-viz.js` + `flatten-loss-viz.js`.
- **Bridge** — the capstone bridge into Year 4 (everything Y3 built now targets databases); callback L096/
  L109/Y1 L035; forward to Y4 L122.

### 118 · Cvitkovic 2019 relational DL — *Cvitkovic 2019, `2002.02046`*
- **Skill** — situate RDL historically: the pre-RelBench GNN-on-RDB approach and what it got right/missing.
- **Teach** — Cvitkovic's method (GNN over DB, beat some AutoFE), its limits (no benchmark, no temporal
  standard); the line to Fey 2024.
- **Lab** — Tier C · crucial fragment: compare Cvitkovic's graph construction to Fey's REG. Deliverable: a
  diff table (what changed and why).
- **Viz** — reuse `hetero-graph-viz.js`.
- **Bridge** — historical grounding so the thesis isn't ahistorical; callback L117; forward to Y4 L121.

### 119 · Year 3 synthesis — *— (writing unit)*
- **Skill** — write "graphs vs flat tables": what message passing represents that aggregation cannot, with
  Y3 evidence.
- **Teach** — synthesis: MPNN = learned aggregation (L081), expressiveness (L088), hetero typing (L091),
  temporal correctness (L104) → the relational case.
- **Lab** — writing deliverable: one page tying Y3 back to the Y1 L035 flatten-loss argument.
- **Viz** — reuse `message-passing-viz.js` + `flatten-loss-viz.js`.
- **Bridge** — connects Year 1's thesis seed to Year 4's payoff; exit-exam rehearsal.

### 120 · **Year 3 exit exam** — *all Y3 papers · Deliverable-based*
- **Deliverable** — build a **heterogeneous + temporal GNN pipeline in PyG** (correct hetero mini-batching,
  time-respecting splits, leak audit) and construct a toy REG by hand; read Fey 2024 completely.
- **Exit criterion (from CURRICULUM)** — working hetero+temporal GNN pipeline; Fey 2024 read fully.
- **Bridge** — gate to Year 4. Artifacts: the PyG graph harness (extends `relkit`) and the hand-built REG.

**Optional (◆, after exit):** over-smoothing `1801.07606`, DropEdge `1907.10903`, PNA `2004.05718`,
over-squashing `2006.05205`, curvature `2111.14522` (depth pathologies that predict Y4 REG failures);
Graphormer `2106.05234`, GraphGPS `2205.12454` (conceptual parents of RelGT).
