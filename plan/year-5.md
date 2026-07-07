# Year 5 — Foundation Relational Models · Lesson Decomposition

**Year goal:** understand pre-training, in-context relational learning, and cross-database
generalization; reproduce one **open** relational FM baseline; and end with a ranked, evidence-backed
research proposal. Research-dominated — units are milestones spanning weeks. This is the fastest-moving
frontier in the whole curriculum: run `curriculum-currency` before **every** quarter, because new SOTA
and failure-mode papers land monthly.

Three synthesis/frontier lessons carry `b` suffixes from `CURRICULUM.md` (165b, 166b, 170b), and the
range `192–194 · Reproduce best open FM baseline` is decomposed into three lessons below.

---

## Q1 · Foundation model concepts (161–170b)

**Papers (chronological):** Bommasani 2021 (FM definition) · Zahradník 2023 `2305.15321` ★ · Griffin
`2505.05568` ★ · Relational Transformer (RT) `2510.06377` ★ · KumoRFM v1 (2025 tech report) · OpenRFM
`2606.04320` ★ · RDB-PFN `2603.03805` ★ · RDBLearn `2602.13697` ★ · RDL survey `2506.16654` (§FM).

### 161 · What is a foundation model — *Bommasani 2021*
- **Skill** — define a foundation model (pretrain on broad data → adapt to many tasks) and scope what
  "foundation model" means for *relational* data specifically.
- **Teach** — emergence/homogenization, pretraining→adaptation, transfer/zero-/few-shot; the relational
  analogue (one model, any database, one forward pass).
- **Lab** — Tier C · crucial fragment: write the relational-FM scope definition (inputs, adaptation modes,
  success criteria). Deliverable: the scoping note.
- **Viz** — reuse `arch-family-viz.js` (FM branch).
- **Bridge** — frames the year; callback Y2 L061 PFN (amortized inference); forward to L162.

### 162 · The relational FM vision — *Zahradník 2023, ★ `2305.15321`*
- **Skill** — articulate the vision: learn from the *full* relational structure (not one table) and scale to
  real-world DB sizes; name the opportunities and obstacles.
- **Teach** — why single-table representation learning ignores neighbor tables, the scale barrier, the LM+GNN
  pretraining sketch, the "FMs for DBs like text/images have" ambition.
- **Lab** — Tier C · crucial fragment: extend the Y4 L159 brief into a full obstacles/opportunities map.
  Deliverable: the vision map.
- **Viz** — reuse `rdl-stack-viz.js` + `arch-family-viz.js`.
- **Bridge** — callback Y4 L159; the conceptual spine of the year; forward to L163/L168.

### 163 · LM encoders for rows — *Zahradník 2023 §4*
- **Skill** — compare text-serialized vs typed-column row encoding and explain the trade-off (LM
  flexibility vs typed-encoder precision).
- **Teach** — serialize-a-row-as-text vs stype encoders (callback Y4 L125/L125b), where LM encoders help
  (text/high-cardinality) and where they add noise; the thesis's rejection of text-only (MISSION out-of-scope).
- **Lab** — Tier B · crucial fragment: encode the same rows via text-serialization vs typed encoder; compare
  downstream. Deliverable: encoder-comparison table.
- **Viz** — reuse `tokenizer-viz.js` + `cell-graph-viz.js`.
- **Bridge** — callback Y2 L074 CARTE / Y4 L125b; guards the "text ≠ relational" boundary; forward to L164.

### 164 · Griffin — graph-centric RDB FM — *Griffin `2505.05568`, ★ (ICML 2025)*
- **Skill** — explain Griffin's unified data encoder + task decoder, its cross-attention module and novel
  aggregator, and its single-table + RDB pretraining.
- **Teach** — one model for diverse tasks (vs per-task models), enhanced MPNN + cross-attention, pretraining
  on 150M+ nodes, low-data strength + transferability; open code.
- **Lab** — Tier B · crucial fragment (paper-repro): run Griffin's encoder/decoder on a RelBench task;
  probe low-data transfer. Deliverable: Griffin vs individually-trained model.
- **Viz** — reuse `rdl-stack-viz.js` (unified encoder/decoder) + `tokenizer-viz.js`.
- **Bridge** — the first open graph-centric RDB FM; callback Y4 L131 stack; forward to L168/L183.

### 165 · KumoRFM — in-context relational learner — *KumoRFM v1 tech report (2025)*
- **Skill** — describe KumoRFM's in-context, few-shot relational learning and treat it as the *proprietary
  upper bound* (not a reproduction target).
- **Teach** — in-context relational prediction across connected tables without flattening, few-shot task
  adaptation, why proprietary numbers set a ceiling but can't ground the thesis.
- **Lab** — Tier C · crucial fragment: read the tech report; extract the claimed capabilities + the
  reproduction gap. Deliverable: capability/ceiling note.
- **Viz** — reuse `retrieval-viz.js` (in-context examples).
- **Bridge** — the commercial ceiling; callback Y2 L061 ICL; forward to L165b (open counterpart), L191 (v2).

### 165b · OpenRFM — open relational ICL — *OpenRFM `2606.04320`, ★*
- **Skill** — explain the two diagnoses of relational ICL and OpenRFM's fixes: (model) relation-level ICL
  fails under sparse label-cell coverage → a **dual-stage ICL** that adds a batch-level ICL layer lifted
  from a tabular FM; (data) synthetic-only vs in-distribution pretraining → a homophily-aware synthetic +
  continual real-data mixture with prototype regularization.
- **Teach** — the Relational Transformer (RT) backbone it dissects (zero-shot, cell tokenization, masked
  pretraining, relational attention over rows/columns/PK-FK), the kernel-regression view of ICL failure,
  ~30% over RT and surpassing KumoRFM v1 on many tasks; open weights.
- **Lab** — Tier B · crucial fragment (paper-repro): run RT zero-shot, then add the batch-level ICL layer;
  measure the lift. Deliverable: RT vs OpenRFM table on a task subset.
- **Viz** — reuse `retrieval-viz.js` (dual-stage) + `tokenizer-viz.js` (cell tokens).
- **Bridge** — the best *open* relational FM (the real thesis reproduction target); callback Y2 L066 (tabular
  ICL lifted here); forward to L178/L192–194.

### 166 · RDB-PFN — synthetic-prior relational FM — *RDB-PFN `2603.03805`, ★*
- **Skill** — explain training a relational FM **purely on synthetic data** via a Relational Prior Generator
  (SCMs → infinite diverse RDBs) and genuine in-context adaptation.
- **Teach** — the data-scarcity problem for RDBs (private, scarce, heterogeneous), the PFN analogy extended
  to relations (callback Y2 L061/L063), 2M+ synthetic tasks, strong few-shot on 19 real tasks vs single-table
  TFMs on the same DFS-linearized inputs; open code.
- **Lab** — Tier B · crucial fragment (paper-repro): sample from a relational prior generator + run RDB-PFN
  few-shot on a task. Deliverable: RDB-PFN vs single-table-TFM-on-DFS table.
- **Viz** — reuse `pfn-prior-viz.js` (relational prior) + `retrieval-viz.js`.
- **Bridge** — the synthetic-pretraining paradigm; callback Y2 L061–L063; contrasts with L165b (real-data) and
  L166b (training-free); forward to L170b synthesis.

### 166b · RDBLearn — training-free relational ICL — *RDBLearn `2602.13697`, ★ (ICML 2026)*
- **Skill** — build the training-free RDB encoder: compress variable-sized RDB neighborhoods into fixed-length
  ICL samples **within high-dimensional shared-unit columns** (not across heterogeneous columns), then pair
  with an existing single-table ICL FM — no training/fine-tuning.
- **Teach** — the theoretical+empirical case for constraining compression within-column, why non-trainable
  encoders don't lose expressiveness, scalable SQL primitives for the encoder stage; DFS + TabICL flavor;
  robust out-of-the-box on unseen datasets.
- **Lab** — Tier B · crucial fragment (paper-repro): implement the within-column compression + pair with a
  single-table ICL model. Deliverable: RDBLearn zero-training result on an unseen DB.
- **Viz** — reuse `retrieval-viz.js` + `tokenizer-viz.js` (within- vs across-column compression).
- **Bridge** — the "no need to train your RDB FM" paradigm; callback Y2 L066 TabICL; the cheapest thesis
  baseline; forward to L178 (three-paradigm comparison).

### 167 · Tabular FM → relational FM transfer — *TabPFN v2 + TabICL recap*
- **Skill** — identify precisely what carries over from tabular ICL (Y2) to the relational setting and what
  does not (cross-table structure, PK/FK, temporal joins).
- **Teach** — the transfer map (row encoding + ICL mechanics carry; graph structure + temporal causality are
  new); why RDBLearn/OpenRFM *reuse* tabular FMs; Molnar's tabular-FM framing.
- **Lab** — Tier C · crucial fragment: build the carries-over / new table grounded in L165b/L166/L166b.
  Deliverable: the transfer map.
- **Viz** — reuse `arch-family-viz.js` (tabular→relational arrows).
- **Bridge** — connects Year 2 to Year 5; callback Y2 L064/L066; forward to L170b.

### 168 · Cross-database generalization — *Griffin + RDB-PFN experiments*
- **Skill** — evaluate zero-/few-shot performance on an *unseen schema* and explain what enables cross-DB
  transfer (schema-invariant encoding, shared priors).
- **Teach** — the cross-DB eval protocol, schema invariance (callback Y4 L125b), where transfer succeeds/fails.
- **Lab** — Tier B · crucial fragment: evaluate a pretrained FM on a held-out database. Deliverable:
  cross-DB transfer table.
- **Viz** — reuse `hetero-graph-viz.js` (source vs unseen schema).
- **Bridge** — the defining FM capability; callback Y4 L139; forward to L175 zero-shot eval.

### 169 · Scaling laws & open questions — *RDL survey `2506.16654` §FM*
- **Skill** — state what is known vs unknown about scaling relational FMs (data, params, schema diversity)
  and where the research gaps are.
- **Teach** — the survey's FM-convergence argument, scaling unknowns for relational data, open questions
  ranked.
- **Lab** — Tier C · crucial fragment: build the known/unknown table for relational-FM scaling. Deliverable:
  the scaling-gap map.
- **Viz** — reuse `arch-family-viz.js`.
- **Bridge** — feeds the Q3 research-gap doc; callback Y4 L147; forward to L189/L198.

### 170 · **Q1 checkpoint** — *Zahradník + Griffin + RDB-PFN + RDBLearn · Deliverable-based*
- **Deliverable** — a written **FM design doc** comparing the paradigms (graph-native pretraining vs
  synthetic-prior vs training-free ICL) on data, compute, transfer, and thesis-fit.
- **Bridge** — consolidates the concepts before building pipelines; callback L164–L166b.

### 170b · Three relational FM paradigms — *synthesis lecture, ★*
- **Skill** — cleanly separate the three paradigms — **graph-native ICL / pretraining** (Griffin, RT/OpenRFM),
  **synthetic-prior pretraining** (RDB-PFN), **training-free ICL** (RDBLearn) — and state when each wins/breaks.
- **Teach** — a comparison matrix (data need, compute, cold-start, cross-DB transfer, openness); the
  when-it-wins/when-it-breaks note per paradigm (extends NOTES skim notes).
- **Lab** — Tier C · crucial fragment: complete the three-paradigm matrix with evidence from L164–L166b.
  Deliverable: the paradigm matrix.
- **Viz** — reuse `arch-family-viz.js` (three FM branches).
- **Bridge** — the organizing frame for the rest of the year and the Y6 baseline suite; callback L165b/L166/L166b.

---

## Q2 · Building pre-training pipelines (171–180)

### 171 · Corpus of databases — *— (data unit)*
- **Skill** — assemble/curate a multi-database pretraining corpus (RelBench multi-DB + ReDeLEx 70+ DBs) with
  schema diversity and dedup.
- **Teach** — corpus curation for relational pretraining, schema-diversity metrics, licensing/privacy.
- **Lab** — Tier B · crucial fragment: build a small multi-DB corpus manifest. Deliverable: the corpus spec.
- **Viz** — reuse `hetero-graph-viz.js` (schema variety).
- **Bridge** — callback Y4 L139 multi-domain; forward to L172/L173.

### 172 · Schema tokenization — *— (encoding unit)*
- **Skill** — tokenize schema + cells into a model-ingestible stream (column types, table names, PK/FK,
  values) — the RT/KumoRFM-2 style cell tokenization.
- **Teach** — cell tokens with table/column metadata (callback L165b RT), schema-as-tokens, handling unseen
  columns via global statistics (callback Y4 L125b).
- **Lab** — Tier B · crucial fragment: implement the cell+metadata tokenizer for the corpus. Deliverable:
  tokenized DB sample.
- **Viz** — reuse `tokenizer-viz.js` (cell + metadata) + `cell-graph-viz.js`.
- **Bridge** — the input representation for pretraining; callback L165b/L166b; forward to L173.

### 173 · Multi-task pre-training — *— (objective unit)*
- **Skill** — design a multi-task pretraining objective (masked cell/token prediction across the four
  RelBench-v2/KumoRFM axes: row, column, FK, cross-sample) and a combined loss.
- **Teach** — masked token prediction on relational data (callback Y2 L071 VIME masking), the four
  pretraining axes, loss balancing.
- **Lab** — Tier B · crucial fragment: implement masked-cell pretraining on the tokenized corpus.
  Deliverable: a pretraining run + loss curves.
- **Viz** — reuse `mask-pretrain-viz.js` (relational masking).
- **Bridge** — callback Y2 L071; the pretraining engine; forward to L174 adaptation.

### 174 · Fine-tuning protocol — *— (adaptation unit)*
- **Skill** — compare freeze vs full fine-tune vs adapter fine-tuning of a pretrained relational encoder and
  choose per data budget.
- **Teach** — adaptation modes, catastrophic forgetting, sample efficiency (RT fine-tuning → SOTA), when to
  freeze.
- **Lab** — Tier B · crucial fragment: fine-tune a pretrained encoder three ways on one task. Deliverable:
  adaptation-mode comparison.
- **Viz** — reuse `rdl-stack-viz.js` (frozen vs tuned blocks).
- **Bridge** — callback L165b RT fine-tuning; forward to L180 checkpoint.

### 175 · Evaluation: zero-shot — *— (eval unit)*
- **Skill** — evaluate a pretrained FM zero-shot on an unseen-database task with a leak-free protocol.
- **Teach** — zero-shot eval discipline (no target-DB training, temporal correctness), reporting vs
  supervised ceiling (RT ≈93% of supervised AUROC).
- **Lab** — Tier B · crucial fragment: zero-shot eval on a held-out DB task. Deliverable: zero-shot vs
  supervised table.
- **Viz** — reuse `checklist.js` (zero-shot protocol).
- **Bridge** — callback L168; forward to L176 few-shot.

### 176 · Evaluation: few-shot ICL — *— (eval unit)*
- **Skill** — measure few-shot in-context performance as a function of k labeled examples and characterize
  the ICL scaling curve.
- **Teach** — few-shot ICL protocol, k-vs-accuracy, label-scarcity effects (callback L165b relation-level
  scarcity), cold start.
- **Lab** — Tier B · crucial fragment: sweep k; plot the few-shot curve. Deliverable: k-vs-accuracy curve.
- **Viz** — reuse `retrieval-viz.js` (k context examples).
- **Bridge** — callback Y2 L062 TabPFN few-shot; forward to L179 failure modes.

### 177 · Compute budget realism — *— (systems unit)*
- **Skill** — estimate and honestly report the compute needed for relational pretraining vs fine-tuning vs
  training-free ICL, and choose a feasible path at baseline vs extended sessions.
- **Teach** — the compute ladder (pretrain from scratch ≫ fine-tune ≫ training-free), why RDBLearn/ICL are
  the realistic solo-researcher path, honest budgeting.
- **Lab** — Tier C · crucial fragment: build a compute-vs-capability table for each paradigm. Deliverable:
  the feasibility ledger.
- **Viz** — reuse `arch-family-viz.js` (annotated with compute).
- **Bridge** — grounds the Y6 method choice in reality; callback L170b/Y4 L134; forward to L192–194.

### 178 · Compare FM vs tuned GNN vs RDBLearn — *RDBLearn + RelGNN, ★*
- **Skill** — run a fair, same-budget three-way comparison — supervised RelGNN vs a pretrained FM vs
  training-free RDBLearn — on the same tasks.
- **Teach** — the three-paradigm bake-off (callback L170b), matched budget/splits, where each paradigm wins.
- **Lab** — Tier B · crucial fragment: matched RelGNN vs FM vs RDBLearn on ≥2 tasks. Deliverable: the
  three-paradigm results table.
- **Viz** — reuse `arch-family-viz.js` + `checklist.js`.
- **Bridge** — the empirical core of the year; callback Y4 L143 RelGNN, L166b RDBLearn; forward to Y6 baselines.

### 179 · Failure modes — *— (failure unit)*
- **Skill** — catalog relational-FM failure modes (schema shift, cold start, sparse labels, rule-governed
  labels) and tie them to the Y2 barriers.
- **Teach** — schema-shift + cold-start failures, the operational barrier (callback Y2 L069b OTT), sparse
  label coverage (callback L165b), when *not* to use an FM.
- **Lab** — Tier B/C · crucial fragment: construct a case that breaks the FM (cold start or rule-governed).
  Deliverable: the failure case + explanation.
- **Viz** — reuse `ott-viz.js` (from Y2 L069b) + `checklist.js`.
- **Bridge** — the honesty guardrail for the thesis; callback Y2 L069b/L069; forward to L195 stress-test.

### 180 · **Q2 checkpoint** — *Deliverable-based*
- **Deliverable** — **fine-tune a public encoder on 1 database** end to end (pretraining-style objective →
  adaptation → leak-free eval), reporting zero-shot and fine-tuned numbers.
- **Bridge** — proves the pipeline skills before the frontier survey; callback L172–L178.

---

## Q3 · Research frontier mapping (181–190)

**Papers:** RelBench v2 `2602.12606` ★ · RelGT-AC `2606.03040` ★ · GelGT `2605.15575` ◆ · RDL survey
`2506.16654` · plus the currency-rule additions of the quarter.

### 181 · RelBench v2 & autocomplete tasks — *RelBench v2 `2602.12606` ★ · RelGT-AC `2606.03040` ★*
- **Skill** — use RelBench v2 (11 DBs, 22M+ rows, external-benchmark integration) and set up **autocomplete
  tasks** (predict a missing in-table attribute value under temporal constraints).
- **Teach** — v2's new datasets + TGB/ReDeLEx/4DBInfer integration, the autocomplete task class vs SQL-query
  forecasting, RelGT-AC's fixes (column masking to avoid trivial solutions, unified head, TF-IDF text encoder).
- **Lab** — Tier B · crucial fragment: run an autocomplete task with column masking + a text encoder.
  Deliverable: autocomplete result vs the GraphSAGE baseline.
- **Viz** — reuse `tokenizer-viz.js` (masked target column) + `hetero-graph-viz.js`.
- **Bridge** — the current benchmark surface for the thesis; callback Y4 L136 leaderboard; forward to Y6 submission.

### 182 · RDB-PFN + composite message passing — *RDB-PFN × RelGNN (hypothesis unit)*
- **Skill** — generate a concrete research hypothesis by combining a synthetic-prior FM (RDB-PFN) with
  atomic-route composite message passing (RelGNN).
- **Teach** — hypothesis generation from architecture composition, feasibility screening, expected-signal
  reasoning.
- **Lab** — Tier C · crucial fragment: write the hypothesis + a minimal test design. Deliverable: the
  hypothesis brief.
- **Viz** — reuse `atomic-route-viz.js` + `pfn-prior-viz.js`.
- **Bridge** — callback Y4 L141/L166; feeds the Y6 direction; forward to L189/L198.

### 183 · Graph-Transformer + pre-training — *RelGT × Griffin (gap unit)*
- **Skill** — identify the architecture-research gap: pretraining a *graph-transformer* relational model
  (RelGT tokenization + Griffin-style pretraining).
- **Teach** — why the gap exists (RelGT is supervised, Griffin is MPNN-based), what a pretrained RelGT could
  buy, the risks.
- **Lab** — Tier C · crucial fragment: write the gap analysis + a probe experiment. Deliverable: the gap brief.
- **Viz** — reuse `tokenizer-viz.js` + `arch-family-viz.js`.
- **Bridge** — callback Y4 L145/L146, L164; a candidate Y6 direction; forward to L189.

### 184 · Latest relational Graph-Transformers — *GelGT `2605.15575` ◆*
- **Skill** — explain GelGT's fixes for long-range dependencies — structure-semantic collaborative sampling +
  a Gaussian graph-attention bias for temporal dynamics — and where information decay bites vanilla RDL MP.
- **Teach** — the long-range/information-decay problem (callback Y3 over-squashing ◆), Gaussian bias for
  time, structure-semantic sampling; up to ~13.8% gains.
- **Lab** — Tier B · crucial fragment: add a Gaussian temporal bias to a RelGT-style attention on one task.
  Deliverable: long-range gain table.
- **Viz** — reuse `temporal-embed-viz.js` + `atomic-route-viz.js`.
- **Bridge** — the frontier of relational graph-transformers; callback Y3 over-squashing, L145; forward to L189.

### 185 · Causal & relational data — *— (causality unit)*
- **Skill** — distinguish predictive correlation from a deployable/causal strategy on relational data and
  flag when a strong benchmark number won't transfer to a decision.
- **Teach** — correlation vs intervention, confounding via relational structure, why leaderboard wins ≠
  deployable policy.
- **Lab** — Tier C · crucial fragment: construct a task where a high-AUROC model encodes a non-causal
  shortcut. Deliverable: the shortcut demonstration.
- **Viz** — reuse `leakage-viz.js` (confounding path).
- **Bridge** — matures the thesis beyond benchmark numbers; callback Y1 leakage, Y2 L069b; forward to L195.

### 186 · Production constraints — *Huyen (Designing ML Systems)*
- **Skill** — enumerate production constraints for relational models (latency, feature/data freshness,
  monitoring, retraining) and their effect on architecture choice.
- **Teach** — serving an RDL/FM model, freshness of the REG, drift monitoring (callback Y2 L068), the
  training-free-ICL serving advantage.
- **Lab** — Tier C · crucial fragment: write a serving-constraints spec for one paradigm. Deliverable: the
  production checklist.
- **Viz** — reuse `checklist.js`.
- **Bridge** — grounds the thesis in deployability; callback L166b/L177; forward to Y6 L236 pilot.

### 187 · Ethics & privacy on REG — *— (ethics unit)*
- **Skill** — analyze node-level privacy and leakage risks in a REG (linkage across tables, membership
  inference) and state mitigations.
- **Teach** — relational data re-identification, node/edge-level privacy, leakage via joins, mitigation
  options.
- **Lab** — Tier C · crucial fragment: audit a REG task for a privacy/leakage risk. Deliverable: the
  privacy-risk note.
- **Viz** — reuse `leakage-viz.js` (cross-table linkage).
- **Bridge** — responsible-research grounding; callback Y4 L139 healthcare; forward to Y6 limitations.

### 188 · Systematic literature tracking — *— (currency unit)*
- **Skill** — set up a durable literature-tracking system (arXiv alerts, RelBench/TabArena leaderboards, a
  paper log) operationalizing the currency rule.
- **Teach** — the `curriculum-currency` workflow, what qualifies as add-worthy (SOTA / failure mode /
  baseline), maintaining the paper log.
- **Lab** — Tier C · crucial fragment: configure alerts + log the last quarter's new papers. Deliverable:
  the tracking system + updated log.
- **Viz** — none (workflow artifact).
- **Bridge** — keeps Y5–Y6 current on a monthly-moving frontier; callback the `curriculum-currency` skill.

### 189 · Identify 3 open problems — *RDL survey `2506.16654`*
- **Skill** — select and rank three tractable open problems by expected impact × feasibility for a solo
  researcher at baseline compute.
- **Teach** — problem selection, tractability estimation (callback L177 compute), impact framing.
- **Lab** — Tier C · crucial fragment: rank candidate problems with an explicit rubric. Deliverable: the
  ranked shortlist.
- **Viz** — reuse `checklist.js` (tractability rubric).
- **Bridge** — the direct input to the Q3 gap doc + Y6 hypothesis; callback L169/L182/L183; forward to L190.

### 190 · **Q3 checkpoint** — *Deliverable-based*
- **Deliverable** — a **5-page research-gap document** with ≥3 tractable open problems, each with related
  work, a feasibility estimate, and a candidate approach.
- **Bridge** — the seed of the Year-6 project; callback L181–L189.

---

## Q4 · Year 5 synthesis (191–200)

The `CURRICULUM.md` range `192–194 · Reproduce best open FM baseline` is decomposed into three lessons below.

### 191 · KumoRFM-2 SOTA tracking — *KumoRFM-2 `2604.12596`, ★*
- **Skill** — record the current RelBench v1+v2 SOTA (KumoRFM-2) and quantify the proprietary-vs-open gap the
  thesis must reason about.
- **Teach** — KumoRFM-2's four pretraining axes + early task injection, first few-shot FM to beat *supervised*
  on common tasks, billion-scale, cold-start/noise robustness; why it's a ceiling, not a baseline.
- **Lab** — Tier C · crucial fragment: tabulate KumoRFM-2 vs best-open (OpenRFM/RDBLearn) numbers.
  Deliverable: the SOTA-vs-open gap table.
- **Viz** — reuse `arch-family-viz.js` (proprietary vs open) + `checklist.js`.
- **Bridge** — sets the honest ceiling for Y6 claims; callback L165/L165b; forward to L195.

### 192 · Open FM reproduction — setup & data — *OpenRFM or RDBLearn, ★*
- **Skill** — stand up the chosen open FM's code/weights + data pipeline and reproduce its reported number on
  one task.
- **Teach** — the reproduction contract at FM scale (env, weights, data, seeds), sanity-checking against the
  paper.
- **Lab** — Tier B · crucial fragment (paper-repro): reproduce one reported task number. Deliverable: matched
  number + env note.
- **Viz** — reuse `checklist.js`.
- **Bridge** — begins the exit-required reproduction; callback Y4 L143; forward to L193.

### 193 · Open FM reproduction — full task set — *OpenRFM or RDBLearn, ★*
- **Skill** — extend the reproduction across the paper's task set and report aggregate + per-task results with
  variance.
- **Teach** — multi-task reproduction, variance/aggregation, honest documentation of any task that won't
  reproduce.
- **Lab** — Tier B · crucial fragment: run the full task set. Deliverable: the reproduction results table.
- **Viz** — reuse `checklist.js` + a results table.
- **Bridge** — the core exit artifact; callback L178; forward to L194.

### 194 · Open FM reproduction — analysis & report — *OpenRFM or RDBLearn, ★*
- **Skill** — analyze *why* the FM wins/loses per task (label coverage, schema, cold start) and write the
  reproduction report.
- **Teach** — mechanistic analysis (callback L165b kernel-regression view / L179 failures), report structure.
- **Lab** — Tier B · crucial fragment: per-task win/lose attribution + report. Deliverable: the reproduction
  report.
- **Viz** — reuse `retrieval-viz.js` + `checklist.js`.
- **Bridge** — completes the exit reproduction; callback L192/L193; forward to L200.

### 195 · Thesis stress-test — *— (falsification unit)*
- **Skill** — actively try to *falsify* the thesis: find the strongest evidence that relational FMs are *not*
  undervalued, using your own reproductions.
- **Teach** — steel-manning the null (trees/tabular-FMs suffice), the OTT barrier (Y2 L069b), non-IID gaps
  (Y2 L056b), where RDL underperforms (Y4 L149).
- **Lab** — Tier B/C · crucial fragment: assemble the strongest anti-thesis case with evidence. Deliverable:
  the falsification brief.
- **Viz** — reuse `checklist.js` + `ott-viz.js`.
- **Bridge** — the intellectual honesty the Y6 project requires; callback Y2 L069b/L056b, Y4 L149; forward to L198.

### 196 · Community engagement — *RelBench mailing list / forum*
- **Skill** — ask one well-formed technical question in a high-reputation relational-ML community and
  incorporate the feedback.
- **Teach** — finding the community (RelBench discussion, PyG forum, relevant labs), asking answerable
  questions, wisdom from practitioners (teach-skill: acquiring wisdom).
- **Lab** — real-world action · crucial fragment: post the question + log responses. Deliverable: the
  thread + what you learned.
- **Viz** — none (community action).
- **Bridge** — the wisdom leg of the teach philosophy; callback Y4 L157; forward to Y6 external review.

### 197 · Year 5 essay — *— (writing unit)*
- **Skill** — write the FM landscape map: the three paradigms, the open/proprietary gap, and where the
  undervaluation thesis stands after Year 5.
- **Teach** — synthesis of L170b + L191 + L195; a defensible mid-thesis position.
- **Lab** — writing deliverable: the landscape-map essay.
- **Viz** — reuse `arch-family-viz.js`.
- **Bridge** — exit rehearsal; callback L170b/L191/L195.

### 198 · Propose 3 novel directions — *— (proposal unit)*
- **Skill** — write three ranked, falsifiable research proposals (hypothesis, baselines, expected result,
  feasibility) grounded in the gap doc.
- **Teach** — proposal structure, falsifiability, baseline suite pre-commitment (callback L178), feasibility
  vs compute.
- **Lab** — Tier C · crucial fragment: three proposals with a ranking rubric. Deliverable: the ranked proposals.
- **Viz** — reuse `checklist.js`.
- **Bridge** — the direct input to Year 6; callback L190/L195; forward to L199.

### 199 · Select primary direction — *— (decision unit)*
- **Skill** — commit to one hypothesis to pursue in Year 6 with an explicit go/no-go rationale.
- **Teach** — decision under uncertainty, opportunity cost, the pre-registration mindset (forward Y6 L203).
- **Lab** — Tier C · crucial fragment: the selection memo (why this, not the others). Deliverable: the chosen
  hypothesis.
- **Viz** — reuse `checklist.js`.
- **Bridge** — locks the Year-6 project; callback L198; forward to Y6 L201.

### 200 · **Year 5 exit exam** — *all Y5 papers · Deliverable-based*
- **Deliverable** — a **written research proposal** backed by RelBench experiments **+ one reproduced
  FM-related baseline** (the L192–194 reproduction), with the three-paradigm framing and honest limitations.
- **Exit criterion (from CURRICULUM)** — research proposal with evidence from RelBench experiments; one
  reproduced FM baseline.
- **Bridge** — gate to Year 6. Artifacts: the reproduced open FM, the gap doc, the selected hypothesis,
  the baseline suite.

**Optional (◆, after exit):** LM+GNN hybrid FM `2605.16085` (resource-efficient path); KumoRFM v1 PDF
(proprietary; use KumoRFM-2 for reproducible numbers); Molnar *Tabular Foundation Models* book (L167
transfer). Beyond these, prefer the **currency rule** — this frontier turns over monthly.
