# Year 5 → 6 bridge: influential architectures and the current frontier

**Planned B01–B24 · research cutoff 2026-09-26.** After L200, before L201. [Curriculum](../CURRICULUM.md#research-bridge) · [Influence and coverage audit](./research-influence-audit-2026-09.md).

The audit distinguishes established families, current baseline updates, coverage-critical mechanisms and exploratory ideas. Core means required for this mission; it does not label every recent paper equally influential. Existing lessons are prerequisites: a bridge revisit requires a new comparative artifact, not repetition of the entire earlier lesson.

## Teaching contract

Pass the Year 5 exit gate first. Each unit follows **closed-book retrieval → primary-source input → exercise → written defense**. Use the CHECK as feedback, repair the explanation with a fresh example, and revisit after 1, 7 and 30 days. Ask the teaching agent follow-up questions about unclear assumptions. Do not mark agent-prepared work as learner mastery.

Core route: **B01–B07 → B10–B14 → B18–B19 → B23–B24** (16 units). Electives: B08–B09, B15–B17, B20–B22 (8 units), selected after core requirements. The plan budgets **70–110 learner hours for core, including one bounded reproduction; 16–24 additional hours for all electives**. Total 86–134 hours, roughly 12–19 weeks at one hour/day. Extend the original calendar or reallocate optional time explicitly; do not cut Year 6 research or exit gates. Compute may run unattended and is a separate budget.

Every core defense must explain the mechanism, its closest competing explanation and one limitation. For future HTML/lab authoring, reuse shared assets, show model-specific inputs/operations/shapes/training/inference, provide visible implementation and TODO/CHECK/EXIT tasks, and link primary sources. Visual suggestions below describe future work, not delivered widgets.

Exercises are course-designed unless B23 pins a named published experiment. Record source commits, checkpoint and data hashes, environment, exact commands, temporal/label visibility, preprocessing, seeds, selection and inference budgets, metric reconstruction and deviations. Distinguish `NOT_RUN`, `NOT_CHECKED`, `INCOMPARABLE` and `CLOSE`. A model API or downloadable checkpoint does not establish open pretraining. No experiments were run for this curriculum revision; status is `PENDING_WRITTEN_DEFENSE`.

## A · Influential tabular families, updated

<a id="b01"></a>

### B01 ★ · Architecture coverage and honest comparison
- **Read:** [TabArena](https://arxiv.org/html/2506.16791v1), evaluation design; [fair RDB benchmark](https://arxiv.org/abs/2607.03659v1), protocol and hop-depth comparison; the influence audit above.
- **Retrieve / skill:** Why can two correct leaderboards disagree? Define the comparison unit.
- **Teach:** Separate trained predictors, pretrained models, feature-engineering pipelines and AutoML systems. Benchmark participation is uptake evidence, not universal superiority.
- **Exercise:** Build a family map with pretraining source, representation, adaptation, task scope and information budget. Retrieve L060/L170b before adding new families.
- **CHECK:** Include tuned trees, numerical MLPs, retrieval, synthetic/real-data ICL, semantic transfer, graph-native and flattened relational approaches. Missing families must be justified.
- **Visual / bridge:** A family-by-design matrix using `arch-family-viz.js`; freeze the comparison questions for B23.

<a id="b02"></a>

### B02 ★ · Numerical embeddings, TabM and TabPack
- **Read:** [Numerical embeddings](https://arxiv.org/abs/2203.05556), [TabM](https://arxiv.org/html/2410.24210v1), [TabPack Fig. 2 and §3](https://arxiv.org/html/2607.05380v1). Recall RealMLP/TabR; optional [ModernNCA](https://arxiv.org/abs/2407.03257).
- **Retrieve / skill:** Is a stronger MLP result caused by its representation or its ensemble? Separate contributions.
- **Teach:** Scalar-to-vector encoding and ensemble design are different axes. TabPack's heterogeneous packed members do not imply TabM-style weight sharing.
- **Exercise:** Specify embedding on/off and ensemble on/off controls with matched selection budgets. Include a tuned tree and RealMLP baseline.
- **CHECK:** Count validation-driven member selection and inference cost; do not call one training run selection-free.
- **Visual / bridge:** Embedding → member computation → selection → averaged prediction; reuse `ensemble-viz.js`. Revisit L053–054.

<a id="b03"></a>

### B03 ★ · PFN and the TabPFN generations
- **Read:** Recall L061–065; [TabPFN-2.5](https://arxiv.org/abs/2511.08667), [TabPFN-3](https://arxiv.org/abs/2605.13986), [TabPFN-3.5 v2](https://arxiv.org/abs/2609.17895v2), variant and evaluation descriptions.
- **Retrieve / skill:** What remains fixed when a pretrained model adapts in context? Separate learned prior, support data and test-time computation.
- **Teach:** Follow the family through changed scale, modalities and inference recipes. A report about Plus or Thinking does not establish the same result for the local base checkpoint.
- **Exercise:** Produce a version/variant matrix: inputs, context cap, access, benchmark split, selection and inference budget.
- **CHECK:** Do not transfer historical scores or licenses across generations; mark unavailable artifacts explicitly.
- **Visual / bridge:** Reuse `l061-pfn-viz.js`; carry a frozen variant choice into B19/B23.

<a id="b04"></a>

### B04 ★ · TabICL and scalable two-stage ICL
- **Read:** Recall L066/066b; [TabICLv2 §§3–7](https://arxiv.org/html/2602.11139v1); [release](https://github.com/soda-inria/tabicl).
- **Retrieve / skill:** Why compress features before dataset-level attention? Trace where computation is saved.
- **Teach:** Distinguish representation construction, long-context attention, prior diversity and optimization. Attribute gains through ablations rather than naming one component as the whole explanation.
- **Exercise:** Trace N rows × F features through row representations to support/query predictions. Specify a context-length stress test at fixed checkpoint and an isolated attention change.
- **CHECK:** Report memory and accuracy; ensure query labels are absent and query batching does not silently change information access.
- **Visual / bridge:** Extend `l066-tabicl-viz.js`; connect to OpenRFM's use of a tabular ICL stage in B12.

<a id="b05"></a>

### B05 ★ · TabDPT: real-data pretraining and retrieval
- **Read:** [TabDPT v3 §§3–4 and Appendix B.1](https://arxiv.org/html/2410.18164v3); [Turbo](https://arxiv.org/abs/2608.01400v1); [version history](https://github.com/layer6ai-labs/TabDPT-inference).
- **Retrieve / skill:** How can an unlabeled table generate prediction episodes? Trace the self-supervised target construction.
- **Teach:** Real-data column prediction and retrieval offer an alternative to synthetic-only training. Turbo changes the context-efficiency recipe; the repository's v1.3 is a later checkpoint release.
- **Exercise:** Remove one column as target before selecting neighboring rows; split support/query rows and draw the row-token path. Write a pretraining/test dataset-overlap audit.
- **CHECK:** Retrieval cannot use the held-out target. Original-paper and newer-checkpoint runs need separate provenance.
- **Visual / bridge:** Retrieval neighborhood → episode → row transformer; adapt `tabr-viz.js`. This fills a missing family, not a minor version lecture.

<a id="b06"></a>

### B06 ★ · Mitra: the prior is part of the model
- **Read:** [Mitra §§3–4.3](https://arxiv.org/html/2510.21204v1); [Mitra-v2 §§2–4](https://arxiv.org/html/2609.04540v1); [release artifacts](https://huggingface.co/autogluon/mitra-finetune).
- **Retrieve / skill:** Can unchanged architecture improve because the task generator changes? Isolate prior effects.
- **Teach:** Separate an outer mixture across tasks from hybrid mechanisms within one synthetic task. Also distinguish forward-only ICL from fine-tuned configurations.
- **Exercise:** Specify SCM-only, tree-only and mixed-prior arms at matched architecture/data/update budgets; record development datasets used to choose the mixture.
- **CHECK:** A benchmark used to select a prior is not an untouched test of generality. Released fine-tuning code does not prove pretraining reproducibility.
- **Visual / bridge:** Generator mixture → episodes → fixed learner → held-out tasks. Extend the L063 prior diagram; prepare B13.

<a id="b07"></a>

### B07 ★ · Semantic transfer: CARTE, ConTextTab and TabSTAR
- **Read:** Recall CARTE at L074; [ConTextTab §§3–4](https://arxiv.org/html/2506.10707v1); [TabSTAR §§3–6](https://arxiv.org/html/2505.18125v2).
- **Retrieve / skill:** When do column names add information beyond values? Separate semantics from numerical pattern learning.
- **Teach:** Compare semantics-aware table-native ICL with target-aware transfer followed by fine-tuning. These are not interchangeable adaptation regimes.
- **Exercise:** Design meaningful-name, anonymized-name and text-removed arms, preserving splits and numeric inputs. Track encoder updates and all training costs.
- **CHECK:** All competitors must receive comparable text information or the difference must be declared. Target identity is not a query's unknown target value.
- **Visual / bridge:** Two paths showing frozen/context adaptation versus gradient updates; reuse `frame-encoder-viz.js`. This repairs the semantic-model coverage gap.

<a id="b08"></a>

### B08 ◆ · LimiX: alternative structured-data objectives
- **Read:** [LimiX](https://arxiv.org/abs/2509.03505), [LimiX-2M](https://arxiv.org/abs/2606.04485v2), [LimiX-2 §2](https://arxiv.org/html/2609.17488v1); [release terms](https://github.com/limix-ldm-ai/LimiX).
- **Retrieve / skill:** Is feature reconstruction the same task as predicting a target? Trace their separate paths.
- **Teach:** Compare feature/task processing and reconstruction objectives; LimiX-2M and LimiX-2 are distinct releases.
- **Exercise / CHECK:** Draw allowed feature/target and support/query attention, then specify an objective ablation. Pin code and weights separately; do not infer causal identification from model terminology.
- **Visual / bridge:** A masked attention matrix. Current comparator, not established influence inferred from one rank claim.

<a id="b09"></a>

### B09 ◆ · Current cost frontier: TabFM, EXAONE and Nori
- **Read:** [Google TabFM release](https://www.research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/), [EXAONE §2](https://arxiv.org/html/2608.25774v1), [Nori model card](https://huggingface.co/Synthefy/Nori/blob/main/README.md).
- **Retrieve / skill:** Do fewer parameters guarantee cheaper predictions? Compare complete operating points.
- **Teach:** Use EXAONE's repeated cross-axis processing as the mechanism example; audit support size, caching, ensembling, task coverage and access for all three.
- **Exercise / CHECK:** Build a latency/memory/quality protocol on identical prediction rows. Do not compare regression-only and mixed-task aggregates or treat model cards as peer-reviewed papers.
- **Visual / bridge:** Reuse `benchmark-budget-viz.js`. These models enter B24's decision inventory even if this elective is skipped.

## B · Influential relational directions, compared

<a id="b10"></a>

### B10 ★ · Relational Transformer: cells, tasks and relational attention
- **Read:** [RT §§3–5](https://arxiv.org/html/2510.06377v1); update: [RT-J author page](https://star-project.stanford.edu/rt-j/), with its linked paper pending full-text access.
- **Retrieve / skill:** How does a cell-token model know which database relationships matter? Construct its attention masks.
- **Teach:** Trace task-table integration, typed cell tokens, row/column/FK attention and masked prediction. RT-J changes the corpus, supervision and retrieval recipe around the backbone.
- **Exercise:** Enumerate allowed attention on a three-table fixture; hide query targets and future records. Specify a schema-name ablation and distinguish unseen task from unseen database.
- **CHECK:** A sampled context limits observable evidence; “zero-shot” does not mean absence of contextual labels.
- **Visual / bridge:** Cell-level masks, not a generic row graph. This gives RT the dedicated unit missing from the original curriculum rows.

<a id="b11"></a>

### B11 ★ · RelGNN versus RelGT: supervised relational baselines
- **Read:** Recall L141–146; [RelGNN §3](https://arxiv.org/html/2502.06784v1), [RelGT §3](https://arxiv.org/html/2505.10960v1); [RelArena implementation inventory](https://github.com/PriorLabs/relarena).
- **Retrieve / skill:** Is a multi-hop graph path the same computational operation as an atomic route? Trace the difference.
- **Teach:** Contrast composite message passing with row tokens carrying relational/temporal structure. Compare mechanisms, not incompatible historical leaderboard cells.
- **Exercise:** Trace both models on the same bridge-table fixture and propose equal-information, equal-budget training arms.
- **CHECK:** Keep sampled evidence and task rows aligned. Add ContextGNN/RelGT-AC only when recommendation/autocomplete makes them applicable.
- **Visual / bridge:** Two model-specific computation paths; reuse `schema-graph-viz.js` for inputs. Update the L150 result rather than assuming perpetual SOTA.

<a id="b12"></a>

### B12 ★ · Griffin, OpenRFM and KumoRFM: adaptation mechanisms
- **Read:** Recall L164/165b/191; [Griffin](https://proceedings.mlr.press/v267/wang25da.html), [OpenRFM §§3–5](https://arxiv.org/html/2606.04320v1), [KumoRFM-2 §3](https://arxiv.org/html/2604.12596v1).
- **Retrieve / skill:** What changes between two tasks: parameters, contextual labels, or both? Classify adaptation precisely.
- **Teach:** Compare shared encoders/decoders, relational support reachability, and relation-level versus cross-example interaction. Industrial comparisons have different artifact-access boundaries.
- **Exercise:** Draw a pretraining/adaptation/inference matrix and an OpenRFM support-label corruption test on high/low reachability tasks.
- **CHECK:** Preserve observed-label visibility; a closed model is a comparator, not an upper bound. Use matched support budgets.
- **Visual / bridge:** Model-specific support-to-query paths. Carry an accessible relational FM into B23 or document its exclusion.

<a id="b13"></a>

### B13 ★ · RDB-PFN and PluRel: two roles for relational synthetic data
- **Read:** Recall L166; [RDB-PFN §§4–6 and Fig. 1](https://arxiv.org/html/2603.03805v1), [PluRel §§2–3](https://arxiv.org/html/2602.04029v1).
- **Retrieve / skill:** Does generating a relational database imply graph-native inference? Separate generator, representation and learner.
- **Teach:** Compare relational-prior generation plus linearization with synthetic databases used to pretrain RT. Schema diversity and row count are different scaling axes.
- **Exercise:** Trace schema → FK structure → values → prediction episode for both routes. Propose a held-out-schema test with matched generated-cell budget.
- **CHECK:** State generator assumptions, including acyclicity where required; do not treat a synthetic prior as a proof about arbitrary enterprise schemas.
- **Visual / bridge:** Two explicit generation-to-prediction pipelines. PluRel's reuse in later work motivates core coverage before the B20 curriculum elective.

<a id="b14"></a>

### B14 ★ · RDBLearn and TabPFN-Rel: the flattening challenge
- **Read:** [RDBLearn toolkit paper](https://arxiv.org/abs/2602.18495v1), [companion encoder analysis](https://arxiv.org/html/2602.13697v2), [RelArena/TabPFN-Rel §§1–3](https://arxiv.org/html/2608.16319v2).
- **Retrieve / skill:** Where is relational information learned versus constructed? Trace feature synthesis and the ICL backend separately.
- **Teach:** Explicitly distinguish the toolkit paper from its theoretical companion, historical backends from current versions, and local text-free from hosted text-enabled variants.
- **Exercise:** Draw query/time → visible database → aggregates → support table → prediction; specify a backbone-swap ablation with unchanged features.
- **CHECK:** Match temporal tuning and refit policies. Audit model/system entries, one-seed release evidence and omitted backends before interpreting rankings.
- **Visual / bridge:** Extend `join-flatten-viz.js`. Core baseline for B23, not a universal claim that flattening always wins.

<a id="b15"></a>

### B15 ◆ · Parameter-free encoders: limits and assumptions
- **Read:** [Parameter-Free Encoders Remain Viable](https://arxiv.org/abs/2607.05476v2).
- **Retrieve / skill:** Which labels are permitted in an encoder? State the theorem's domain.
- **Teach:** Study the label-as-input setting; do not conclude that learned encoders can never help.
- **Exercise / CHECK:** Construct observed-support/hidden-query fixtures and a leaky counterexample. Explain why each meets or violates the assumptions.
- **Visual / bridge:** Use `label-budget-viz.js`; challenge the learned component of the Year 6 proposal. Valuable counterargument, not established influence merely from recency.

<a id="b16"></a>

### B16 ◆ · AutoGrable: selecting or declining a graph
- **Read:** [AutoGrable](https://arxiv.org/abs/2608.11431v1), partition-based graph construction.
- **Retrieve / skill:** Can distinct graphs be indistinguishable to message passing? Reason with row partitions.
- **Teach:** Task alignment and block occupancy motivate construction selection and abstention.
- **Exercise / CHECK:** Enumerate candidate partitions on eight rows; separate training, validation-based construction selection and test evaluation. Pure singleton blocks alone do not prove generalization.
- **Visual / bridge:** Extend `wl-viz.js`; compare with B11's fixed schema graphs. Exploratory mechanism, now elective.

<a id="b17"></a>

### B17 ◆ · FlexTab and GTAlign: reusable representations
- **Read:** [FlexTab](https://arxiv.org/abs/2606.30336v2), [GTAlign](https://arxiv.org/abs/2607.11374v1).
- **Retrieve / skill:** What must remain invariant for a representation to transfer? Identify the adaptation boundary.
- **Teach:** Contrast target-agnostic row representations with graph-to-table alignment and community-guided episodes. Choose one paper for the exercise.
- **Exercise / CHECK:** Trace two tasks through a shared encoder, or ablate community supervision with equal support labels. Target-domain adaptation cannot be relabeled zero-shot; promised code release is not verified availability.
- **Visual / bridge:** Extend `frame-encoder-viz.js`; connect B07 semantics and B11 structural encoding. Neither paper is designated broadly influential by this audit.

## C · Stress-test the claimed gains

<a id="b18"></a>

### B18 ★ · Context-window failure and sufficient evidence
- **Read:** [Context Window Failures](https://arxiv.org/abs/2609.00460v1); revisit RT's context ablations.
- **Retrieve / skill:** Can a bounded sample preserve a target determined by all historical events? Diagnose information loss.
- **Teach:** Animus provides a specific counterexample to relying on raw sampled context; it is not an industry-wide prevalence estimate.
- **Exercise:** Compare full historical totals, truncated totals and count-corrected estimates across entity degrees. Contrast pre-aggregation with a larger context budget.
- **CHECK:** Keep future records excluded and report degree-stratified errors. Distinguish a course fixture from reproduction of Animus.
- **Visual / bridge:** Extend `temporal-visibility.js`; identify one falsification test for B24. Core by mission relevance, not demonstrated citation influence.

<a id="b19"></a>

### B19 ★ · BeyondArena, contamination and moving benchmarks
- **Read:** Recall L055/056b; [BeyondArena protocol](https://arxiv.org/html/2606.30410v1), B03/B06 current reports and TabDPT's contamination appendix.
- **Retrieve / skill:** Can a new model solve a previously reported failure without contradicting the old study? Align evidence versions.
- **Teach:** Distinguish IID, grouped and temporal splits, development-set reuse, dataset overlap and imputed missing runs.
- **Exercise:** Build a cross-paper comparability matrix, then pre-register one untouched temporal/grouped dataset comparison with identical accessible information.
- **CHECK:** Separate measured scores from imputation. Use dataset-level uncertainty; seeds do not increase the number of independent datasets.
- **Visual / bridge:** Reuse `drift-viz.js`; forward to L203/L219. No blanket “trees always win” or “TFMs solved non-IID” conclusion.

<a id="b20"></a>

### B20 ◆ · Curriculum Matters: ordering versus scale
- **Read:** [Curriculum Matters](https://arxiv.org/abs/2607.29120v1); prerequisites B06/B13.
- **Retrieve / skill:** Does fewer generated databases mean less compute? Separate data, optimization and ordering.
- **Exercise:** Specify staged/shuffled and single-table/relational arms with a shared task pool and matched updates. Report generated cells, time and stopping separately.
- **CHECK:** Do not attribute changed generators or budgets to ordering. A percentage of raw AUROC is not the same percentage of excess-over-chance signal.
- **Visual / bridge:** Reuse `growth-viz.js`. Hypothesis-generating follow-up to the influential priors, not a replacement for them.

<a id="b21"></a>

### B21 ◆ · Privacy or structural robustness
- **Read:** Choose [TabPATE](https://arxiv.org/abs/2606.31474v1) or [integrity-constrained structural attacks](https://arxiv.org/abs/2607.07089v1).
- **Retrieve / skill:** Does an empirical stress-test result establish a formal guarantee? Define the protected unit or threat model.
- **Exercise / CHECK:** For privacy, specify entity/row adjacency and teacher/query accounting; near-random attack success does not prove DP. For attacks, enumerate admissible FK edits and compare equal perturbation budgets; check integrity and temporal constraints independently.
- **Visual / bridge:** Teacher-vote release boundary or FK-edit trace. Select the option relevant to L187/L219; do not require both papers without a thesis need.

<a id="b22"></a>

### B22 ◆ · Newest mechanisms: RefineICL or tabular JEPA
- **Read:** Choose [RefineICL](https://arxiv.org/abs/2609.27679v1) or [JEPA recipe](https://arxiv.org/abs/2609.25541v1).
- **Retrieve / skill:** Can an internal mechanism change without improving downstream generalization? Design a discriminating intervention.
- **Exercise / CHECK:** Specify a controlled support-state intervention, or value-only versus JEPA at fixed compute and at convergence. Separate benchmark-informed continuation from untouched tests; one run per arm cannot characterize seed uncertainty.
- **Visual / bridge:** Attention-state trace or paired learning curves. These are explicitly exploratory September preprints, not established influential architectures.

## D · Reproduction and research handoff

<a id="b23"></a>

### B23 ★ · Reproduce one declared comparison
- **Read:** Chosen model's exact experiment section and release; [RelArena protocol](https://arxiv.org/html/2608.16319v2) or [TabArena methodology](https://arxiv.org/html/2506.16791v1).
- **Retrieve / skill:** What distinguishes a numerical match from a faithful experiment? Audit the full chain.
- **Exercise:** Choose one named published task/table cell or bounded ablation with accessible artifacts; freeze its contract before running. Candidate lanes: trained TabM/TabPack, a released TabDPT/TabICL comparison, or local RDBLearn/TabPFN-Rel versus a supervised relational baseline. Include the simpler baseline.
- **CHECK:** Preserve original protocol for reproduction; report the thesis's matched-budget comparison separately if different. Deliver source/data hashes, exact command, predictions, metric reconstruction, cost and deviation ledger.
- **Visual / bridge:** Use `exit-gates-viz.js`. A smoke test or teaching fixture leaves the reproduction gate pending; do not pretend a full pretraining run fits this budget.

<a id="b24"></a>

### B24 ★ · Defend the architecture and the thesis
- **Deliverable:** A 3–5-page proposal, B23 evidence folder, coverage map and baseline inclusion/exclusion table.
- **Required inventory:** Tuned trees + time-safe feature engineering; RealMLP/TabM/TabPack; retrieval (TabR/ModernNCA when relevant); TabPFN, TabICL, TabDPT and Mitra families; semantic transfer when text is material; RDBLearn/TabPFN-Rel; RelGNN/RelGT; accessible RT/relational FM. Audit LimiX, TabFM, EXAONE, Nori and RT-J as current candidates even if their electives were skipped. This is a decision inventory, not a requirement to train every model.
- **Falsification:** Specify a simpler-baseline win and a failure on an untouched database/task that would change the proposal. Preserve matched information and budget, and distinguish public from proprietary systems.
- **Rubric:** Protocol validity, baseline fairness, reproducibility, evidence interpretation and falsifiability score 0–2 each. Pass at 8/10 with no zero and no unresolved leakage. Missing reproduction keeps that gate pending regardless of score. Learner status stays `PENDING_WRITTEN_DEFENSE` until assessed.
- **Handoff:** L201 gets the hypothesis, L202 the source matrix, L203 the protocol, L204 the baseline decisions and L219 the stress tests. Optional-paper knowledge is required only for the chosen thesis direction.

## Revision and selection ledger

This replaces the initial B01–B16 proposal, which had no authored or completed B lessons. Mapping: old B01→B01; B02→B02; B03→B03; B04→B14; B05→B15; B06/B07→B17; B08→B20; B09→B16; B10→B18; B11→B19; B12/B13→B21; B14/B15→B22; B16→B23/B24. The 001–240 spine and existing suffix identifiers are unchanged.

New coverage: numerical embeddings, TabDPT, Mitra, ConTextTab/TabSTAR, dedicated RT, PluRel, and current comparison candidates. Existing influential families receive retrieval and comparative exercises; they are not falsely presented as newly discovered. Recent diagnostics retain explicit exploratory status. The accompanying audit records evidence of uptake, source-access limits and rationale for core versus elective placement.

Before teaching, recheck versions and releases. Method sections above were inspected for major additions; optional papers retained from the initial pass still need full method/code audits. RT-J is supported by its author page because OpenReview access was blocked. No claim of independent replication, full-paper parity or learner mastery is made.
