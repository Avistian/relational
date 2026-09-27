# Lessons 103–135 · individual teaching review

Reviewed 2026-09-27. Scope: sequence, readability, prerequisite load, computation diagrams and publication. This review does not upgrade any scientific reproduction result, live-Colab status or learner-mastery record.

The existing sequence is technically substantial. The primary improvements are a missing benchmark entry lesson, compact causal handoffs, optional prerequisite refreshers, model-specific diagram trace questions, and two more readable computation visuals. Existing experimental code and saved notebook outputs were preserved.

## Lesson-by-lesson findings

### 103 · TGAT: attention that knows when

The time-feature derivation was a steep opening. Added a plain-language first-pass route and defined query/key/value vectors before the forward pass. Retained the recursive-cutoff architecture and source-versus-corrected-sampler distinction.

**Next question:** Audit which records actually entered that representation.

### 104 · Information leakage in time: audit the prediction

The two-clock and label-maturity examples are strong. Added an entry reminder and explicit handoff from dependency audits to representation loss. Retained paired-candidate and frozen-weight boundaries.

**Next question:** Ask what a daily snapshot discards even when its inputs are legal.

### 105 · Continuous time: event streams versus snapshots

The event/snapshot comparison is coherent and uses one six-event example. Retained separate aggregation, ordering and release diagrams; made the next candidate-evaluation question explicit.

**Next question:** Define the candidate interactions against which a prediction will be scored.

### 106 · Temporal link prediction: test the candidates

EdgeBank correctly earns a state/membership architecture, although it has no learned parameters. Retained the tied-score arithmetic and negative-set intervention; added a short precision/recall refresher.

**Next question:** Choose which state a model carries between completed snapshots.

### 107 · Snapshot methods: decide what remembers

The two axes of recurrence are visually distinct. Added a GCN-versus-recurrent-update reminder; retained the released-O discrepancy and unsuccessful reproduction evidence.

**Next question:** Retrieve a bounded event history without confusing a faster implementation with a changed sampler.

### 108 · Temporal neighbor sampling: spend context carefully

The interval and recursive-cutoff diagrams answer different questions and should remain separate. Added fanout, hop and padding reminders; linked the need for timestamp provenance to L109.

**Next question:** Decide which database timestamps can actually support those cutoffs.

### 109 · Database timestamps: reconstruct what was known

The text correctly changes from TGAT child cutoffs to a fixed database snapshot cutoff. Made that transition easier to locate. No neural architecture is needed for the version/join audit.

**Next question:** Carry the information contract through training and checkpoint restoration.

### 110 · Q3 checkpoint: a temporal GNN you can defend

The score-before-update diagram and atomic checkpoint story connect the quarter. Added the distinction between parameters and temporal state, and repaired the missing next-lesson route by supplying L111.

**Next question:** Transfer this protocol discipline to an official static-graph benchmark.

### 111 · OGB setup and the benchmark contract

Missing from both the learner sequence and publication. Added the official OGB loader/split/evaluator lesson, split and held-out-label counterexamples, full-data constant baseline, notebook pair, reference and protocol. This is course evidence, not another GCN reproduction.

### 112 · OGB reproduction: defend a full GCN experiment

The GCN architecture is complete, including its third graph multiplication and BN state. Added the direct L111 handoff and a transductive/percentage-point refresher; retained the official experiment.

**Next question:** Determine what changes when a full graph no longer fits one training step.

### 113 · Scaling OGB: preserve the computation, measure the cost

The text correctly separates cluster sampling from GraphSAGE and distinguishes the GCN bridge. Added that distinction to the entry card; retained the products result outside tolerance.

**Next question:** Return to arxiv and identify which populations the trained model gets wrong.

### 114 · OGB error analysis: where does the GCN lose?

The MLP/GCN normalization-population contrast is informative. Added slice/degree/homophily reminders and a concrete transition from an error report to module design.

**Next question:** Describe a proposed change in terms of encoder, processor and prediction head.

### 115 · Graph ML design patterns: encoder → message passing → head

The diagram correctly leaves propagation inside the final head. Added encoder/processor/head reminders and a trace question that detects an apparently shape-preserving but incorrect refactor.

**Next question:** Probe whether the implemented modules and optimizer fulfill those contracts.

### 116 · Debug GNN training: make the failure falsifiable

The probe-first narrative and scalar gradient witnesses are useful. Added the gradient-versus-update reminder and the transition to constructing a database graph.

**Next question:** Build the graph from database keys rather than accepting it as a given.

### 117 · RDL bridge: from database rows to predictions

The old overview largely listed stages. Redrew it to show a legal/illegal row example, separate table encoders, neighbor and root paths inside each relation, repeated layers and seed-only supervision. Propagated the portable figure to dependent lessons.

**Next question:** Compare this modern stack with the earlier target-specific extraction and readout in Cvitkovic.

### 118 · Cvitkovic: relational graphs before modern RDL

The target-specific extraction and gate/value readout make this architecture distinct. Added the historical transition and a trace question about graph-local pooling; retained unrun historical experiment boundaries.

**Next question:** Turn representation mechanisms and their limitations into a defensible synthesis.

### 119 · Year 3 synthesis: graphs versus flat tables

The representation collision is appropriately narrow and acknowledges a manual-feature repair. Added the transition from competing constructions to a conditional argument; retained original course versus benchmark evidence.

**Next question:** Demonstrate the complete pipeline and defend it in the Year 3 exit exam.

### 120 · Year 3 exit exam: a heterogeneous temporal GNN

The exam already separates the course model from the published-model lane. Added an index-space reminder and a diagram comparison prompt. Updated its inherited portable RDL architecture.

**Next question:** Use the historical map to understand why the Year 4 design choices exist.

### 121 · History of relational ML: rules, features, and graphs

The historical map explains design choices rather than declaring every newer method superior. Added grouping as the entry concept and an explicit transition to auditable graph construction.

**Next question:** Construct the row graph and verify every edge against a relational join.

### 122 · REG construction: from database keys to a verified graph

Key mapping and relation direction are demonstrated numerically. Added local-index reminders and a graph-to-clock handoff. Updated the inherited model overview without changing graph code.

**Next question:** Restrict each prediction to the rows visible at its own cutoff.

### 123 · Temporal heterogeneous graphs: what may this query see?

The fixed root cutoff is crucial and differs from L103. Put that distinction in the prerequisite reminder. Retained two-clock fixtures separately from unavailable real ingestion histories.

**Next question:** Specify the future target and identify each question in a task table.

### 124 · Entity vs task table: what question does this row ask?

Task rows and entity rows are clearly separated. Added the horizon/query reminder and a diagram prompt tracing one question to one loss term.

**Next question:** Turn the allowed mixed-type feature columns into trainable row vectors.

### 125 · PyTorch Frame: from typed columns to graph node features

The six-type overview was dense at reading width. Replaced the tiny four-column grid with six larger computation cards and a shared output-shape summary. Retained the separate course model and historical-reproduction limits.

**Next question:** Use the benchmark API to keep database inputs, task questions and evaluation aligned.

### 126 · RelBench beta: from database to evaluated predictions

The API tour preserves query identities and reconstructs AP ties. Added a reminder that equal array lengths do not establish aligned questions. An evaluation-flow diagram is appropriate; no new neural architecture is needed.

**Next question:** Freeze the entire neural training and selection protocol for RelBench v1.

### 127 · RelBench v1: run and audit the RDL baseline

The full neural experiment follows naturally from the API tour. Added validation/test and seed-SD reminders and a query-to-head trace prompt.

**Next question:** Change head, loss and metric when the prediction request changes.

### 128 · Task taxonomy: choose the target, head, loss and metric

The lesson correctly separates temporal prediction from autoregression and distinguishes output meaning from output width. Added probability/logit/ranking reminders; retained the historical task-label reconstruction caveat.

**Next question:** Construct the strong feature-engineered competitor under the same question contract.

### 129 · Manual feature engineering: what RDL must beat

The SQL/LightGBM competitor is more meaningful than raw-row trees. Added as-of features and human-versus-machine effort reminders; retained the distinction between Table 7 and the feature-engineering study.

**Next question:** Assemble the complete RDL pipeline and defend it against this comparison.

### 130 · Q1 checkpoint: run and defend the complete RDL pipeline

The checkpoint assembles prior contracts and includes a strong keyed-MAE counterexample. Added packet identity/MAE reminders and a handoff to actual activation tracing.

**Next question:** Instrument a real forward and backward pass to explain what is learned.

### 131 · GNN + tabular encoder stack: trace the forward and backward pass

The measured batch, time-owner lookup and backward path are explicit. Added autograd/detach reminders and a forward-versus-gradient trace prompt. Retained the observed nonfinite-gradient finding.

**Next question:** Ask how representations can depend on which entity is the current query root.

### 132 · Identity-aware message passing: who is asking?

The two-tower and query-conditioned architectures are distinct. Added root-marker/global-ID reminders and a comparison prompt; retained ownership and four-hop reachability checks.

**Next question:** Open the relation-specific arithmetic that combines all those messages.

### 133 · Heterogeneous convolution on a relational entity graph

The two sums and empty-versus-absent relations are useful mechanisms. Added answers to the retrieval prompts and an entry reminder that counts repeated root/bias contributions.

**Next question:** Budget the sampled occurrences while preserving this layer computation.

### 134 · Training at scale: budget a temporal mini-batch

The host/sampler/device diagram and typed expansion arithmetic connect correctness to cost. Added throughput and allocated/reserved-memory reminders and the direct tuning-budget handoff.

**Next question:** Use the measured costs to reserve and enforce a fair tuning budget in Lesson 135.

### 135 · Tuning on the REG: spend a budget, defend a decision

Read-only review of the concurrently authored package. The L134 cost-to-search transition, nested selection worked example, budget chart, fresh paired seeds and explicit evidence boundary are coherent. This is an experiment-controller lesson; a new neural architecture would be misleading. The authoring session is complete, confirmed by the user; the final package is included in publication.

## Verification outcome

All 33 lessons pass the final 66-view desktop/mobile audit, with zero JavaScript errors, failed local HTTP requests or document overflow. All 33 also passed no-JavaScript and print checks. The copied Pages tree has 2,675 valid local links/anchors across lessons, reference cards and rendered notebooks. Detailed per-lesson checks pass after repairing outdated gallery/disclosure selectors and refreshing changed visual fingerprints.

The review retained measured scientific failures, including L113's out-of-tolerance result and historical/unrun boundaries in later lessons. These delivery checks are not new model experiments.

## Verification records

- `build.json`: builders and notebook code/output preservation.
- `delivery.json` and individual logs: arithmetic, interactions, reset, keyboard, no-JS, print and copied staging checks.
- `pages.json`: learner lesson, reference and rendered-notebook local links and anchors against the real workflow copy commands.
- `browser.json`: final desktop/mobile rendering and asset checks.
- `publication.json`: clean Git-index build, pushed commit, Pages workflow and live-byte verification (written after publication).

Existing scientific failures/unrun lanes remain visible. No paid experiments were launched in this review. L111 is a newly executed CPU course baseline; prior experiments are reused evidence.
