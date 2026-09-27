# Lesson 137 — Error analysis on REG

Approved 2026-09-27: fresh full-data rel-f1/driver-position comparison, five basic RDL fits (RelBench v1 Table7) and five released SQL/LightGBM ten-trial searches. USD10 aggregate, USD3 overhead reserve, 900-second GPU workers, no automatic retries. FE local runtime capped at one hour aggregate. No publication requested.

## Teaching design

L136 establishes a score contract; L137 asks where errors concentrate and which explanations the evidence supports. Align predictions by entity and cutoff; calculate paired absolute-loss differences; define slices using past history, recency and FE missingness; nominate a supported slice on validation and freeze it before test analysis. Require at least 30 queries and 10 drivers. Report all planned slices, seed variability and paired driver-cluster bootstrap intervals conditional on this task, split and fitted models. Previously inspected test data are not a new untouched holdout. Slice associations are not causal diagnoses.

Preserve the released frozen snapshot and its feature gaps. Both pipelines use a database snapshot through 2010-01-01, but different feature construction/temporal access and preprocessing; certify neither historical arrival times nor an architecture-only causal comparison. Audit actual source before final prose.

## Implementation and verification sequence

1. Freeze inherited source, data and runtime identities; create L137 operators and budget. Execute pilot then five GNN fits; separately prepare full SQL features and run five complete FE searches in pinned CPU runtime.
2. Test then implement keyed loss pairing, validation slice nomination, explicit unsupported slices, and driver-cluster resampling. Independently verify query labels, metrics, SQL features, source parity and every seed artifact.
3. Freeze validation-selected slice with hashes before test analysis. Preserve full training and FE replay alongside original error-analysis extension. No invented historical FE scalar or whole-paper parity.
4. Build canonical HTML, student/solution notebooks with live TODO/CHECK functions, visible model/trainer/SQL/search, portable figures, reference and runnable full gates.
5. Execute portable notebook and complete gates, scientific mutation checks, desktop/mobile/keyboard/reset/no-JS/print checks, deterministic rebuild and copied Pages navigation/assets. Update curriculum/manifest/resources/notes/thesis without learner completion claims.

## Evidence boundaries

GNN: selected Table7 reproduction under inherited descriptive .2 MAE tolerance. FE: complete released pipeline replay with historical data/RNG adapters; historical score identity NOT_ESTABLISHED. Slice analysis: original course extension, not a published paper result. Whole-paper/all-task training NOT_RUN; live Colab/deployment NOT_CHECKED.
