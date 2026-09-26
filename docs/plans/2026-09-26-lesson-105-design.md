# Lesson 105 — continuous time: event streams versus snapshots

User approved the full-data representation audit on 2026-09-26. This follows L104's causal boundary and precedes L106's future-edge evaluation and L107's snapshot models. No new architecture or paid compute.

## Contract
Retrieval-first lesson, interactive aggregation and release-time visualizations, independently readable student/solution notebooks, portable figures, reference card and manifest-driven navigation. Three live tasks: assign half-open bins; aggregate typed endpoint pairs without losing counts; enforce snapshot release times. EXIT requires a justified representation choice. Prepared material is PENDING_WRITTEN_DEFENSE.

Complete course experiment: all 157474 interactions in SHA256-pinned JODIE Wikipedia bytes; fixed origin 0 seconds; widths 3600, 86400, 604800 seconds. Compare event records, binary window edges, count-weighted window edges. Preserve user/item identity spaces. Report multiplicity collapse, strictly ordered event pairs hidden within windows, observed timestamp ties, and release latency. Every event time is a query; count strictly past global-history events withheld by closed-window access and nonpast records exposed by reading the completed current window. These are representation counts, not sampled model dependencies or predictive scores. Snapshot availability is max(window end, latest constituent availability); real data assume arrival=event time. Incomplete last window remains unreleased during the observed stream.

## Evidence and sources
TGN v3 section 2 supplies event/snapshot definitions; JODIE authors supply Wikipedia and format; pinned TGN preprocessing provides typed-ID verification. This is a fully reproducible course audit, not a published benchmark replication. No AP target, model fitting, split selection, or seed uncertainty applies. Windows are predetermined, not selected by outcomes. Raw edge features/state labels are outside the projection; complete original bytes are hashed. Missing ingestion histories are explicit.

## Implementation and validation
1. Write meaningful semantic tests; observe missing implementation failure.
2. Implement visible standalone NumPy stream/snapshot functions with validation and three student seams.
3. Hash source data, parse raw columns independently, run all three full-data conditions. Save compact projection, snapshots, configuration, environment, code/data hashes and measured report. No unvalidated processed caches or resumed results.
4. Verify aggregates through SQLite, temporal-pair counts and release/access counts through independently grouped timestamps; include adversarial ties, exact boundaries, delayed arrival, empty input and order-collision fixtures.
5. Build coherent HTML, notebook pair, reference and figures. Execute the solution in a clean temporary working directory with pinned raw input; compare all results to author evidence.
6. Test both interactive widgets and feedback, desktop/mobile/print/no-JS, visible notebook figures, links, deterministic rebuild and a copied Pages tree matching the workflow.
7. Update curriculum/resources/notes and preserve unrelated working-tree changes. Live Colab and deployment remain NOT_CHECKED unless actually exercised.

The writing-plans skill named by brainstorming is not installed (searched available skill roots). This document supplies the concrete implementation plan directly.
