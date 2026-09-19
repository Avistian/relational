# Lesson 083 delivery

Prepared GraphSAGE lesson and full inline PPI lab. Authoring does not assert learner completion. Design follows the scheduled Year3 unit: explain the transition from transductive Cora to disjoint PPI graphs; expose neighborhood sampling, mean aggregation, and training access; replay a full named experiment with explicit release/paper differences.

Deliverables: canonical prose and HTML, student and executed solution notebooks, reference card, four portable figures, two causal controls, three live TODOs, full-data runner and audit, pinned source/data identity, environment snapshot, and manifest/gallery integration.

Named target: Table1 supervised GraphSAGE-mean PPI .598. Execute all3 published learning rates,10 epochs, full width/data, seeds123–125. Observed mean .59092968, sample SD .00605956. Historical parity remains INCOMPARABLE: original TF not run and historical seeds/config/selection are incomplete. See `labs/l083-reproduction.md`.

Visual review: unlike L082's shared adjacency normalization, L083 shows the outward support tree and inward shared-weight computations, explicitly separating self and neighbor paths. Static arithmetic and interactive changes preserve the same [2,4] receiver. Figures inspected at notebook and browser width; narrow screens retain a scrollable diagram and reflow the prose/controls. Browser checks exercise controls and keyboard input.

Evidence is recorded by `_verify_l083_results.json`, `_execution_l083_results.json`, `_browser_l083_results.json`, and `_delivery_l083_results.json`. Full execution and source parity are separate from historical training parity. Live Colab and remote deployment NOT_CHECKED.
