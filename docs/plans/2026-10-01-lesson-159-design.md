# Lesson 159 approved design and implementation plan

Approved in chat 2026-10-01. Foundation-model preview: masked reconstruction versus demonstrated transfer. Correct arXiv2305.15321 attribution to Vogel, Hilprecht and Binnig (2023), including curriculum references which carry the same mistaken attribution.

Named published target: Table1 wikiTables BART_table versus BART_table+GNN, all three reconstruction tasks, 10,000 tables,70/20/10 split, best-validation selection, three runs. Exact subset/splits, released implementation, model configuration and training schedule not located. Audit sources and report missing fields. Full benchmark NOT_RUN; fidelity NOT_ESTABLISHED. Do not manufacture a paper preset or substitute another model.

Approved executable lane: explicitly synthetic, CPU-only mechanism experiment with visible masking, classifier loss, two-stage training, frozen row encoder/decoder and graph updates. Small neural codec is not BART; binary classification is not text generation. Three paired seeds, fixed table-disjoint splits, all held-out predictions, validation-only selection; independent loss/metric/freeze/leakage checks. No cross-database inference or paper-score parity claim.

Budget: USD0 cloud spending; standing USD10 total ceiling does not authorize a benchmark with unknown recipe/cost. Bound local run to 120s and one CPU thread; stop on limit without silently reducing scope. Ship HTML, reference, student/solution notebooks, model-specific architecture and portable mechanism figures, interactive leakage/gradient controls, one-page brief template, source/protocol/deviation ledger and checks. Live Colab/deployment are NOT_CHECKED.

Implementation sequence:
1. Pin source HTML and audit release discovery; tests first for semantic masking, selected-target loss and frozen-weight gradient flow.
2. Implement visible miniature codec/graph/trainer, execute declared lane, retain predictions and validation history, independently rescore and test counterfactuals.
3. Write intuitive lesson with L158 callback, model/operation visuals, worked loss and masking trace, critique Table1 scope, and brief rubric. Generate portable notebooks with three live TODOs.
4. Execute solution from empty directory; audit code parity and deliberately incorrect learner functions. Check desktop/mobile/keyboard/reset/no-JS/print, deterministic generation and local links.
5. Update manifest, primary sources, glossary, retrieval and paper deck, author-preparation notes and thesis ledger. Preserve pre-existing changes; stage only this lesson's additions and intended shared-file updates, check clean Git-index Pages build. No push/deployment.

Writing-plans skill was not found in installed skill roots. This document records the plan directly. No sub-agents requested or used.
