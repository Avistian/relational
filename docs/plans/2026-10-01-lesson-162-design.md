# Lesson 162 Implementation Plan

Approved in chat: conceptual lesson plus **L162 Relational Context and Scale Audit**. Goal: trace the LM + GNN relational vision and defend an opportunities/obstacles map, extending L159 objectives and L161 scope. Primary source: Vogel, Hilprecht and Binnig, arXiv:2305.15321v1, §§2–4. Use the existing teaching workspace and preserve staged L161 work.

Architecture: dependency-free deterministic Python audits with three live learner functions; portable notebooks embed all source and fixture inputs. Shared lesson components supply retrieval, prediction, teach-back and responsive controls. New diagrams expose row tokens, graph reachability and frozen/trainable stages. The graph is an explicitly illustrative course construction, not an inferred historical implementation.

Tech stack: Python standard library; nbformat/nbclient/nbconvert; matplotlib; HTML/CSS/JavaScript; Playwright.

1. Write behavioral checks for reachable rows, token accounting and evidence classification in labs/_check_l162.py. Create failing stubs in labs/relkit/vision_l162.py and record the expected failure. Implement each contract; test invalid inputs, alias-free identities, induced-subgraph sampling and additive evidence issues.
2. Create pinned input fixtures, source ledger, complete deterministic audit and report in labs/evidence/l162 and labs/sources/l162. Independently verify graph walks by matrix powers, token costs by pair counting, and reject three incorrect learner functions.
3. Write lessons/content/0162-the-relational-fm-vision.md, reference and map template. Build architecture and reachability figures and interactive context/token controls in assets/. Clearly distinguish row length from row count, structural reachability from predictive usefulness and reconstruction from transfer.
4. Build HTML and standalone student/solution notebooks with three meaningful TODOs, CHECKs and an EXIT map. Execute the solution in an empty temporary directory; check report equality and embed portable images.
5. Integrate manifest, curriculum, Year5 plan, resources, glossary, retrieval/paper deck, dossier and preparation record. Preserve learner PENDING_WRITTEN_DEFENSE and Year4 gates.
6. Check all desktop/mobile widget states, keyboard/reset, print/no-JS, figures, source parity, notebook export, deterministic generation and copied Pages links. Stage only intended deliverables and validate the actual Pages build from the Git index. No publication.

Budget: USD0 cloud, one numerical thread, 600-second local execution limits. No paid training. Historical selected target remains Table1 wikiTables BART_table versus BART_table+GNN, all three tasks, three runs, 10,000 tables,70/20/10 splits, validation-selected checkpoints. Matching code, exact subset/split identities and training settings were not located. Historical result NOT_RUN; fidelity NOT_ESTABLISHED. No fabricated historical trainer or numerical tolerance. USD10 standing aggregate ceiling does not authorize an unpriceable reconstruction. Source gaps must be resolved before fresh benchmark planning.

Validation establishes local computation and artifact delivery only. It does not establish full-paper reproduction, transfer gains, live Colab, deployment or learner mastery. Human review evaluates the map's scientific reasoning.
