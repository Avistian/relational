# Lesson 099 approved design and implementation plan

User approved the combined scope on 2026-09-20: a paper-grounded R-GCN/HGT lesson, fully executed shared-graph comparison, attention ablation and feature-only baseline, standalone notebooks and explicit published-experiment reproduction tracks. No publication requested.

Use the complete conference-filtered DGL ACM raw graph: paper/author/subject types, four directed relation stores, conference edges excluded, paper bag-of-words features and constant auxiliary features. Pin raw bytes and reference source. Define a new stratified course split and record every ID; this is not the ACM3025/HAN historical split. Keep all selected papers and all connected auxiliary nodes. Transductive graph access is explicit.

Freeze before scoring: four arms (R-GCN, HGT, uniform-attention HGT, feature-only MLP), two learning rates, three paired seeds, 60 complete epochs per fit; choose learning rate per arm by average validation cross-entropy, select checkpoints by validation cross-entropy. Same depth2/width32/input adapters/dropout0/training updates. Report actual trainable/active parameter counts and runtime; equal width and updates are not equal capacity or FLOPs. Uniform attention removes score parameters and retains the HGT message/output/residual operators. No test-driven tuning. The experiment is one graph, not an architecture-wide winner claim.

1. Pin data/source and define loader/split/model correctness checks first.
2. Implement visible relation mean, receiver softmax, typed HGT and controlled trainer; verify with hand/dense/gradient oracles and held-out-label interventions.
3. Execute all 24 fits and preserve traces, predictions, checkpoints, selection decisions, hashes and environment. Audit independent metric reconstruction and checkpoint reload.
4. Provide named R-GCN AIFB and HGT CS reproduction launchers and protocol/deviation ledgers, retaining missing historical/full-paper evidence explicitly. Fresh AIFB replay where feasible; never label ACM comparison paper reproduction.
5. Build coherent lesson, editable whole-computation visual, student/solution notebooks with three live tasks, reference sheet, attribution table and manifest entries.
6. Execute the standalone solution, inspect desktop/mobile lesson and prepared notebook, verify deterministic rebuild and copied Pages delivery. Do not infer learner mastery.

The requested writing-plans skill is not installed (skill catalog and filesystem search checked); this file supplies the implementation plan directly.
