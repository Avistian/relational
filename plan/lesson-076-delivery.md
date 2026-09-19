# Lesson 076 delivery design

Follow the approved curriculum: actual PyTorch Frame table encoders → visible key/time filtering → manual one-hop mean → trainable head. A two-table, fixed-cutoff synthetic fixture is appropriate for this architecture preview. Alternatives were a diagram-only lesson (no runnable evidence) and replacing the preview with RelBench training (too many prerequisites). Deliver the complete preview plus a separately labeled historical RelBench replay.

Learning contract: trace [4,2,4] customer and [5,2,4] event tokens into width-4 rows, map noncontiguous foreign keys to positions, exclude unavailable events, reduce repeated destinations, concatenate [4,8], predict [4], and differentiate into both encoders. Three live TODOs implement key routing, time eligibility and mean reduction. Compare a differentiable dense oracle, test permutations and future interventions, execute a short overfit diagnostic without claiming generalization.

Reproduction target: RelBench v1 Table 6 rel-f1 driver-dnf. Archive the July 29 2024 implementation; expose its full encoder/GNN/head/trainer source alongside the small composition. Audit the published defaults against actual released fanouts, pin source and environment, provide CLI/Colab/Modal operators, and attempt local preflight. Do not substitute toy loss for AUROC or claim a completed benchmark when runtime prerequisites prevent it.

Visuals: complete architecture with shapes; exact key routing; live mean/sum and time intervention with baseline; gradient path with empty/future edges. Reuse shared retrieval/predict/teachback. Validate executed notebook, canonical source parity, primitive oracle, desktop/mobile controls, portable figures, copied Pages links and manifest. No learner mastery is inferred.
