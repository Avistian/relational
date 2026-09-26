# Lesson 114: approved design and execution plan

User approved the combined scope: OGB error analysis plus a full selected ogbn-arxiv MLP reproduction, with all ten L112 GCN checkpoints freshly replayed. No new GCN training. No publication requested.

## Frozen protocol

OGB v6 Table 6 MLP targets validation57.65±0.12%, test55.50±0.23%. Ten seeds0–9, full original data/splits, 500 epochs, 3 linear layers128→256→256→40, BN/ReLU/dropout0.5 after hidden layers, Adam0.01. Train-row-only MLP forward/BN during fitting. Select first validation maximum. Predeclared absolute mean tolerance0.5pp, descriptive rather than equivalence. Pin same OGB commit as L112; modern runtime/separate seeds are explicit deviations.

Degree: unique undirected neighbors excluding self. Bins0,1–2,3–5,6–10,11–20,21–50,51+. Retrospective label homophily: same-class fraction of unique neighbors; undefined isolates distinct; bins[0,.25),[.25,.5),[.5,.75),[.75,1]. Classes0–39; each publication year. All nodes of valid/test included; no sampling. Report accuracy, counts, both-correct/GCN-only/MLP-only/neither and seed SD. No IID node confidence interval. Choose one lowest validation GCN-minus-MLP slice with n>=200 among degree/homophily/class; freeze its rule and report test performance without reselection. All-slice test exploration remains descriptive. Homophily and true class require labels and cannot become a serving-time selector.

## Execution

1. Test analysis against small graph/SQL oracles; check original MLP output/gradients/BN updates and training eligibility.
2. Pilot10 epochs on T4,2 cores,8GiB,600s maximum. Full workers10×600s, no retries. Rate verified2026-09-26: .000164+.0000262+.00001776=.00020796 USD/s; total resource ceiling1.372536 USD, remainder of USD10 reserved for overhead. Pilot gates full launch.
3. Collect full histories/states/predictions. Replay all ten MLP and GCN states through original code; independently reconstruct aggregate and slice metrics.
4. Write connected lesson, reference, three live notebook tasks, full visible code and training command, figures, slice explorer and written error-report exercise. Preserve PENDING_WRITTEN_DEFENSE.
5. Execute solution; reject broken student functions; validate browser/mobile/keyboard/print/no-JS and copied Pages staging. Live Colab and deployment separate.

The brainstorming skill's suggested writing-plans skill was searched for and is unavailable; this checklist supplies the implementation plan.
