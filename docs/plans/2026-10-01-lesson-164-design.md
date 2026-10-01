# Lesson 164 — Griffin: approved design and implementation plan

Approved by user 2026-10-01 after source inspection. Teach Griffin's shared encoder, task-conditioned cell cross-attention, hierarchical relational aggregation and decoder. Bridge L163's fixed row bottleneck to task-dependent cell retrieval; distinguish unified interfaces from demonstrated transfer.

## Named reproduction

Griffin arXiv2505.05568v1 Table12: Others-2 FULL SFT → rel-f1-driver-dnf versus no-pretrain; 512 and4096 training queries; split seeds42–46, fixed source model seed42;20fits. Targets: no-pretrain .6558/.7176; Others-2 SFT .7098/.7275 AUROC. Release b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427; data e0c54ceada75317b06f11f8dcda7aa8304fbb593; checkpoint bd8c5be5130f34e7faa31099d0bd81d95d0aa995. Up to200epochs, patience10 (source semantics), validation every2epochs, batch256, hop2, fanout20, hidden512, four message-passing layers, lr3e-4, wd2e-4, reverse edges on, gate off, fewshotfanout3, full validation and test. Pin/audit source/data and preserve all deviations before score claims. Source checkpoints substitute for fresh pretraining; whole-paper reproduction NOT_RUN.

## Aggregate budget

USD10 inclusive of preparation, seeds, retries, validation and overhead: USD1 preparation/timing, USD7 fits/checks, USD2 reserve. L4 +4physical CPU cores +32GiB = USD0.00034544/sec at current Modal rates (USD1.243584/hour). Reserve worst-case runtime before each dispatch; do not consume reserve automatically. A bounded timed probe gates the full20fits. Stop with INCOMPLETE_BUDGET_GATE if conservative projection exceeds remaining allowance. No reduction of seed count, train sizes or validation coverage to manufacture a full result. USD0 spent before approval. No push/deployment.

## Teaching and delivery

Self-contained illustrated HTML; portable student and executed solution notebooks with readable full mechanism/model/trainer and separate release reproduction lane; quick reference; immediate learner checks; prediction-before-reveal, spaced retrieval and written defense. Visuals expose cell Q/K/V, per-relation mean and across-relation maximum, temporal eligibility, frozen/shared/trained boundaries, classification/regression decoding. Default notebook executes bounded mechanism/source parity, not paid training. Author evidence is not learner mastery.

## Implementation sequence

1. Archive primary sources and pins; inspect model, data mapping, checkpoint and temporal sampler. Audit all train/validation/test keys and the source protocol; record actionable gaps.
2. Write independent numerical/contract checks first. Implement visible cell attention, hierarchical aggregation, decoder and aligned source adapter; test output/gradient parity where feasible.
3. Build the managed budget ledger and runnable20-fit release lane; run bounded preparation/timing, then full fits only if projected total fits. Independently rescore every completed seed.
4. Write the causal lesson, model-specific architecture and interactive worked trace; build portable notebooks with meaningful live TODO/CHECK/EXIT tasks and gated full run.
5. Execute solution in an empty directory; check independent oracles, source/hash parity, browser controls at desktop/mobile, images, copied links, deterministic builds and Git-index Pages. Update manifest/course/reference/provenance/dossier and record only actual evidence.

No writing-plans skill is installed at the searched skill roots; this explicit implementation plan supplies the transition after the approved brainstorming design.
