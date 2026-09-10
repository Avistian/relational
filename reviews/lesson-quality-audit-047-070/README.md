# Lessons 047–070: individual quality and fidelity audit

Requested 2026-09-10. Status: in progress. Each lesson is reviewed and repaired by a fresh subagent, in numerical order. The primary agent checks the comparison standard and integration. Individual reports record actual findings and verification; this index is not a certificate of paper reproduction.

The user subsequently emphasized that 055–070 are shorter and lower quality, and authorized extensive paper reading and expansion. Those reviews must assess substantive depth and load-bearing paper coverage, with particular attention to whether reduced demonstrations leave the main method unexplained.

## What the comparison with 042–045 means

The useful standard is the teaching sequence, not their wording or length:

- **042:** turn “fair comparison” into a shared evaluation frame, model-specific search spaces, a validation-selected checkpoint, and a cross-dataset interpretation. Tasks explain the goal and why the decision matters.
- **043:** show the whole encoder before deriving sparsemax, the attentive transformer, the prior update and the feature transformer. Then connect a controlled synthetic mask experiment to a separate real-data comparison. The explicit failed validation is valuable evidence, not something to hide.
- **044:** start with an oblivious tree, derive differentiable feature choice, splits and routing, then assemble the network and measure quality and cost. The implementation check and comparative experiment answer different questions.
- **045:** connect contextual categorical embeddings to a context-free ablation, explain the numerical bypass, and link pretraining to a label-efficiency experiment. Identify which paths exist only during pretraining.

These examples are structural benchmarks, not assumed factually perfect. A close metric does not by itself validate an implementation; a winner changing across three datasets does not establish a universal impossibility result; and an architectural motivation is not proof of performance. The audit must avoid carrying such overstatements into revised lessons.

## Acceptance standard

A reader should be able to explain the mechanism, trace its diagram, find the corresponding visible code, implement a meaningful piece, execute diagnostic checks, and interpret a saved or newly produced result without confusing it with the paper's benchmark. Each substantive claim needs an appropriate primary source, derivation, or measured artifact, with the scope of that support made explicit.

New-model lessons need an end-to-end architecture overview in the lesson and notebook. Evaluation lessons need an equally concrete protocol trace. Both need motivation, terminology, worked examples, retrieval and explanation tasks, and links to the larger architecture/evaluation progression.

An individual repair is complete only after source, builder, student notebook, solution and prepared HTML agree and relevant behavioral/delivery checks pass. Paper-scale reproduction, live Colab and deployment are separate statuses. Read each numbered report for what was and was not verified.

## Lesson reports

See [the assignment rubric](BRIEF.md) for the full checklist.

| Lesson | Repair and verification | Reproduction boundary |
|---|---|---|
| [047 SAINT](047.md) | Complete and pushed as `fb7a691`; Pages deployment succeeded and seven live files matched checked local bytes. Default pinned full-path checks, deeper derivations/source-gap explanations, notebook execution, image checker and Pages evidence-link repairs, desktop/mobile review | Full pretraining and paper benchmark not reproduced; Bank smoke remains INCOMPARABLE |
| [048 DCNv2](048.md) | Complete and pushed as `fe70a03`; Pages succeeded and 23 live files matched the commit. Original-DCN/V2/readout derivations, paper experiment guide, live synthetic construction, fresh notebook execution, saved prediction reconstruction, image-checker/CD display/Pages link repairs | Full-data local MovieLens attempt remains INCOMPARABLE; original search and training results not reproduced |

| [049 ExcelFormer / Trompt](049.md) | Complete repair: full numeric Trompt model and live loss/routing tasks, measured probe, paper/implementation inventory, refreshed executed notebook, resume/provenance fixes; browser and 113 copied-Pages links pass. Pushed as `628da78`; Pages succeeded and 23 live files matched the commit | Both paper benchmarks remain NOT_REPRODUCED; Trompt independent component parity and local probe have explicit scope |

| [050 FT-Transformer / XGBoost](050.md) | Complete repair: corrected numeric architecture and AUROC traces, fifth live ensemble exercise, stable live-code identity, fresh 26-cell notebook with exact prior scores; browser and 90 copied-Pages links pass. Separate commit/push in progress | Historical Higgs attempt remains INCOMPARABLE; new ensemble diagnostic is post-hoc, not Table 4 reproduction |

051 is in progress. 052–070 are pending individual review.
