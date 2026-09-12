# Lessons 047–070: individual quality and fidelity audit

Requested 2026-09-10; resumed 2026-09-12 for lessons 061–070. Status: lesson 061 repair complete; lesson 062 in progress. Each lesson is reviewed and repaired by a fresh subagent, in numerical order. The primary agent checks the comparison standard and integration. Individual reports record actual findings and verification; this index is not a certificate of paper reproduction.

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
| [047 SAINT](047.md) | Complete and pushed as `fb7a691`; Pages deployment succeeded and seven live files matched checked local bytes. Mobile numeric follow-up `d84eed7` also pushed, deployed and 18 live files checked. Default pinned full-path checks, deeper derivations/source-gap explanations, notebook execution, image checker and Pages evidence-link repairs, desktop/mobile review | Full pretraining and paper benchmark not reproduced; Bank smoke remains INCOMPARABLE |
| [048 DCNv2](048.md) | Complete and pushed as `fe70a03`; Pages succeeded and 23 live files matched the commit. Original-DCN/V2/readout derivations, paper experiment guide, live synthetic construction, fresh notebook execution, saved prediction reconstruction, image-checker/CD display/Pages link repairs | Full-data local MovieLens attempt remains INCOMPARABLE; original search and training results not reproduced |

| [049 ExcelFormer / Trompt](049.md) | Complete repair: full numeric Trompt model and live loss/routing tasks, measured probe, paper/implementation inventory, refreshed executed notebook, resume/provenance fixes; browser and 113 copied-Pages links pass. Pushed as `628da78`; Pages succeeded and 23 live files matched the commit. Mobile numeric follow-up `ff7e7ad` also pushed/deployed and 23 live files checked | Both paper benchmarks remain NOT_REPRODUCED; Trompt independent component parity and local probe have explicit scope |

| [050 FT-Transformer / XGBoost](050.md) | Complete repair: corrected numeric architecture and AUROC traces, fifth live ensemble exercise, stable live-code identity, fresh 26-cell notebook with exact prior scores; browser and 90 copied-Pages links pass. Pushed as `c3d6951`; Pages succeeded and 23 live files matched the commit | Historical Higgs attempt remains INCOMPARABLE; new ensemble diagnostic is post-hoc, not Table 4 reproduction |

| [051 Why trees still win](051.md) | Complete repair: corrected smoothing interpretation, detailed protocol/uncertainty map, duplicate-row and covariance derivations, new 36-fit bandwidth diagnosis, 28-cell fresh notebook, stronger source parity and mobile evidence display; final browser and 99 Pages links pass. Pushed as `eeaac3c`; Pages succeeded and 21 live files matched the commit | Full tuned benchmark remains NOT_REPRODUCED; one-task fixed-recipe curve is INCOMPARABLE and does not recreate the aggregate paper pattern |

| [052 TabR](052.md) | Complete repair: deeper paper design/projection/scaling analysis, annotated lab and live real-row trace, all-parameter source gradients, corrected tie check and precise official-evaluation route; fresh 24-cell teacher, browser and 89 Pages links pass. Pushed as `ce2c0fe`; Pages succeeded and 25 live files, including linked downloads, matched the commit | Numeric TabR-S mechanism validated; historical California result remains INCOMPARABLE; official 15-seed evaluation NOT_RUN |

| [053 RealMLP](053.md) | Complete repair: corrected architecture/meta-development diagram, TD versus TD-S and benchmark derivations, matched Adam state checks, new measured clipping intervention and live EXIT; fresh 26-cell teacher, browser and 96 Pages links pass. Pushed as `08bc5ce`; Pages succeeded and 29 live files matched the commit | Historical results unchanged; full benchmark NOT_REPRODUCED; local paired clipping effects remain INCOMPARABLE to Table B.2 |

| [054 TabM](054.md) | Complete repair: corrected mini adapters, shared biases and fan-in in separately measured v2; eight full-path source comparisons, fresh 21-cell teacher, live independent-batch reproduction lane and stable resume; browser and 99 copied-Pages links pass. Pushed as `2cf8abe`; Pages succeeded and 26 live files matched the commit | Historical operators remain immutable; corrected local suite and closer smoke remain INCOMPARABLE; tuned paper benchmark NOT_RUN |

| [055 TabReD](055.md) | Complete repair: extensive paper/appendix analysis, 2,879 original-report reanalysis with independent committed-blob checks, same-pool protocol and corrected TabM, measured 54-evaluation suite, five live tasks and fresh 29-cell teacher; browser and 108 copied-Pages links pass. Pushed as `52f0cbf`; Pages succeeded and 26 live files matched the commit | Author-report reanalysis is separate from training reproduction; precise Ecom split-alignment gap remains; full paper training and larger local lane NOT_RUN |

| [056 TabArena](056.md) | Complete repair: full protocol and paper-appendix expansion, live paired-win task and fitted Bradley–Terry model, current-source rating parity, three-regime reanalysis and 300 bootstrap refits; fresh 23-cell teacher, eight figures, browser and 106 copied-Pages links pass. Pushed as `69d7e86`; Pages succeeded and 25 live files matched the commit | Original ranks and source artifacts unchanged; new four-method Elo reconstruction is INCOMPARABLE to the full paper leaderboard; original training and full-roster reconstruction NOT_RUN |

| [057 Cross-family ensembling](057.md) | Complete repair: corrected TabM with27 new fold fits and54 verified archived arm outputs, full portfolio/OOF/selection explanation, actual-library source mismatch diagnosis, five live tasks and30-cell teacher; browser and97 copied-Pages links pass. Pushed as `f80b153`; after clearing a stale Pages deployment, successful deployment of `6dda5d0` published all 28 checked files unchanged from the 057 commit | Hybrid archive and conditional family removals remain exploratory; full TabArena portfolio and paper training NOT_REPRODUCED |

| [058 Surveys / meta-benchmarks](058.md) | Complete repair: primary survey and TALENT protocol analysis, five live tasks, strict parsing, held-out-method subset search at 1,000/10,000 proposals, 19-cell teacher, independent raw-table reconstruction and 1,800-rank source agreement; desktop/mobile, static access and 120 copied-Pages links pass. Pushed as `6dda5d0`; Pages succeeded and 24 live files matched the commit | Six-method reanalysis covers 300 tasks; missingness-aware tiny track covers 276 tasks and selects 41; original 45-task membership, training and Table 7 NOT_REPRODUCED |

| [059 Validation selection](059.md) | Complete repair: full paper and original GKM source analysis, KRR/PRESS and actual nested selection, 30/1,000 measured repetitions, five live tasks and 25-cell teacher, independent OOF reconstruction and translated source algebra checks; six portable figures, desktop/mobile, static access, regeneration and 133 copied-Pages links pass. Pushed as `64525b5`; Pages succeeded and 29 live files matched the commit | Exact equations and generator with a declared finite-grid extension; original paper figures/tables and native MATLAB/optimizer parity remain INCOMPARABLE / unverified |

| [060 Broad model comparison](060.md) | Complete repair: corrected five-family 210-result checkpoint, explicit row and panel identity, five live tasks, 32-cell teacher with actual kernel provenance, independent raw-label/metric/interval reconstruction and exact/Monte Carlo rank calibration; portable protocol and results, desktop/mobile, static access, regeneration and 141 copied-Pages links pass. Pushed as `b7bff72`; Pages succeeded and 24 live files matched the commit | Original TabArena/TabReD training remains INCOMPARABLE; reduced recipes, convenience panels, reused splits and variable host load limit inference |

| [061 Prior-data fitted networks](061.md) | Complete repair: full paper/source reading, original row Transformer and full-support Riemann head, fixed-GP derivations, three measured seeds, independent reconstruction of 9,216 author and 3,072 teacher predictions, five live tasks and 34-cell teacher with actual defaults/source identity; five portable figures, desktop/mobile controls, static access, regeneration and 127 copied-Pages links pass. Separate commit and live publication verification are reported in the session | Local GP approximation, fixed-head restrictions and prior mismatch are measured; original long-context Figure 4 training remains INCOMPARABLE; larger and live Colab runs NOT_RUN |

**Resumed for 061–070 by user request on 2026-09-12.** A fresh 062 author is working; 063–070 remain pending and will follow in numerical order. Each repaired lesson receives its own commit, push and publication check.
