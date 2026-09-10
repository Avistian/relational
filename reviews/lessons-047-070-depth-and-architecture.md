# Lessons 047–070: explanatory and visual revision

Date: 2026-09-10. Comparison baseline: `938d76f94c6aa57e7c48dce053a31a9b8f89cf22`.

All 24 lesson packages now carry additional, individually authored explanations beside the concepts they develop. The same 113 source blocks appear in the lesson, student notebook and local teacher notebook. Each reference card also gains a compact computational aid. The standard comes from lessons 042–046: establish the problem, define the objects, derive the operator, trace it, and explain what the evidence permits.

The revision changes teaching material and presentation. It preserves the existing executable student exercises and exact saved teacher code/output pairs. It does not create new benchmark results or upgrade any reproduction claim.

## What each lesson now develops

| Lesson | Additional explanation and learner trace |
|---|---|
| 047 SAINT | Token meaning, feature/row reshapes, companion-value arithmetic, companion policy, contrastive objective, and parity scope. |
| 048 DCNv2 | Polynomial degree, the fixed input anchor, a complete rank-one update, parallel/stacked branches, and mixture limitations. |
| 049 ExcelFormer / Trompt | Claim populations and budgets, directed attention, mixed targets, prompt reduction axes, and experiment comparability. |
| 050 Q1 checkpoint | The fitted procedure as the estimand, fit boundaries, candidate selection, paired ranks, and uncertainty. |
| 051 Tree inductive biases | Capacity versus bias, smoothing weights, rotation geometry, noise selection, and discriminating interventions. |
| 052 TabR | Memory as model input, top-k versus softmax, relative-key value correction, legal serving memory, and parity boundaries. |
| 053 RealMLP | Transfer of defaults, clipping derivatives, parameter/gradient coordinates, zero-head gradients, and schedule troughs. |
| 054 TabM | Ensemble covariance, a computed adapter fixture, loss of a mean versus mean loss, probabilities versus logits, pruning, and cost. |
| 055 TabReD | Temporal estimands, row cutoffs, availability clocks, validation/refit distinctions, and falsifiable drift claims. |
| 056 TabArena | Benchmark population, fitted procedures, order of seed/rank aggregation, ties, Elo, and paired bootstrap units. |
| 057 Ensembles | Complementary errors, an OOF row trace, class schema, probability averaging, selection, and deployment cost. |
| 058 Surveys | Taxonomies versus benchmarks, missingness during parsing, ranks, outcome selection, covariance, and task/row weighting. |
| 059 Validation overfitting | Adaptive selection, a two-candidate optimism calculation, independent null controls, nested selection, and feedback. |
| 060 Broad comparison | The complete evidence chain, split/recipe identity, selection and reconstruction, three resolutions of evidence, and regime interactions. |
| 061 PFNs | A task as a training example, the Beta posterior derivation, proper cross-entropy, analytic grid audits, and two kinds of uncertainty. |
| 062 TabPFN v1 | Context as input, Q/K/V arithmetic, mask induction, historical version scope, timing boundaries, and context permutation. |
| 063 SCM prior | Topological generation from computed parents, a paired fixed-noise edge intervention, observation/target roles, and latent observation processes. |
| 064 TabPFN v2 | Identity-coded reshape traces, two attention axes, receptive fields, query isolation, operation counts, and cache validity. |
| 065 Query embeddings | Label routes through a frozen encoder, cross-fit scatter, linear probes, averaging positions, and fold context size. |
| 066 TabICL | Cell/row/dataset hierarchy, two-stage inducing attention, conditioned affine embeddings, feature identity, and inducing points versus centroids. |
| 067 Local PFNs | The locality hypothesis, geometry, row identities, shared episodes, adaptation evidence, class coverage, and context reuse cost. |
| 068 Temporal PFNs | A generator of drifting tasks, smooth coefficient paths, a paired sign change, legal clocks, and label-delay selection bias. |
| 069 Open environments | Prediction contracts, all-row loss with unsupported classes, schema changes, cost thresholds, failure matrices, and calibration. |
| 070 Foundation checkpoint | Roster/version identity, rank-pool dependence, reconstructed contracts, seed roles, lifecycle break-even, and a falsifiable relational handoff. |

Visible lesson text grows from roughly 1,300–4,000 words to 2,600–6,000 words, depending on topic. Counts in `_depth_delivery_results.json` are diagnostics, not the quality criterion; every addition has a specific explanation or trace purpose.

## Architecture studies

[Open the architecture gallery](../labs/html/architecture-review/index.html).

Sixteen redesigned studies cover fifteen lessons: SAINT, DCNv2, ExcelFormer, Trompt, FT-Transformer, TabR, RealMLP, TabM, OOF ensembling, CountPFN, TabPFN v1, the local axial PFN, query embeddings, TabICL, local PFNs, and drifting PFNs. Evaluation-only lessons keep their protocol visualizations instead of receiving an artificial model diagram.

Each study places the complete numbered forward path beside a worked internal operation. Tensor axes, blocked information routes, intermediate values, outputs, and pictured variants are explicit. Branching graphs are used where branching matters; matrices expose attention masks and reductions. Teal and amber roles are also identified by text, and forbidden matrix cells use hatching. Native HTML reflows on phones. Portable notebook PNGs retain readable labels through horizontal scrolling and link to the responsive view.

The standalone exports were visually inspected at desktop and mobile widths, along with print-media rendering. Integrated checks additionally caught older lesson CSS overriding matrix dimensions and long formulas extending the mobile page. Those issues were corrected in the shared styles. Print-media checks do not establish physical print pagination.

## Regeneration and evidence

Canonical prose lives in `lessons/depth/NNNN.md`. Canonical panel content and routing live in `labs/_architecture_revision.py`; presentation lives in `assets/architecture-atlas.css`. Existing notebook builders call the shared enrichment helpers.

Run from the repository root, using a Python environment with the repository notebook dependencies and Playwright/Chromium for browser steps:

```bash
python labs/_check_architecture_revision.py --export
python labs/_revise_lesson_depth.py
python labs/_lesson_depth.py
python labs/_check_depth_revision.py
python labs/_delivery_check_foundation.py
python labs/_browser_depth_revision.py
python labs/_browser_check_foundation.py
python labs/_browser_check_foundation_access.py
```

The revision operator preserves executed outputs only when teacher code is identical. It does not execute newly generated solutions. Local teacher notebooks follow the existing gitignored convention; they must be built and executed for delivery checks on a fresh checkout. The committed ledger fingerprints saved code/output pairs without depending on an ephemeral backup path.

Validation evidence:

- `_depth_delivery_results.json`: all 24 packages, exact canonical-text agreement, idempotent lesson generation, unique HTML IDs, 1,043 local links, 21 worked calculations, blank students, and 428 unchanged executed teacher cells.
- `_architecture_revision_results.json`: all sixteen standalone studies, desktop/mobile geometry, SVG label bounds, and print-media rendering.
- `_depth_browser_results.json`: actual copied Pages tree, all 24 lessons and 24 prepared labs at 1100px and 375px, native range/select states, loaded images, and keyboard access with JavaScript disabled.
- `_delivery_foundation_results.json`: saved prediction rescoring, pinned source identities, notebook contracts, and the actual Pages workflow's copied file tree.
- `_browser_foundation_results.json` and `_access_foundation_results.json`: prediction/reveal controls, changing computations and reset, plus lesson → lab → exercise navigation and a failed-manifest fallback.
- Existing available L047–057 visual checks and L049–053/L055–057 delivery checks were rerun; no new training was required for this explanatory revision.

Live Colab UI and new paper-scale/Modal runs were not performed. Portable payload integrity, browser-rendered prepared notebooks, model mechanism checks, and full paper reproduction remain distinct evidence levels.
