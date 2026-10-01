# Lessons 161–170: sequential teaching and publication review

Reviewed 2026-10-01. Scope: continuity, self-contained explanations, model-specific architecture, notebook alignment, and actual Pages delivery. This is an editorial and delivery review of the existing approved experiments, with no new paid compute or training.

## 161 · What is a foundation model?

**Finding:** A sound bridge from L159's reconstruction objective and L160's evidence gates into reusable pretraining. The four adaptation routes and database-boundary example are distinct and useful. Gradient/forward-pass terminology needed a short explanation before the table.

**Revision:** Added that explanation and linked the actual L162 successor. Retained the two protocol diagrams; this conceptual lesson does not introduce a model needing a new architecture. Its inventory classifier checks declarations, not undisclosed pretraining or transfer performance.

## 162 · The relational FM vision

**Finding:** The moon → planet → star example clearly motivates relational context after L161. The BART → graph → decoder figure contains the training boundary and unresolved interface, but its right side was clipped at desktop reading width.

**Revision:** Widened the desktop figure, added an explicit forward-path/learning-path reading route, and repaired compressed arithmetic prose. The graph example and token-pair calculation make the computation visible. Historical wikiTables reproduction remains NOT_RUN; the exact graph/decoder interface is not invented.

## 163 · LM encoders for rows

**Finding:** Directly addresses L162's unanswered row-representation choice. Its side-by-side text/typed architecture is model-specific and shows shape, preprocessing, fitted heads, and shared splits. Standard deviation, broadcasting and sequence-token acronyms were insufficiently unpacked.

**Revision:** Added units and a plain explanation of both padding masks and BOS/EOS, and repaired spacing. The numeric worked vector and masked mean remain coherent with the diagram. The synthetic target favors the typed baseline; no general LM ranking or historical reconstruction claim is added.

## 164 · Griffin

**Finding:** Strong continuation from L163's fixed row summary to repeated task-conditioned cell reads. The architecture, two-coordinate attention trace and relation mean/max example expose real operations. The opening was demanding for a learner returning after a gap.

**Revision:** Added a reading route, plain intuition, shape/key/hop/cutoff reminders and definitions of feedforward updates/projections. Repaired compressed protocol notation. Kept the existing detailed architecture and separate training figure. The selected 20-fit experiment remains INCOMPLETE_BUDGET_GATE; source checks are not transfer scores.

## 165 · KumoRFM in-context relational learning

**Finding:** Clearly contrasts Griffin's weight updates with input-based adaptation. The two-clock example is essential and well worked. The closing paragraph incorrectly sent the main sequence to OpenRFM; the architecture was clipped on desktop.

**Revision:** Linked the actual next lesson, RDB-PFN, identified OpenRFM as optional, explained ICL/rooted subgraphs before the diagram, and widened the desktop figure. Retained the distinction between within-graph and across-example labels. Proprietary historical reproduction stays NOT_RUN / NOT_ESTABLISHED.

## 166 · RDB-PFN synthetic relational priors

**Finding:** Answers where reusable ICL weights could come from. The generator, DFS, two attention axes and selected checkpoint evidence connect well, but PFN/SCM/MLP/GELU/logit/imputation terms accumulated quickly. The architecture was clipped on desktop.

**Revision:** Added staged reading and short vocabulary/intuition explanations, clarified the two attention operations plus feedforward update, repaired spacing, and widened figures. Preserved the attention-mask diagram and portable architecture. Complete selected checkpoint evaluation remains distinct from fresh pretraining and historical feature reconstruction.

## 167 · Tabular → relational FM transfer

**Finding:** Productively revisits L166's shared DFS inputs to separate representation, prior and adaptation. The three predictor paths have distinct internal operations. Switching from L165's inclusive label-arrival rule to this lesson's strict-before fixture could look like an inconsistency.

**Revision:** Added a three-decision gloss and an explicit cross-lesson boundary comparison with a day-5/day-10 counterexample. Both conventions retain the context example's own feature cutoff. Kept the count/mean collision and paired-evidence visualization. The 30 evaluations are reused evidence, not new inference or ten databases.

## 168 · Cross-database generalization

**Finding:** Correctly changes the evidence unit after L167's one-task replay. Architecture/feature compatibility is separated from verified exclusion and broad transfer. The two-database macro example is useful, but statistical terminology needed unpacking.

**Revision:** Added an intuitive unseen-database explanation and definitions of estimand, macro aggregation and SD. Kept the shared-input architecture, exposure/adaptation protocol and measured comparison. Two selected databases do not establish a general advantage or exact schema exclusion.

## 169 · Scaling laws and open questions

**Finding:** Naturally varies labeled context after L168's fixed-512 comparison. The lesson carefully distinguishes context response from pretraining scale and preserves the nonnested source sampler. The next-lesson link went to the curriculum rather than L170.

**Revision:** Added a reading route and intuition before the power-law expression; linked L170 directly. Kept the axis map, curves and signed gain plot: this is an evaluation lesson, not a new architecture. The full selected grid remains 300 evaluations, with 60 reused and 240 originally fresh; no new inference was run in this review.

## 170 · FM design checkpoint

**Finding:** Synthesizes the preceding lessons into representation/prior/adaptation choices and a falsifiable design. The paradigm figure incorrectly called Griffin's output head target-trained, contradicting L164's frozen decoder.

**Revision:** Corrected the figure to “Frozen decoder on adapted state,” identified shared-weight updates, and added the same explanation beside it. Rebuilt the portable notebook figure. Kept paired reversals, missing-evidence gates and the written-defense rubric. Computational replay does not certify learner mastery, full pipeline validity or paradigm superiority.

## Sequence and verification boundaries

The main path is L160 → definition → relational vision → row encoding → graph transfer → graph ICL → synthetic relational prior → representation/prior comparison → another database → context-size response → defended design. Optional bridge lessons are not prerequisites. Examples progress from inspectable synthetic arithmetic to scoped released-checkpoint evidence, then evidence-based design.

Model-specific diagrams already cover inputs, internal operations, output and training/inference boundaries. The review fixes desktop presentation and one scientific label instead of replacing these with generic box diagrams. Phone figures retain local horizontal scrolling, focus, captions and adjacent text explanations; the page itself must not overflow.

`notebooks.json` records exact unchanged executable cells and preserved successful saved outputs after prose/figure rebuilding. This review does not rerun training or upgrade historical execution receipts. Per-lesson contract and browser-delivery logs, clean Git-index links, and desktop/mobile/print/no-JavaScript checks are recorded separately. Publication is verified against a real workflow and live bytes in the publication record. Live Colab and personal written defenses remain outside this delivery claim.
