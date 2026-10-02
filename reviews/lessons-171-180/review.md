# Lessons 171–180: individual teaching and publication review

Reviewed 2026-10-02. Scope: continuity, prerequisites, explanation, architecture and actual delivery. Editorial review of existing experiments; no fresh model training, benchmark inference or paid compute. Lesson 179 is absent and cannot be certified as reviewed or published.

## 171 · Corpus of databases

**Finding:** A clear answer to L170's unresolved corpus question. Definitions, the copy/bridge/tail worked example, and the foreign-key counterexample make it self-contained. The three diagrams distinguish lineage, observed relationships and time windows; a model architecture is not applicable. Desktop figures slightly exceeded their reading column.

**Revision:** Linked L172 directly and widened the desktop figure area. Retained the conservative connected-component policy and the distinction between source declarations and audited rows. Seven-source inventory is complete; only F1 has the full row audit. Training and other-six row audits remain NOT_RUN.

## 172 · Schema tokenization

**Finding:** Naturally turns L171's declared corpus into model inputs. The same integer serving as ID, count or category gives a concrete reason for semantic types. Training-only fitting and the VALUE/MISSING/UNKNOWN/MASKED states are explained before the lab. The real points-cell trace exposes the numerical operation rather than merely naming a tokenizer.

**Revision:** Added a direct bridge to L173's objective and widened figures. Preserved exact key identity, local vocabularies, explicit untimed defaults and the distinction between transforming a snapshot and admitting query-time information. This is an input contract, not a learned model architecture.

## 173 · Multi-task pretraining

**Finding:** Strong continuation from typed inputs to context, target and loss. The architecture shows two separate context pools, a shared MLP, task-specific heads, dimensions and loss aggregation. Embedding/projection/MLP/logit terminology arrived too quickly for a returning learner.

**Revision:** Defined those building blocks and shape symbols before the diagram; added intuition before the two loss means and linked L174. Enlarged the figure area for readable operations. Retained the explicit autocomplete scope, six-fit results and constant baseline. One-database reconstruction is not transferable foundation-model evidence.

## 174 · Fine-tuning protocol

**Finding:** Uses L173's disappointing baseline result to ask whether pretraining actually helps. Four update policies, identity initialization, an adapter derivative trace and retrospective split caveats connect well. The desktop architecture clipped the output head; the derivative example assumed fluent gradient/activation vocabulary.

**Revision:** Explained residual correction, ReLU, matrices, offsets, gradients and the arriving derivative before the formula. Widened all figures so the complete forward path and controls are visible. The adapter remains the specific 32→8→32 module in the course model, not LoRA or the original BERT experiment. Prior test exposure remains explicit.

## 175 · Zero-shot evaluation

**Finding:** Correctly changes from updating familiar-task heads to evaluating a publicly released relational model. The RT-v1 architecture shows four attention relations inside repeated blocks. Its timeline makes label-window readiness distinct from row time. The diagram uses internal operator names without enough inline recall.

**Revision:** Added a forward-path reading route and short explanations of attention, masks, heads, residuals, RMSNorm and SwiGLU; widened figures. Preserved the observed temporal stop and absence of model scores. Future schedule dates violate the declared event-time rule; absent arrival history does not prove outcome leakage.

## 176 · Few-shot ICL evaluation

**Finding:** Explicitly explains the switch from stopped RT evaluation back to RDB-PFN. Nested prefixes answer the composition confound left by L169. Its model-specific attention matrix shows support/query information flow, and its sampling figure carries one concrete ordered draw through increasing budgets. Attention “keys” could be confused with relational keys.

**Revision:** Defined shape symbols, attention masks, attention keys/values, GELU and FFN before the architecture. Widened figures. Retained support-fitted preprocessing as part of the measured pipeline, paired support-draw variability, separate replay/intervention tracks and the inherited exact-repeatability limitation.

## 177 · Compute budget realism

**Finding:** Connects L176's accuracy/context trade-off to complete protocol costs. Distinct timers, estimates, reservations and missing invoices are unusually clear. The accounting figure exposes nesting instead of inviting double-counting; the routes figure is an accounting map, not a new model architecture.

**Revision:** Added a direct successor link to L178 and widened the three figures. Retained historical price-snapshot wording and non-comparable workload boundaries. No present-day quote, human-time measurement or provider-enforced cap is inferred from the saved ledger.

## 178 · Fair model comparison

**Finding:** Combines L176's information contract and L177's budget contract. The three routes have distinct operations; label complement and NaN gradient traces expose concrete reasons a leaderboard may be invalid. Forward/backward terminology needed a short refresher. The ending misleadingly linked “Lesson 179” to a generic curriculum page.

**Revision:** Defined both passes, gradients and finite/NaN values before the failure. Widened clipped route/evidence figures. Marked 179 pending and linked the usable L180 continuation. Complete saved Table 9 replay remains separate from the incomplete fresh comparison; no winner is invented.

## 179 · Failure modes

**Finding: MISSING.** The curriculum names schema shift and cold start, but there is no lesson page, source, notebook or manifest entry. No pedagogical, architecture or execution review is possible.

**Revision:** Marked the curriculum entry as planned/not yet available in both source and web curriculum. L178 now discloses the gap; L180 already recalls the necessary failures from existing lessons. The release contains nine reviewed lessons, not ten completed lessons. Creating 179 remains separate work.

## 180 · Public encoder checkpoint

**Finding:** Clearly synthesizes course pretraining, adaptation, temporal validity, cost and training-health gates. Its RT figure distinguishes frozen text inputs from trainable relational operations, but was substantially clipped on desktop. The update trace assumed familiarity with decoder/logit/loss/optimizer terms.

**Revision:** Added those definitions before the trace and widened the architecture. Preserved the caveat that no public checkpoint weights were loaded here, the full saved-context audit, the temporal failure, and the over-budget full schedule. Fresh public-encoder fine-tuning remains NOT_RUN and the practical exit INCOMPLETE.

## Sequence and evidence boundary

The usable route is L170 → corpus identity → typed inputs → objective → adaptation → zero-shot access → few-shot support → whole-run accounting → comparison validity → L180 evidence checkpoint, with L179 explicitly pending. Optional internal-operator derivations are separated from the core reading route. Existing diagrams were retained because their operations, attention masks, update boundaries and evidence lanes are specific to the taught mechanisms; the display fix makes their entire paths visible.

Notebook executable cells and saved outputs are preserved, with hashes recorded in `notebooks.json`. Per-lesson contract and delivery results are recorded separately from clean-index build, browser and live publication checks. Historical experiment status is not upgraded by successful delivery. Live Colab and learner written defenses are outside this publication claim.
