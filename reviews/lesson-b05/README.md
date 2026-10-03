# B05 delivery review

Prepared 2026-10-03 under the approved design. Learner status **PENDING_WRITTEN_DEFENSE**.

- [Lesson](../../lessons/b05-tabdpt-real-data-retrieval.html)
- [Student notebook](../../labs/b05-tabdpt-real-data-retrieval.ipynb)
- [Executed solution](../../labs/html/b05-tabdpt-real-data-retrieval.html)
- [Reference](../../reference/b05-retrieval-episodes.html)
- [Reproduction contract](../../labs/b05-reproduction.md)
- [Portable archive](../../labs/evidence/b05/reproducer.zip)

## Scientific result

Complete course matrix: 18 real-column episodes, 144 predictions, independent scalar neighbors and keyed scores, all target-reversal selection checks unchanged. The predictor is a four-neighbor mean, not trained TabDPT. Eight corrupted evidence cases and three wrong learner implementations were rejected.

Pinned released `use_knn` method, executed unchanged with an exact L2 index oracle replacing FAISS, changes neighbors from [0,2,4] to [0,1,3] under a target-only intervention. Paper-order course selection stays [0,1,2]. This is scoped source behavior, not historical training identity or measured benchmark effect.

Original checkpoint bytes matched their Git LFS hash. Original architecture has four attention heads; later v1.1 metadata has eight. Original-result checkpoint/evaluator mapping, exact selected fold artifacts and references/RNG remain unresolved. **B05-TABDPT-BANKNOTE-TWO-FOLD: INCOMPLETE_SOURCE_PROTOCOL.** Zero benchmark inference runs. Full pretraining, full benchmark and Turbo/v1.3 execution NOT_RUN. No downscaled substitute presented as reproduction. The command is an executable refusing preflight; a complete benchmark runner is not claimed.

The full declared training-ID versus classification-test-ID audit finds no intersection. Content/derived-dataset independence remains NOT_ESTABLISHED. A newer version's performance is not original-paper parity.

## Delivery evidence

The solution executes 10 code cells from an empty directory with exact report parity. The blank student notebook fails at its first live TODO. Full original inference source and released training sampler are visible inline. Source AST parity, deterministic builder, independent archive execution and changed-source rejection pass.

Actual Chromium checks cover 24 retrieval states across desktop/375px widths and six quiz states, keyboard/reset, print and no-JavaScript fallback. Model diagram and actual mobile screenshots were inspected; final distance-cell wrapping was corrected. Manifest navigation and 30 local links pass.

Actual workflow Build site script runs on a copied checkout from a temporary Git index, authenticating all 131 sealed B05 publication files and 30 copied-site links. Existing B01–B04a publication dependencies are included only in that temporary index; prior lessons are not re-certified. User index remains unchanged by the publication check.

Receipts: `labs/_verify_b05_results.json`, `_execution_b05_results.json`, `_delivery_b05_results.json`, `_checkout_b05_results.json`; numerical accounting in `labs/evidence/b05/local-budget.json`. The final budget receipt is `final-summary.json` in this review directory.

Only the approved design was committed. Package is local; no push or deployment. Live Colab NOT_CHECKED. No external messages sent. Author checks are not learner mastery.
