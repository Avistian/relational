# B05 reproduction contract

Approved 2026-10-03. Named target **B05-TABDPT-BANKNOTE-TWO-FOLD**: TabDPT original-paper checkpoint, OpenML dataset1462/task10093, official TabZilla folds0/1, complete test populations, train+validation support, context setting2048, eight ensemble members. Main metrics accuracy and ROC-AUC; log loss supplementary. No score-dependent selection. Banknote has fewer than2048 total rows, so this setting cannot test active neighbor pruning at inference.

## Current status: INCOMPLETE_SOURCE_PROTOCOL

88 frozen source files are hash-authenticated. Original v1.0 Git commit5214f9267b49d3907b074be9d90a660054484416 and original312,440,778-byte checkpoint were recovered; SHA2567d002b557cf37ff038d209ebfc515be57f92a37ef7b1c3ab5268b39831b47658 matches Git LFS. Its model configuration has16layers,768width,4heads,100features,10classes. Weight bytes are cached outside the publication package; the archived pointer and download receipt preserve recovery instructions.

Paper-era v1.1.12 commit eb5c0d9ff303d8fd8a053d9ac8be21af0d4e9157 loads tabdpt1_1.safetensors. Its authenticated repository metadata and fetched safetensors header show8heads. The header is not a full-weight hash verification. Original configuration has dropout0.1 while paper Appendix C says0; the original custom model's dropout argument is unused. This does not establish original pretraining settings. Do not equate chronology with result identity.

Unresolved: paper-result mapping across original/v1.1 checkpoint and evaluator, original TabZilla banknote fold bytes/row identities, per-fold reference scores/predictions and ensemble RNG mapping. The public Drive archive page was reached; exact selected fold artifacts were not recovered. The newer evaluator's default seed0 is not proof of the paper's historical ensemble RNG. No inference was dispatched. These are unresolved requirements, not proof that the artifacts cannot be recovered.

Run from the repository:

```sh
.venv/bin/python labs/_reproduce_b05.py
.venv/bin/python labs/_reproduce_b05.py --run
```

The first authenticates all frozen source evidence and reports missing requirements. The second deliberately exits nonzero. It is an executable preflight, **not a complete benchmark runner** while the contract remains unresolved. The upstream full evaluator/source is included. Completion requires recovered identities, a pinned compatible environment, immutable keyed predictions, independent scoring and an affordable complete-two-fold forecast; do not substitute random splits or current weights.

## Confirmed source discrepancy

Training commit af0340c5cdebe2ceb6b94c09d6f9564ee80b89df constructs its FAISS index from all standardized columns including the original target. `use_knn` selects neighbors before removing the randomly chosen SSL target. Paper Algorithm1 selects from the target-removed view. The unchanged method AST, executed with an exact L2 oracle replacing FAISS, selects[0,2,4] then[0,1,3] when only target values change; the course paper-order path selects[0,1,2] both times. This proves a dependency in the released method under that input. Historical training identity and benchmark effect remain NOT_ESTABLISHED. No silent upstream repair.

## Complete separate course experiment

**B05-COURSE-REAL-COLUMN-EPISODES** uses all178 Wine feature rows (13columns; class labels unused), targets0/6/12, seeds0/1/2 and episode sizes32/96. Each episode has8queries; remainder support. Target removed before support-population-standardized squared-L2 retrieval, identity ties, seeded partition without replacement. Four-neighbor target mean is the readout, not TabDPT. Eighteen episodes/144predictions; all checked with an independent scalar distance/scoring oracle. Target-reversal intervention keeps all18selected identity lists fixed. Targets have different units and episode sizes change queries; do not aggregate MSE across targets or infer that more support improved performance. This diagnostic establishes no learned-model transfer.

```sh
.venv/bin/python labs/_run_b05.py
.venv/bin/python labs/_audit_b05.py
.venv/bin/python labs/_source_probe_b05.py
.venv/bin/python labs/_verify_b05.py
```

The123declared training IDs and72classification test IDs have no exact intersection; banknote1462 is absent from training IDs. This metadata audit does not exclude renamed/derived data. Appendix B.1 additionally describes hashes, dimensions, feature/target statistics and manual review. Raw-content overlap audit NOT_RUN; independence NOT_ESTABLISHED.

## Resources and scope

USD10 total cap; stop commitments atUSD8 withUSD2reserve. Local numerical cap3600s including preparation, errors and verification, enforced by `_budget_b05.py`. USD0cloud/API; no paid dispatch while source gate is closed. Full pretraining, full benchmark, Turbo andv1.3 inference NOT_RUN. Saved replay, fresh course mechanisms, original weight identity, paper-result parity and learner mastery are separate. Learner PENDING_WRITTEN_DEFENSE. No deployment; liveColab NOT_CHECKED.


## Archived-page sanitization (2026-10-04)

The saved Google Drive folder page contains Google web-application configuration. Eleven API-key occurrences were redacted before repackaging the source archive and portable notebooks. `sources/b05/manifest.json` records the original and sanitized page hashes and the replacement policy. This page is now explicitly a sanitized capture, not byte-exact original HTML. No dataset, prediction, metric or reproduction status changed. The source gate and notebook builder reject a reintroduced Google API-key pattern even with an updated checksum.
