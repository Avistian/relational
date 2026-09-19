# Lesson 074 — CARTE cross-table transfer

Authorized: planned lesson with lab and local verification; no deployment or learner completion.
Design: follow heterogeneous cell values through string vectors, column-conditioned star graphs and the released CARTE readout. Source-check a visible PyTorch implementation with copied pretrained weights. Use the released YAGO checkpoint and real FastText vectors on three released wine tables with different schemas, paired small target splits and scratch controls. Explain knowledge-graph-to-table transfer separately from supervised source-table mixing. Teach paper/release discrepancies explicitly.

Alternatives considered: synthetic string vectors would isolate mechanics but could not test pretrained CARTE; a full 51-table benchmark is broader than this unit. Choose actual checkpoint transfer on capped real tables, with full benchmark INCOMPARABLE and larger CPU follow-up.
Deliver manuscript, portable model architecture, interactive row graph, attention arithmetic, source-visible student/solution notebooks, reference, evidence, provenance and a reproduction contract. Test graph masking, permutation invariance, grouped attention and source output parity, execute the solution, inspect browser desktop/mobile, validate copied Pages links. Student completion remains unscored.

Implementation finding: Wine Poland seed2 has a constant observed training-volume column. A raw power fit produced exponent35 and float32 overflow. Final policy uses standard scaling for train-constant columns and omits train-all-missing columns; all varying columns retain a train-only power transform. The whole comparison reruns under this policy. Regression evidence is stored separately; model source parity remains a distinct check.

Execution optimization: preserve the complete forward pass for source parity, but compute only consumed center outputs in training/embedding extraction. Row-output and parameter-gradient parity passed (max gradient difference 2.384185791015625e-07). Final author and notebook runs use this identical path.

## Final verification

- Source graph, full encoder and center-only output/gradient checks passed.
- The numerical regression fixture and all default numeric splits passed.
- 54 evaluations completed; all 21 solution code cells executed and freshly reproduced every saved record exactly.
- Five portable figures inspected; desktop1100/mobile375 browser interactions and horizontal keyboard scrolling passed without JavaScript errors or page overflow.
- Copied Pages workflow checked 39 package links and byte-verified the cached data/vector/checkpoint files.
- Student TODOs remain blank; solution outputs are retained in the local teacher notebook.
- Paper benchmark INCOMPARABLE; larger run NOT_RUN; live Colab and deployment NOT_CHECKED. No commit, publication or learner-completion claim.
