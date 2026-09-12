# Parent source and implementation review

The parent read the TabPFN-3 report (https://arxiv.org/html/2605.13986v1), including sections 2 and 3.4 and appendices C, E.2 and E.3, alongside the released TabPFN 8.5.0 and TabICL 2.2.0 implementations and actual cached checkpoint configurations. The TabPFN-2.5 report (https://arxiv.org/html/2511.08667v1) and TabICLv2 report (https://arxiv.org/html/2602.11139v1) establish the intervening model identities. The author ledger records the exact primary inputs and immutable versions.

The review led to explicit corrections in the final manuscript and computation figures:

- Axial attention scores scale as N G² + G N C when N=C+Q and only C context rows supply row keys. Quadratic scaling requires stating how context grows. Row compression removes G from the later N C term.
- TabPFN-3 changes to three induced/row stages and row ICL, rather than merely deepening the Nature axial model. Actual configuration: D128, 128 inducing summaries, four CLS embeddings concatenated to 512, 24 ICL blocks, six retrieval decoder heads and maximum class configuration 160. The compared TabICLv2 classifier has 12 ICL blocks and its own classifier/hierarchy.
- Released circular feature-group source offsets are (1,2,4) from the output group index; relative to the first selected feature they are (0,1,3). The [a,b,d] example has to state its anchor convention.
- Retrieval-decoder weighted one-hot context votes precede numerical flooring, logarithms and wrapper calibration. A cardinality-flexible algebra does not supply evidence for a class absent from context or bypass released encoder/configuration limits.
- The large-table envelopes are alternatives: 1M rows × 200 features, 100K × 2000, or 1K × 20K. They do not imply the largest row and feature counts simultaneously.
- TabArena evaluation includes eight-fold bagging and full-data refit; binary AUROC and heterogeneous GPU timings differ from this local log-loss panel. TALENT paper rank aggregation differs from ranking locally averaged seed losses. Five convenience datasets remain five empirical units.
- The relational discussion distinguishes TabPFN-REL within relational foundation models from stronger overall RelGNN results and flags protocol-mismatched baselines.

Independent measurements go beyond checking saved scalars. `070-original-historical.json` and `070-original-current.json` retain fresh original-package validation and test vectors for all 75 pretrained cases, reconstructed from raw data with pinned checkpoints. All 8,910 archived test probabilities match exactly. `070-evidence.json` independently reconstructs all 60 fresh candidate fits with the immutable source-checked corrected TabM architecture, including selection histories and frozen interventions. This verifies the declared local procedures; it does not establish original pretraining or full paper benchmark reproduction.
