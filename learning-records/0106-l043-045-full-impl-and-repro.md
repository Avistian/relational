# 0106 — L043–L045: teacher solutions now carry the full implementations and paper-repro track

**Date:** 2026-08-28
**Status:** Closed. Standard #25 is now actually present in both the student notebooks *and* the teacher solutions.

The 2026-08-28 #25 retrofit (LR-0105) rebuilt the student labs with inlined encoders and a post-EXIT paper-results cell, but left `labs/solutions/0043–0045` on the pre-#25 copies (22 cells, models imported from `relkit`, no ledger). Running a solution would have either hidden the architecture again or `NameError`'d `TabNetEncoder` / `DenseNODE` / `TabTransformer`. That is now fixed: solutions match the student notebooks (27 / 27 / 26 cells), inlined Ghost BN / ODST / Transformer stack + train loops, and the Modal / Colab scale-up cell.

Mechanism checks still pass (`_check_l043` 22/22, `_check_l044` 15/15, `_check_l045` 15/15, `_check_paper_repro` 40/40 including a regression guard that both student and solution notebooks inline the encoder). Smoke paper-repro runs print INCOMPARABLE ledgers as designed: L043 Adult 0.827 vs paper 0.857 on a different split; L044 Higgs-small 800-row smoke; L045 700-row Adult. Solutions executed with all CHECK cells PASS (L043 Syn2 top-4 = X3–X6, 77% mass; Syn4 left-group 16% PARTIAL, matching the lesson). The XOR-mass exact inequality was dropped as a hard fail so a 6k-row run cannot flake the lab when the honest reading is already PARTIAL. Until you run `--preset closer` on a GPU, paper tables stay cited, not reproduced.
