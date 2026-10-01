# Lesson 162 reproduction contract

Approved 2026-10-01: conceptual lesson and **L162 Relational Context and Scale Audit**, USD 0 cloud. Each local run/check has a 600-second cutoff; numerical libraries use one thread. There is no training dispatch. The standing USD 10 aggregate cap includes preparation, all seeds, retries and validation if a future historical experiment can be specified; this package does not silently spend that budget.

## Named published target: NOT_RUN

Vogel, Hilprecht and Binnig2023, arXiv2305.15321v1, Table 1 wikiTables: BART_table encoder/decoder versus BART_table encoder/decoder+GNN. Three masked reconstruction tasks. Published accuracies (%), not measured here:

| Task | BART_table | +GNN |
|---|---:|---:|
| Missing values |20.75|46.15|
| Column names |66.88|83.91|
| Table names |36.99|37.85|

Known protocol: 10,000 tables;70/20/10 train/validation/test; checkpoint at best validation accuracy; averages over 3 runs. First adapt BART to rows, then freeze the LM while training a GCN with decoded-text cross entropy. Frozen decoder parameters must still permit gradients to graph outputs. Evaluation corpora are single-table wikiTables/gitTables; this does not establish held-out multi-table database transfer. The selected target covers wikiTables and two arms, not every paper experiment.

Unresolved: exact corpus snapshot/subset and splits, seeds, model/tokenizer checkpoint, serialization details, graph construction and decoder interface, masking/objective schedule, optimizer/LR/batching/epochs, decoding/answer normalization and matching code/checkpoints. The prior L159 search plus today's title/arXiv-ID search and inspected primary links did not locate a matching release. Absence is not proven globally. Source bytes and SHA256 are pinned in sources/l162/source-ledger.json; the source HTML is stored alongside it.

Historical fidelity NOT_ESTABLISHED. No arbitrary tolerance or fake paper preset is provided. To unlock: obtain artifacts, hash data/code, resolve every protocol field, build the aligned trainer, estimate complete current-rate costs with margin, and obtain approval for any revised scope. A new implementation based on guesses would require a separately named experiment.

## Complete local experiment: synthetic mechanism audit

Run from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_run_l162.py
timeout 600 .venv/bin/python labs/_verify_l162.py
```

All inputs are pinned in evidence/l162/fixtures.json. There are 16 graph cases (hops 0–3 × star-edge retained/removed × all tables/planet table removed), 8 token cases (four synthetic length vectors × limits 8/1024), and 64 Boolean evidence configurations. No sampling, randomness, fitting, selection or test metrics occur. All cases are evaluated; there is no favorable-case selection.

The illustrative undirected graph has m1/m2→p1→s1 key-like relationships and isolated z1. Edges pass information both ways, not as a statement of released prototype topology. Table restriction forms an induced subgraph BEFORE traversal; it cannot jump through excluded tables. Results are potential information access within at most h hops, not learned influence, predictions or proof of usefulness. No virtual table/column nodes are inferred for the historical model.

Token lengths are supplied synthetic integers, not measured BART tokenization. Pair counts are (sum lengths)^2 for a hypothetical entire-table dense attention input and sum(length^2) for independent rows. A third count applies explicit per-row truncation. These omit heads, layers, projections, feed-forward blocks, padding, decoder and graph cost; they are not end-to-end FLOPs, memory or speed. The1024-token source limit includes the actual model's input convention; real serialization must account for schema and special tokens. Our small 8-token cases are arithmetic exercises.

Six declarations screen whether a transfer claim can proceed to human evidence review. RECONSTRUCTION_ONLY and MULTITABLE_ONLY are scope ceilings conditional on supplied facts, not certified findings. Even all six true produce TRANSFER_REVIEW_ELIGIBLE with transfer NOT_ESTABLISHED. The review must inspect actual provenance, paired predictions, temporal legality and uncertainty.

## Deliverables and evidence limits

Visible functions in relkit/vision_l162.py; three live TODOs in the portable student notebook; solution embeds the same definitions and inputs. Independent matrix-walk and pair-enumeration checks and deliberately broken learner functions provide feedback. The vision map requires human review. Author execution does not establish learner mastery: PENDING_WRITTEN_DEFENSE. Numerical historical reproduction NOT_RUN; historical fidelity and transfer NOT_ESTABLISHED. Browser, copied Pages, live Colab and deployment are separate receipts. No publication is authorized.
