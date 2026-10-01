# Lesson 159: reproduction contract

Approved 2026-10-01. Two distinct lanes; completion of one never completes the other.

## Named published target — NOT_RUN

Vogel, Hilprecht and Binnig (2023), *Towards Foundation Models for Relational Databases [Vision Paper]*, arXiv2305.15321v1, Table1, wikiTables: BART_table encoder/decoder baseline versus BART_table encoder/decoder + GNN. Three tasks: missing values, column names, table names. Primary source: https://arxiv.org/html/2305.15321v1#S4.

| Task | Published row baseline accuracy (%) | Published +GNN accuracy (%) |
|---|---:|---:|
| Missing values |20.75|46.15|
| Column names |66.88|83.91|
| Table names |36.99|37.85|

These numbers are transcribed published targets, not local measurements. Paper reports means across three runs; no per-run scores or SD are provided in this table. No numerical tolerance is declared because protocol identity is unresolved. Relative gains are not percentage-point gains.

Known: 10,000 tables per corpus;70/20/10 train/validation/test; best validation accuracy checkpoint; three runs; BART row adaptation before GCN training; masked cell/column/table reconstruction; frozen LM in the GNN stage; decoded text cross-entropy. Paper also evaluates gitTables and two initialization ablations, outside this selected target. Evaluations use single-table corpora, not unseen multi-table database transfer.

## Missing protocol details

- Exact 10,000-table subset, corpus snapshot hashes and split IDs/seeds.
- BART variant and checkpoint revision; tokenizer and serialization details.
- Graph construction details, representation-to-decoder interface and GCN widths/depth.
- Mask sampling rates, objective mixture, corruption semantics and caching policy.
- Optimizer, LR schedule, epochs, batch sizes and stopping rules; exact run seeds.
- Text decoding, answer normalization and accuracy aggregation conventions.
- Publication implementation, training logs and released checkpoints.

`labs/sources/l159/manifest.json` pins retrieved bytes (2026-10-01). We inspected the full arXiv HTML, its links, the public DataManagementLab repository listing and targeted title/author/code searches. No matching release was located in these searches. This is a bounded search finding, not proof that code does not exist. The inspected LukasZahradnik/deep-db-learning repository concerns a different paper; it is explicitly rejected as a reproduction source. The curriculum's old attribution was wrong.

There is NO runnable historical benchmark trainer in this package. `python labs/_paper_l159.py` reports the gate and exits2. It does not relabel the synthetic trainer as a paper preset. To unlock: obtain and hash the missing artifacts, audit the actual released model and protocol, implement/replay them, estimate all runs at current compute rates, then obtain any required paid-run approval. A larger synthetic run cannot unlock this gate.

## Executed synthetic mechanism — COMPLETE, INCOMPARABLE to Table1

Command from repository root: `.venv/bin/python labs/_run_l159.py`.

- TierC, generated in visible `make_examples`;120 independent four-row tables, split seed159,84/24/12 table-disjoint train/validation/test. Each table supplies three masked-target examples;252/72/36 examples by split. No graph crosses a table boundary.
- Three independent binary target levels; four noisy context observations agree with each level70% of the time. Cell class is shared by rows by construction, making neighboring unmasked cells particularly informative. This built-in redundancy intentionally favors graph context and is not an empirical database claim.
- Semantic IDs distinguish a shared table/column name from distinct cells. Corruption removes every occurrence of the selected identity BEFORE embedding. Unrelated equal-valued cells remain legitimate context; there is no clean-embedding cache. Vocabulary and generation rules are fixed by the synthetic design, not fitted to test data.
- Codec:14-token embedding vocabulary,16-dimensional mean pooling and tanh affine row encoder; three binary linear heads. Graph: one learned16×16 GCN-style transform of complete-table mean aggregation with self edges, tanh, same frozen decoder. This is not BART, not a language model and not the paper's unspecified graph interface.
- Three paired seeds0/1/2. Stage1 trains codec80 full-batch Adam updates, lr.025. Stage2 freezes the selected codec and trains graph80 updates with the same optimizer/LR. Both select highest mean validation accuracy across selected targets; earliest tie. Test is evaluated after both stages are selected. Graph receives additional training and context: no claim of matched total training compute or an isolated architectural effect.
- Loss is mean binary cross-entropy across the selected task per example. Binary accuracy is not generated-text reconstruction accuracy. Every test example has a stable `(table,task)` key.72 predictions per seed (36 per arm);216 total.12 test examples per task per seed.
- Checkpoints, complete histories and logits in `evidence/l159/mechanism.json`. Reports show sample SD across three seeds, conditional on one synthetic generator and split. Zero seed SD here is not zero uncertainty. No cross-database transfer experiment was performed.

Independent checks: `.venv/bin/python labs/_check_l159.py` and `.venv/bin/python labs/_verify_l159.py`. The verifier reconstructs every saved logit with NumPy, rescores labels, checks split/key coverage and best-validation selection, tests semantic target counterfactuals, and rejects three learner mutations. Notebook execution additionally reruns all480 updates from an empty directory.

## Budget and cutoffs

Additional cloud spendUSD0. No cloud job, model checkpoint download or paid API call. Local experiment: one CPU thread,120-second aggregate training cutoff per execution; timeout raises INCOMPLETE rather than reducing seeds/epochs. The author run, independent forward verification and portable solution execution are separate checks, not new scientific replication cohorts. Runtime/environment receipt is in `evidence/l159/runtime.json`. StandingUSD10 includes every future seed/retry/preparation/validation/overhead charge; it is not an authorization to dispatch the under-specified historical benchmark. Full benchmark cost NOT_ESTIMABLE from located information.

## Delivery and evidence status

Source audit: selected paper identified, release/protocol gaps recorded. Local mechanism: see generated report and checks. Historical target NOT_RUN; paper fidelity NOT_ESTABLISHED; cross-database transfer NOT_TESTED; learner PENDING_WRITTEN_DEFENSE. Browser/notebook checks have separate receipts. Live Colab and deployment NOT_CHECKED. The missing fourth-year portfolio task and historical availability evidence from L158 remain unresolved.
