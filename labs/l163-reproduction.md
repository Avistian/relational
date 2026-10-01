# Lesson 163 reproduction contract

Approved 2026-10-01: **L163 Frozen Row Encoder Comparison**; USD0 cloud, 1800 seconds aggregate local preparation/compute/checks, including retries. Budget events are recorded in `evidence/l163/budget.json`; 300 seconds reserved for checks. Single numerical thread, CPU float32, batch 8. Stop with INCOMPLETE on timeout/resource trouble; no silent row/seed reduction. Standing future reproduction cap: about USD10 for all seeds, preparation, retries and validation combined. No paid execution or publication authorized.

## Full historical target — NOT_RUN

Vogel, Hilprecht and Binnig (2023), [arXiv2305.15321v1 §4/Table1](https://arxiv.org/html/2305.15321v1#S4), wikiTables BART_table versus BART_table+GNN: masked cell values, column names and table names. Published accuracies respectively 20.75/46.15%, 66.88/83.91%, 36.99/37.85%. These are external targets, not local results. Known: 10,000 tables, 70/20/10 train/validation/test, three-run means and best-validation-accuracy checkpoints. Adapt BART on rows, freeze LM weights, train GCN through BART decoder cross-entropy.

Unresolved: matching code/checkpoints, exact corpus version/subset/splits, seed IDs, model/tokenizer checkpoint, serialization, masking schedule, graph topology/decoder interface, optimizer/LR/batch/epochs, decoding and answer normalization. L159's public repository inspection is retained; L163 refreshes the primary paper, model metadata and public DataManagementLab listing. No matching historical release was identified in these inspected sources; global absence is not proven. Source bytes, URLs and retrieval hashes are in `sources/l163/source-ledger.json`.

Historical fidelity NOT_ESTABLISHED. No speculative historical trainer or arbitrary accuracy tolerance. To unlock: obtain missing artifacts, freeze a complete source/data/protocol contract, implement the aligned trainer, price all seeds and overhead with reserve, and approve revised scope. Whole-paper reproduction also includes other corpora/arms; the named wikiTables comparison alone is narrower.

## Fully specified local course experiment

The approved design in `docs/plans/2026-10-01-lesson-163-design.md` preceded outcomes. `fixtures.json` pins all 240 rows, seed163 generator, three split assignments (seeds0/1/2), five alphas and immutable BART revision. Each split has144/48/48 rows. The deliberately additive target is `2*price_usd+5*weight_kg+condition_offset+colour_offset+Normal(0,2)`; condition offsets new8/used−4/refurbished2, colour offsets red3/blue−2/green0. This favours a linear numeric/one-hot baseline and does not measure natural-language understanding. IDs and labels are excluded from both encodings.

**Text:** ordered compact JSON, table='products', then price_usd, weight_kg, colour, condition. Strings use JSON escaping; numbers retain their values. Baseline, reversed field order and opaque field names c0..c3 retain the same values. Frozen facebook/bart-base, revision recorded in fixtures; model files individually SHA256-pinned in encoding receipt. Encoder only, six layers,768 hidden features, final hidden-state mean over attention_mask=1 including BOS/EOS; PAD excluded. BART remains eval/frozen; no decoder/fine-tuning. No truncation; >1024-token rows raise an error. Unknown numerical/category formats are rejected rather than coerced silently.

**Typed:** train-only numerical population mean/SD (constant scale→1) and sorted train-only category one-hot vocabularies (unseen→all-zero block). This is a transparent typed feature baseline, not PyTorch Frame neural training. Canonical field identities are preserved when presentation names/order change. Such invariance is designed into this baseline, not learned transfer.

**Prediction:** head fitting/scoring uses float64 (encoder/cache remain float32). Independent ridge verification exposed a 0.000462 maximum prediction discrepancy from mixed-precision head fitting; the head path was made consistently float64, with no grid/data/protocol change. Both paths use train-only StandardScaler and Ridge with unpenalized intercept/SVD solver; alphas0.01/0.1/1/10/100. Lowest validation MAE wins, first grid entry on ties. No train+validation refit. Six selected heads and their preprocessing are written and hashed BEFORE test evaluation. Reordered/renamed test inputs use unchanged baseline heads; no intervention-based retuning. Archive all864 `(seed,method,variant,row_id)` predictions. Report all18 test scores, aggregate means/sample SD, train-mean baseline and paired absolute prediction changes. Overlapping split SD is descriptive, not a confidence interval. Higher feature dimension and prior pretraining differ across paths; this is not equal model capacity or equal pretraining compute.

A frozen pretrained encoder can process all input rows before splitting because no target or corpus-fitting is used in that step. Learned preprocessing/head fitting stays train-only. This does not establish the historical pretraining corpus's identity or contamination status.

## Executable lanes

From repository root, using the pinned environment (`labs/l163-requirements.txt`; CPU torch installation is platform-specific):

```sh
# Fresh encoder lane: downloads approximately 560 MB of model weights once.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python labs/_budget_l163.py .venv/bin/python labs/_encode_l163.py
# Complete head fitting and scoring from verified stored embeddings.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python labs/_budget_l163.py .venv/bin/python labs/_run_l163.py
# Independent numerical/key/provenance checks.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python labs/_budget_l163.py .venv/bin/python labs/_verify_l163.py
```

`_prepare_l163.py` documents fixture generation and source retrieval; do not rerun it when replaying pinned evidence, because upstream metadata can change. `_encode_l163.py` deliberately performs fresh encoding; `_run_l163.py` refuses modified input/embedding/helper hashes. Cached replay is not fresh encoding. The standalone notebooks embed all small evidence and readable operators; their default lane refits all heads from the hashed cache without downloading a model. A separate explicit opt-in cell executes the full frozen-encoder lane at the pinned revision. That run is user-controlled and is not included in the stored author receipt.

## Evidence boundaries

Author execution, independent numerical verification, portable notebook execution, browser checks and Pages build are separate receipts. Local comparison is synthetic; historical reproduction NOT_RUN, historical fidelity/general encoder superiority/relational transfer NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE. Live Colab and deployment NOT_CHECKED. No push or publication.

Local cost accounting: `evidence/l163/timing.json` records all240-row typed fit/transforms separately for each split and BART tokenization/forward/pooling for each input variant. These are single local elapsed observations, not equalized throughput benchmarks. Notebook default replay also repools24 real rows (first8 per variant) from stored token states, while the other features are cached; it is not a new encoder pass.
