# Lesson 163 Implementation Plan

Approved in chat 2026-10-01: Lesson 163, LM encoders for rows, plus **L163 Frozen Row Encoder Comparison**. Execution continues in the existing teaching workspace to preserve the staged L161/L162 material. No publication or paid runs.

**Goal:** encode identical rows as text and typed features, compare downstream predictions, and defend the information and evidence trade-offs.

**Architecture:** pinned synthetic fixtures feed a frozen BART encoder or train-fitted numeric/category features, then equally selected ridge heads. Human-readable functions are embedded in portable notebooks; cached, hashed embeddings support cheap full head replay, and an explicit fresh-encoding lane remains runnable. Canonical Markdown and reusable assets generate HTML and reference material.

**Tech Stack:** Python, NumPy/scikit-learn, CPU PyTorch, pinned Transformers/BART, nbformat/nbclient/nbconvert, matplotlib, Playwright.

## Frozen experiment protocol (before observing results)

- 240 synthetic product rows. Inputs: price_usd, weight_kg, colour, condition. Seed 163 generates values; target is 2*price_usd + 5*weight_kg + condition offset + colour offset + Gaussian noise(sd=2). This construction favours numeric/typed structure by design; it is not a natural-language or universal model benchmark. Row IDs and target never enter either encoding.
- Independent split seeds 0,1,2; 144 train/48 validation/48 test, full partition per seed. Same rows and assignments for both methods. Archive every prediction and ID. Repeated splits overlap; SD is descriptive, not a confidence interval.
- Text is schema-aware ordered JSON (table name plus named values). Three versions: baseline; reversed column order; opaque column names. Only baseline trains heads. Interventions apply at inference, keep values/row IDs fixed, and use frozen baseline heads. No test-based tuning.
- facebook/bart-base model/tokenizer pinned by immutable revision and hashes. Frozen CPU float32, eval mode, six encoder layers, d=768. Final hidden states mean-pooled across non-padding tokens (including BOS/EOS); no decoder or fine-tuning. Right padding; no truncation; reject input exceeding 1024 tokens. Batch size 8, numerical threads 1. Cache each complete embedding matrix with identities.
- Typed baseline: training mean/SD for numeric values and train-only sorted one-hot categories (unknown = all zero). This is a transparent typed baseline, not PyTorch Frame neural training. Schema reordering/renaming maps back to canonical field identities; it must not alter typed features.
- Both feature matrices receive train-only column standardization before Ridge(alpha), unpenalized intercept. Grid [0.01,0.1,1,10,100]; smallest validation MAE, first grid value on ties. No train+validation refit. Freeze selected heads before test. Record selected alpha, all validation losses, test MAE, train-mean baseline MAE, paired row losses, mean and sample SD over splits.
- Sensitivity: all 240 row embeddings for each intervention; report token counts, per-row cosine change, baseline-head test MAE and absolute prediction change. Same target held fixed. No intervention-head retuning.
- Budget: USD0 cloud. 1800 seconds aggregate measured local compute including environment preparation, fetching, runs, retries and checks; numerical processes single-threaded. Per-call timeout is bounded by remaining budget. Resource trouble or exhausted allowance stops execution with INCOMPLETE; never silently reduce rows/seeds. Reserve 300 seconds for checks. No historical training authorized.

## Task 1 — source and environment pins
Create labs/sources/l163 with source ledger and primary source bytes. Recheck matching historical release availability, retain unresolved fields. Install Transformers in the existing environment without replacing torch; freeze exact dependencies in labs/l163-requirements.txt. Pin weights outside Git; record downloaded artifact hashes.

## Task 2 — behavioral contracts, then implementation
Create labs/_check_l163.py and stubs in labs/relkit/rows_l163.py. Run .venv/bin/python labs/_check_l163.py and observe missing-behavior failures. Implement serialization (target exclusion, escaped strings, exact columns), padding-aware pooling and validation-only choice; add tests for leakage, unknown categories, train-only scaling and invalid inputs. Keep three live learner TODOs with adversarial checks.

## Task 3 — complete measured experiment
Create labs/_prepare_l163.py, _encode_l163.py, _run_l163.py and input manifests. Run under labs/_budget_l163.py. Before fitting, pin fixtures/splits/configuration and downloaded model identities. Before test, write selection receipt. Store all arrays, keyed predictions, token traces, timings and complete reports. Independently check pooling, ridge against NumPy normal equations, metrics and split/key coverage in labs/_verify_l163.py. Reject three deliberately wrong learner functions.

## Task 4 — lesson, portable notebook and figures
Create lessons/content/0163-lm-encoders-for-rows.md, assets/row-encoder.js/.css, assets/l163-lesson.js, labs/_build_l163.py and reference/row-encoders.html. Reuse retrieval, prediction, teach-back and shared typography. Existing tokenizer-viz and cell-graph-viz depict different operators; link their prior lessons rather than imply they implement BART. New diagram shows both actual paths, shapes, frozen/fitted boundaries and train/test flow. Measured token explorer changes column order/schema and updates true stored tokenization; no browser inference claim. Student implements serialization, mean pooling and validation choice. EXIT is an encoder comparison and written defense.

## Task 5 — execute and deliver
Create labs/_execute_l163.py and _delivery_l163.py. Run portable solution in an empty directory from embedded evidence; report fresh encoding separately. Check inline source parity, all widget states at desktop/mobile widths, keyboard/reset, no-JS/print, figures, links and deterministic regeneration. Integrate manifest, curriculum, Year5 plan, resources, glossary, retrieval/paper deck, notes and preparation record. Stage only intended files, then run labs/_check_pages_checkout.py and retain receipt. Inspect actual figure/browser screenshots. Do not claim learner mastery, live Colab or deployment.

## Historical full reproduction gate
Named target: Vogel/Hilprecht/Binnig arXiv2305.15321v1 Table1 wikiTables BART_table vs +GNN, three tasks, 10,000 tables, 70/20/10 split, three-run means and validation-selected checkpoints. Matching implementation, exact subset/splits/seeds, checkpoint, serialization, masking, graph/decoder interface and training/decoding settings remain unresolved. Source audit is inherited from L159/L162 and refreshed here. Historical result NOT_RUN; fidelity NOT_ESTABLISHED. The local frozen-text comparison is a separate course experiment. No fabricated historical trainer or tolerance. The standing USD10 cap covers all future reproduction costs but does not authorize guessed historical training.
