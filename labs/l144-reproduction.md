# L144 ContextGNN reproduction protocol

Approved scope: complete selected `rel-trial/site-sponsor-run` experiment, ContextGNN and ShallowItem; USD10 aggregate including all preparation, attempts, retries and validation. Paper target: ContextGNN 28.02% and ShallowItem 10.66% test MAP@10 (Table2, arXiv2411.19513v1). Whole-paper results are outside this selected-task attempt.

## Source and runtime

- Authors commit `ca4a96985b7ef73c36a40da32e70710ad9b59a1e`; exact vendored files and SHA256 in `_sources_l144.json`; MIT license retained.
- Torch2.5.1 CUDA12.4; PyG2.6.1; pyg-lib0.4.0+pt25cu124; PyTorch Frame0.2.3; RelBench1.1.0; numpy1.26.4; pandas2.2.3; Optuna4.0.0; sentence-transformers3.3.1. Full list: `requirements-l117-runtime.txt`, plus launcher pins.
- Frozen GloVe model revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`. Full archive downloads use RelBench's registry; materialization records every database/task parquet hash and graph hash. Historical archive identity is not established by a matching current registry.
- Fresh full row/text feature preparation, source-inferred column types, seed42. No sampled table population or feature removal.

## Released protocol versus reconstruction

| Component | Runnable contract and boundary |
|---|---|
| Population | 669310 train, 37003 validation, 27428 test query rows in downloaded release |
| Labels | Distinct sponsor IDs joined through study IDs from facility-study events in (t,t+365days]; released dangling-entity filtering |
| Graph | Primary/foreign-key graph, reversed relations; owner-specific temporal cutoffs |
| Sampling | Four layers on rel-trial, fanout128/64/32/16, last strategy, bidirectional disjoint temporal batches |
| Encoder | Per-table ResNet; column encoders; root identity; relative-time encodings |
| Model | Visible source-adapted full ContextGNN and ShallowRHSGNN classes; local overwrite delegates to learner function |
| Objective | Full-catalog sparse cross-entropy over positive pairs; no sampled-softmax shortcut |
| Search | 50 Optuna trials per architecture; median pruner; same released search space |
| Selection | First strict validation MAP improvement; stop if validation falls >.001 below best; test only at validation improvements, never used for selection |
| Schedule | Adam, exponential scheduler, up to20epochs; preserve source `steps >2000` (=2001batches) cap |
| Repeats | Five fits of selected configuration per arm |
| RNG deviation | Source seeds once42 but does not seed Optuna; reconstruction seeds Optuna42, trials42+i, repeats100..104; historical trajectories unrecoverable |
| Metric | Sigmoid then topk10, as source; reject repeated IDs; independent AP calculation after full query-key alignment |
| Cache limitation | Source RHS evaluation cache persists across epochs; preserved and disclosed. Clearing only for a differential forward probe does not alter stored validation/test scores |
| Gradient health | Record nonfinite gradients from first real training batch; matching source does not establish healthy optimization |
| Reporting | Pilot timing, partial validation and fixture outputs never count as final paper fits |

Search space: encoder channels32/64/128/256/512; encoder layers2/4/8; GNN channels32/64/128/256/512; item dimensions32/64/128/256/512; layer/batch normalization; fusion/feature/lookup item embeddings; batch256/512/1024; log-uniform LR[.001,.01] and decay[.8,1]. The paper lists a narrower tuning set. The predeclared descriptive comparison tolerance is ±0.02 MAP (two percentage points) per arm; report means and sample standard deviations across the five completed repeats. This is a reporting convention, not a statistical equivalence test.

Full historical hyperparameter selections and original Optuna history are unavailable here.

The source paper describes injecting shallow item vectors into the graph. The release does not do that; the contextual score also includes root-item dot product and separate branch offsets. This package implements the pinned release rather than silently substituting the paper's simpler formula.

## Cost gate and commands

Current base resource estimate: T4 .000164 + two physical cores .0000262 +16GiB .00003552 = **.00022572 USD/sec**. The materialized graph is about12.1GB, so later training workers reserve32GiB host memory at **.00026124 USD/sec**; preparation retains its original16GiB allocation. USD2 is reserved for startup/build/storage and other overhead; credits do not enlarge the cap. Each pilot/prepare allocation reserves its entire timeout before launch. `_budget_l144.json` is the aggregate guard; actual worker-body estimates are not an itemized invoice.

```bash
# From repository root. Both cloud launchers reserve budget before dispatch.
.venv/bin/modal run --detach modal/l144_repro.py::main --phase prepare
.venv/bin/modal run --detach modal/l144_repro.py::main --phase pilot-contextgnn
.venv/bin/modal run --detach modal/l144_repro.py::main --phase pilot-shallowrhsgnn
.venv/bin/python labs/_collect_l144.py
# Only if the measured decision is PROCEED; otherwise this refuses dispatch:
.venv/bin/modal run --detach modal/l144_repro.py::full
```

The timing pilots use the full materialized graph with 16 training batches and four validation batches per architecture. They are partial cost probes; they are not complete epochs, full validation, or a fair model comparison. Pilot configuration:128hidden/encoder channels,4encoder layers,64item dimensions,layernorm,fusion,batch256,Adam.001,decay1. Full search does not inherit the pilot's truncation.

For an isolated GPU/Colab environment, install torch2.5.1 and the listed pins plus matching pyg-lib, open the self-contained solution notebook, and set `RUN_FULL_REPRODUCTION=True` only after reviewing the measured cost and runtime decision. Both models and the full trainer are inline. That manual gate launches 50+5 fits per architecture; it is not a cheap notebook demonstration. The managed Modal path additionally enforces the approved aggregate budget and three-hour timeout per search arm.

## Verification and evidence

- `_check_l144.py`: live fusion, owner-cutoff, query-key MAP contracts; initial TODO stubs failed, completed implementation passes.
- `_mechanism_l144_results.json`: exact original fusion output/gradient comparison, complete small neural fixture, and independently demonstrated stale RHS evaluation cache.
- `evidence/l144/`: original failures, per-attempt timing/cost, preparation hashes, independent label audit, saved pilot rankings and cost decision as collected.
- `_execution_l144_results.json`: isolated local notebook execution; `_delivery_l144_results.json`: browser, print, keyboard, gallery and copied-Pages evidence.
- The CPU fixture converts timestamps explicitly for pandas3 compatibility. The pinned GPU benchmark uses pandas2.2.3; this compatibility adapter is fixture-only.

Final statuses are generated from the collected evidence in `evidence/l144/summary.json` and `cost-decision.json`. Full selected reproduction remains INCOMPLETE until all50+5 fits per arm are complete. Historical identity NOT_ESTABLISHED; whole paper NOT_RUN; live Colab and deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. No test-based tuning or post-result tolerance change is allowed.

## Measured decision (2026-09-30)

**STOP — full selected-task reproduction INCOMPLETE.** All733741 labels were rebuilt both by source SQL and by an independent interval/join algorithm. Fresh graph:5434924rows,53241sponsor candidates,12.132GB serialized. Each of two successful pilots trained16batches (4096queries) and evaluated the first1024validation queries. All2048 saved rankings were shuffled by key and independently rescored; test was not evaluated.

ContextGNN:7.306s training/3.129s partial validation; ShallowItem:6.940s/2.924s. Extrapolating this pilot configuration gives approximatelyUSD28.75 for110fits even at one train/validation epoch each, orUSD575.03 for20epochs each. Different configurations, pruning and changing sampled neighborhoods can alter throughput; this is a scenario estimate, not a mathematical lower bound. It excludes test passes and additional setup. No50-trial search or repeat fits were dispatched. Full code remains available; a larger paid run requires a new budget authorization.

Both first training batches contained1408 nonfinite numerical-encoder gradient entries. They are recorded separately from finite output parity. Real sampled forward comparison with original source passed numerical tolerance; the ContextGNN maximum absolute difference was6.676e-6. The synthetic score-operator output and gradient comparison was exact. No source cache or gradient repair was folded into these pilots. A transient Modal heartbeat/final-log warning occurred; completed artifacts and cost records were collected successfully.

## Final delivery and accounting

Local and pinned-runtime notebooks execute the identical25-code-cell source successfully. The expensive full search gate remains off; the pinned notebook check is not a live Colab session or a completed benchmark. Desktop1200/mobile375,20ranking intervention states,keyboard/reset,noJS,print,four portable figures,38copiedPages links and both galleries pass. Deterministic regeneration and inline AST agreement pass. Conservative allocation reservations plus overhead:USD3.503504. Recorded worker-body estimate:USD0.327385; invoiceNOT_ITEMIZED. Clean Git-index Pages verification is recorded separately. No push/deployment requested.

The cost decision was made with USD3.346760 reserved including overhead. The later pinned notebook check reserved another USD0.156744, producing the final USD3.503504 ceiling. No benchmark fit was added after the STOP decision.
