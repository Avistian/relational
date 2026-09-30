# Lesson148 — ablation discipline

Approved2026-09-30: five arms × five seeds × ten complete epochs. Named baseline: RelBench v1 arXiv2407.20060v1 Table7 RDL rel-f1/driver-position. Target validation3.193/test4.022MAE; descriptive closeness ±.2, not statistical equivalence. Four additional arms are exploratory course extensions, not published ablation results. Whole paper NOT_RUN; historical identity NOT_ESTABLISHED. Reused test population precludes a fresh confirmatory claim.

## Source and data

RelBench release1.1.0 commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Current upstream byte checks in evidence/l148/sources.json. Full visible baseline in relkit/rdl_l117.py; full L148 model interventions in relkit/ablation_model_l148.py; complete trainer in _train_l148.py. Student and solution notebooks inline these definitions. PyG/Frame numerical kernels remain pinned dependencies; no independent-kernel claim.

Fresh graph from checksum-matched F1 database ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 and task775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e. Nine tables,74063rows,7453/499/760train/validation/test queries. Frozen GloVe revisione5e8fec6971be8960cfaa853a77a6ddc62a265d7. Preprocessing uses released test-censored snapshot and seed42 type inference; statistics are not train-only. Fresh preparation audits every FK edge against a key lookup, saves graph hash and reference query arrays. Labels independently reconstructed from uncensored raw future result rows in (cutoff,cutoff+60days]. Future participation selects label population.

Pinned runtime Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame0.2.3,RelBench1.1.0,pyg-lib0.4.0+pt25cu124; labs/requirements-l117-runtime.txt pins remaining principal dependencies. Transitive package versions are not all historically recoverable. No original RNG/type-inference-state identity claim.

## Fixed training contract

Table-specific four-block128wide FrameResNet; numerical/categorical/timestamp/frozen-text semantic embeddings; relative-age encoder; two heterogeneous SAGE layers with sum neighbors and sum relations; destination root transforms; nodeLayerNorm/ReLU; scalar head. Uniform temporal sampling128/64,batch512,zero loader workers. PaperTable9 says128; released source halves second-hop fanout. All7453training queries every epoch for ten epochs, Adam.005,L1 loss, no early stopping. Five seeds0–4. First strict validation minimum, full checkpoint restored. Clip evaluation to train-target2/98percentiles; evaluate all499/760queries. Final validation is resampled and can differ from selection-time value. Owner upper bounds checked in all training/evaluation batches. Baseline selected outputs compared with original Model on every held-out batch at rtol=atol1e-5. Test never selects a checkpoint.

## Exact interventions

- full: unchanged pinned architecture and schedule, freshly trained.
- encoder: instantiate full model with same seed, retain semantic encoders, first residual block and decoder; remove remaining three blocks. Width unchanged, capacity/dropout/random-number consumption and compute differ.
- messages: empty every sampled edge tensor before forward. Keep node/root transformations, relation biases, temporal encoders and head. Inactive parameters remain counted; equal parameter count does not mean equal capacity.
- history: use baseline upper-bound temporal sampler; prune dated non-root nodes older than365days relative to their own query. Keep undated nodes and legal roots. Remap retained edge endpoints and delete incident edges before encoding. Sampling slots are not refilled; legal disconnected sampled rows can remain. No pre-sampling-window claim.
- combined: encoder + messages; only their interaction is estimable. History interactions are unmeasured.

Paired difference per seed is arm−fullMAE. Interaction is combined−encoder−messages+full. Report mean/sampleSD; neither SD nor five seeds quantifies cross-database uncertainty. Query identity is(entity,cutoff), never entity alone or row order. All predictions shuffled during independent rescoring; missing/duplicate/extra/nonfinite values rejected. Checkpoint hashes and all25run IDs preserved. Checkpoint binaries collected locally under ignored labs/results/l148; public artifacts retain hashes and executable commands.

## Gradient limitation

The initial real batch diagnostic finds512nonfinite gradient entries in each arm. The released numerical missing-value path is preserved; no silent repair or discarded run. Synthetic neural fixtures have finite gradients and exact baseline output parity. These are different evidence populations. Source-model output agreement and finite scores do not prove healthy optimization. Report effects as conditional on this implementation.

## Budget and recovery

Current rate T4 +2physicalCPU+16GiB =USD.00022572/s, https://modal.com/pricing checked2026-09-30. Initial approved reservations25×900s training,1800s preparation,1800s notebook,600s audit,1800s retry,USD3overhead =USD9.43302. First mount/root-cache failed before worker startup. Image-time cache environment populated the second mount/cache and also failed before startup. Both full1800s reservations remain charged to the conservative ledger; configure cache only at worker startup for successful third attempt. Revised all-in plan30300s×rate+USD3=USD9.839316. Pinned notebook packaging first failed locally before dispatch, then its first remote setup failed on a local-only dynamic import. That full1800s reservation is retained. A guarded600s recovery brings the conservative total to USD9.974748. No automatic retries. CapUSD10 includes all attempts/validation/overhead; credits do not increase cap. Worker-body estimates exclude startup/build/storage and are not an invoice.

Full seed0 pilot is included once in primary statistics. Complete pilot measured32.6346s;125%+120s forecast160.7933s fits900s timeout. Remaining24dispatches are reserved before launch. Additional dispatches require remaining budget; exhausted reservations are never erased. A failed or timed-out fit remains INCOMPLETE and is not silently shortened.

## Commands and fresh execution

From repository root:

```bash
.venv/bin/python labs/_check_l148.py
.venv/bin/python labs/_neural_l148.py
.venv/bin/python labs/_sources_l148.py
.venv/bin/modal run modal/l148_repro.py --phase prepare
.venv/bin/python labs/_collect_l148.py prepare
.venv/bin/modal run modal/l148_repro.py --phase audit
.venv/bin/modal run modal/l148_repro.py --phase pilot
.venv/bin/python labs/_collect_l148.py pilot
.venv/bin/modal run modal/l148_repro.py --phase remaining
.venv/bin/python labs/_collect_l148.py full
.venv/bin/python labs/_audit_l148.py
.venv/bin/python labs/_figures_l148.py
.venv/bin/python labs/_build_l148.py
.venv/bin/python labs/_execute_l148.py
.venv/bin/modal run modal/l148_notebook_check.py
.venv/bin/python labs/_verify_l148.py
.venv/bin/python labs/_delivery_l148.py
.venv/bin/python labs/_check_pages_checkout.py
```

Existing phase IDs refuse duplicates. For an independent rerun use a separate checkout/run namespace: change Modal volume/app names and evidence/results directories, initialize a new empty budget ledger, retain the same cap and pilot gate. Never delete the published reservations or overwrite previous evidence. The optional notebook lane runs all25fits on an existing pinned local CUDA environment, without invoking Modal; it is OFF by default and does not itself enforce a cloud spending cap. Use a fresh empty directory. Default notebook execution independently audits saved predictions and real sampled context; it does not retrain.

Browser/noJS/print/mobile, notebook execution, clean Git-index Pages, liveColab and deployment are distinct checks. No publication requested. Learner PENDING_WRITTEN_DEFENSE.

## Observed completion and delivery

All25primary fits completed,10epochs each, with25unique run IDs. Baseline validation3.224491±.053071/test4.193107±.242783MAE: both CLOSE under frozen±.2rule. Independent raw labels8712; keyed primary predictions31475. History paired effect validation−.294033/test+.770824; no-message validation+.399012/test−.153775MAE. Test encoder–message interaction+.055403±.285042. No retuning or dropped primary seeds. Source-gradient limitation retained.

Default25-code-cell solution PASS. Pinned26-cell validation PASS, including one extra full history seed100fit using notebook definitions;1259additional predictions independently checked and excluded from primary statistics. Full25fit notebook gate was not executed as a unit. CPU/GPU intervention probes,4deliberate fault rejections,desktop1200/mobile375,16interactive states,keyboard/reset,print/noJS,4portable figures,deterministic rebuild,25copied-Pages links and clean Git-index Pages PASS. LiveColab/deployment NOT_CHECKED.

Final conservative resource reservations plus overheadUSD9.974748; known worker-body estimateUSD0.166615. The latter excludes setup failures/startup/build/storage; invoiceNOT_ITEMIZED. Failed setup app was explicitly stopped. No remaining paid work.
