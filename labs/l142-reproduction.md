# Lesson 142 — full selected experiment and controlled-scope comparison

Approved 2026-09-30. Paper: Chen et al., RelGNN ICML2025, arXiv2502.06784v2 §§3–4 and Table2. Named target: rel-f1/driver-position, five-seed reported test MAE3.798. This package completes five fresh full-data reconstructed RelGNN fits and five full-data ordinary edge-attention fits. It does not establish historical training identity or reproduce the whole paper.

## Provenance

- Source commit cffdb8b54627e92c7dd112c1243dde739c90d35b; verbatim MIT sources reused under `sources/l141/`, checked against `_sources_l142.json` hashes. https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b
- Paper https://arxiv.org/html/2502.06784v2 . Motivation §§3.1–3.2; composite equations1–5; Table2 target3.798.
- Full database archive SHA256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; task archive775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e.
- Fresh L142 preprocessing:74,063 rows,169,421 forward foreign-key edges,338,842 directed edges;20atomic routes. Graph hash and runtime package versions: `evidence/l142/prepared/prepared.json`. Seeds reuse this fresh graph, never L141 weights.
- GloVe revision e5e8fec6971be8960cfaa853a77a6ddc62a265d7, `sentence-transformers/average_word_embeddings_glove.6B.300d`. Feature inference seed42; full released test-cutoff snapshot and statistics, not train-only normalization.
- Python3.11,Torch2.5.1+cu124,PyG2.6.1,pyg-lib0.4.0+pt25cu124,PyTorchFrame0.2.3,RelBench1.1.0; exact remaining pins `requirements-l117-runtime.txt`.
- Preprocessor also downloads the pinned released checkpoint as inherited provenance, but L142 never loads it into either training arm. The L141 compatibility reconstruction is not reused or scored here.

## Frozen protocol and deviations

Queries7453/499/760; target mean recorded positionOrder in (cutoff,cutoff+60days]. Future participation determines label-row inclusion. All8712 labels and identities independently reconstructed from raw result rows. This does not establish historical feature arrival-time legality or a prospective all-driver cohort.

Both arms: uniform temporal sampling128/64,bidirectional subgraphs,batch512,no loader workers,seeds0–4,10complete epochs,15batches/epoch,Adam.005,L1loss,first strict minimum validationMAE; train-target percentiles2/98=[2,31] at evaluation only. One evaluation-mode training batch initializes lazy parameters before optimizer construction. Test labels are removed from model input and accessed for final scoring only after selection. Final validation is resampled, so it can differ from the selection metric.

Composite: one released RelGNN layer,128channels,four128-channel heads,512→128projection,sum route outputs,source-specific fact updates,LayerNorm/ReLU,scalar head. Parameter count10,553,601. Visible arithmetic uses the live `edge_sum` and `route_fuse` functions without changing the released computation.

Ordinary: two synchronous directed-edge-type attention layers with the same head shape,projection,sum relation outputs and per-node LayerNorm/ReLU. Parameter count20,519,169. Row/time encoders and head have the same design. The constructor initializes the common model and replaces its graph operator; this consumes extra random numbers before lazy materialization. Same seed labels and sampling distribution do not imply identical weights, dropout or sampled batches. This is a course architecture comparator, not the paper's published GNN baseline, not an exact ablation reproduction, and not a capacity-matched causal estimate. Intermediate normalization depth also differs.

Historical optimizer/schedule/seed list/initialization/feature-type cache are incompletely released. The five fits use frozen reconstruction choices. Their descriptive ±.20 MAE band was declared before execution and is not a statistical equivalence test. No test tuning, discarded seeds or automatic retries. This F1 test split has been inspected in earlier lessons; new training runs are not a new independent test population.

## Primary evidence

| Arm | Validation mean±sampleSD | Test mean±sampleSD |
|---|---:|---:|
| Composite RelGNN |3.133519±.185000|4.267390±.087863|
| Ordinary attention |3.045583±.127300|4.418519±.305754|

Mean paired ordinary-minus-composite test gap+.151129±.278950MAE (sampleSD across5seed pairs). Composite wins3/5test pairs; ordinary has lower mean validation error. Neither split was used to revise the frozen models. Composite versus paper: OUTSIDE_TOLERANCE. Historical identity NOT_ESTABLISHED; whole paper NOT_RUN.

Ten unique fresh run IDs; all12590held-out predictions keyed to original archive queries and independently rescored.807790query occurrences audited with zero future-node violations. Every composite held-out raw output checked against the original release at identical weights/batches; every ordinary output against independent PyG attention propagation. Toleranceatol/rtol2e-4. The oracle is constructed inside an RNG-preserving context. Checkpoint SHA256s and complete histories are retained; large checkpoint files remain local under results/l142 and in the evidence volume.

First-batch nonfinite numerical-encoder gradients remain present: composite seed0 has640; ordinary seed0 has512. Full per-seed counts are in result.json. This inherited optimizer-health limitation is not fixed or hidden. Synthetic fixtures have finite gradients through encoders,time,graph and head, but do not certify real-data gradient health.

Independent float64 ordinary output/input/parameter-gradient checks cover92parameter tensors; composite checks cover46 across4ordinary/composite/empty cases with exact outputs. Mechanism matrix-power audit derives coefficients[1,2,2,1]; six semantic mutants rejected, including positional pairing and repeated-destination overwrite. These mechanism fixtures are not benchmark evidence.

## Notebook and delivery

Portable student and solution notebooks embed all model/trainer code, original differential source files, four PNG figures and hashed compressed author predictions. Three learner functions remain live. Default execution runs both neural fixtures and rescoring, not full fresh training. The explicit RUN_FULL_REPRODUCTION gate uses full data with default five seeds per arm. Its separate pinned-GPU validation uses one additional full seed100 per arm, excluded from primary means. See `evidence/l142/notebook/execution.json` and notebook-audit.json for actual validation status.

Browser checks: desktop/mobile375,keyboard/reset,swap interventions,noJS,print,portable figures,manifest discovery,deterministic build and copied Pages. Final actual report: `_delivery_l142_results.json`. LiveColab/deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. No deployment requested.

## Aggregate budget and commands

Maximum USD10 across preparation,primary fits,retries and validation. USD3 reserved for startup/build/commit/storage/unitemized overhead. T4+2physicalCPU+16GiB rateUSD.00022572/s, verified2026-09-30 at https://modal.com/pricing . Timeout reservations: preparation1800s,ten primary fits1800s each,notebook600s; total upper worker allocationUSD4.604688, plus3overhead =USD7.604688. Pilot durations52.704/73.437s pass the25percent+120s forecast against1800s. No model retry. A local artifact collector ran before the notebook worker committed its volume, received FileNotFoundError, and was rerun after successful commit; its failure log is preserved. No new cloud worker was dispatched for that collection retry. All reservations persist and duplicate phase dispatch is rejected. Body-time estimates are not itemized invoices; invoice NOT_ITEMIZED.

Run from repository root:

```bash
.venv/bin/python labs/_check_l142.py
.venv/bin/python labs/_parity_l142.py
.venv/bin/python labs/_mechanism_l142.py
.venv/bin/modal run modal/l142_repro.py --phase prepare
.venv/bin/python labs/_collect_l142.py prepare
.venv/bin/modal run modal/l142_repro.py --phase composite-0
.venv/bin/python labs/_collect_l142.py composite-0
.venv/bin/modal run modal/l142_repro.py --phase ordinary-0
.venv/bin/python labs/_collect_l142.py ordinary-0
.venv/bin/modal run modal/l142_repro.py --phase remaining
.venv/bin/python labs/_collect_l142.py full
.venv/bin/python labs/_audit_l142.py
.venv/bin/python labs/_figures_l142.py
.venv/bin/python labs/_build_l142.py
.venv/bin/python labs/_execute_l142.py
.venv/bin/modal run modal/l142_notebook_check.py
.venv/bin/python labs/_verify_l142.py
.venv/bin/python labs/_delivery_l142.py
.venv/bin/python labs/_check_pages_checkout.py
```

Existing phase identities deliberately refuse a second dispatch. For a new paid experiment allocate a new evidence volume/output identity and approved aggregate ledger. For portable execution use a fresh working directory and pinned runtime; switch the explicit gate only when the runtime/budget is available. Default notebooks do not launch cloud jobs.
