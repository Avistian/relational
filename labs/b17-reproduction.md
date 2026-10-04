# B17 reproduction contract

Approved 2026-10-04. Frozen design: docs/plans/2026-10-04-b17-design.md. Primary versions: FlexTab 2606.30336v2 and GTAlign 2607.11374v1. Dated HTTP bodies and SHA256 hashes: sources/b17/manifest.json.

## B17-REPRESENTATION-BOUNDARY

Complete course dependency experiment; no optimization or accuracy claim. Seeds0/1/2 initialize independent encoders/decoders. CPU float64, one thread; 2 encoder blocks, 2 decoder blocks, width16, single attention head, FF width32, GELU, pre-LayerNorm eps1e-5, residual branches, no dropout. Numeric affine tokenization plus learned column identity. [ROW] queries feature keys but never becomes a feature key. Row attention keys/values contain only first4 support rows. Collect/project/sum every encoder layer's [ROW] output, then final projection. Decoder embeds binary support labels and query MASK, injects each row embedding through single-key cross-attention, then target-stream support-only attention, FF and binary softmax head. Same weights used across tasks. Implemented from the paper's mechanism description, not authenticated upstream source.

Fixed support features: [[-1,0],[0,1],[1,0],[0,-1]]. Query: [[.5,.5],[-.5,-.5]]. IDs s0,s1,s2,s3,q0,q1. TaskA labels[0,0,1,1], taskB[0,1,0,1]. Externally held query labels[1,0] are reporting annotations only; no metric is calculated from them. No dataset splitting, fitting, HPO, best-seed selection or training schedule is involved.

For each seed/task: baseline; complement all support labels; add2 to s0 feature0; replace q1 features by[7,-3]; support permutation[2,0,3,1]; query permutation[1,0]. This yields36 full forward records,3456 encoder coordinates and72 query probabilities. Reorder outputs by immutable IDs before deltas. Four possible external query-label assignments per seed/task yield24 exclusion checks. Conservative whole-table caching hashes features, ordered IDs, support size, preprocessing identity, dtype/shapes, encoder state and implementation version. Label-only edits reuse encoder cache, other feature/order edits invalidate it. Check separate state/preprocessing invalidations. Cached and recomputed outputs must agree to1e-12. Independent NumPy full forward tolerance1e-10. All arrays and model state saved before analysis.

Scope: these are numerical dependency checks of random networks. Sensitivity is not learned usefulness. The feature representation depends on its support population; it is target-agnostic at inference, not target-independent across pretraining. Changing the actual target column changes the feature set and may invalidate reuse. Query batching invariance is limited to the declared support-only attention mask. Temporal safety also requires preprocessing and encoder context to exclude information unavailable at each cutoff.

## B17-FLEXTAB-MULTI-TABLE5-F1-DNF

Named published target: FlexTab-Multi, v2 Table5, rel-f1/driver-dnf, AUROC0.746 (74.6%, rounded in the paper). Select the entire published test population and original temporal task split; align full(driverId,date) keys. AUROC comparison is undefined until protocol identity is recovered. Do not select a new tolerance from observed outputs.

Reported settings to retain: encoder context8192, decoder context16384, BFS breadth16; typically2 hops (not an authenticated exact task configuration). Auxiliary rows precede the root timestamp. Paper describes12-layer,768-wide encoder with12 heads and FF3072. Stage1 jointly pretrains encoder and classification/regression decoder; later task decoders train with the encoder frozen. Proprietary approximately300k-table corpus prevents full pretraining replication from available evidence. Published auxiliary-row filtering alone does not prove per-cutoff safety of shared table encodings.

Unresolved prerequisites:

1. Author release and original FlexTab-Multi checkpoint bytes, tokenizer assets, environment and configuration.
2. Exact dataset version, feature processing, task population and original support/query membership.
3. Sampling/repetition seeds, bagging/refit recipe, padding/truncation, relational traversal direction and per-hop policy.
4. End-to-end temporal policy for shared encoder context and preprocessing, not only BFS row filters.
5. Original evaluator, run receipts, historical model identity and uncertainty/repetition protocol.

The paper-linked repository returned404 in the archived API probe; HF FlexTab model search returned[]. This is a dated availability check, not proof that no privately shared model exists. GTAlign's paper promises a release; name-search results include unrelated projects and do not authenticate the graph model. No replacement repository is assumed.

Status: INCOMPLETE_SOURCE_PROTOCOL_GATE. Selected paper inference NOT_RUN. There is no complete benchmark launcher hidden behind the gate: missing author assets/protocol must be recovered and audited before implementation of that lane can be completed. No paid dispatch is possible through this gate. Full FlexTab pretraining, other11 relational tasks, other benchmarks and all GTAlign training/benchmarks NOT_RUN. No paper-result or released-source parity claim.

## Commands and budget

From repository root:

```
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_test_b17.py
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_run_b17.py
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_verify_b17.py
.venv/bin/python labs/_reproduce_b17.py --phase audit
.venv/bin/python labs/_reproduce_b17.py --phase paper
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_build_b17.py
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_execute_b17.py
.venv/bin/python labs/_budget_b17.py .venv/bin/python labs/_delivery_b17.py
```

The paper command must exit nonzero. Source hashes authenticate archived bytes, not the truth of claims or availability of missing weights. Artifact seals similarly establish package integrity, not paper parity.

Approved paid spendUSD0. Standing ceilingUSD10 total; no cloud use in this scope. Local numerical/preparation/verification cutoff3600 aggregate wall seconds, with failed attempts included in evidence/b17/local-budget.json. Notebook learner reruns are separate executions. LiveColab NOT_CHECKED; deployment NOT_REQUESTED; learner PENDING_WRITTEN_DEFENSE.
