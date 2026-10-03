# Lesson 143 — RelGNN selected-experiment reproduction

Approved 2026-09-30. Named target: Chen et al. RelGNN, ICML2025, arXiv2502.06784v2 §5.2/Table2, rel-f1/driver-position test MAE3.798. Two separate lanes: released-checkpoint compatibility replay and five full fresh reconstructed training fits. No whole-paper or historical-identity claim.

## Sources and data

Architecture/evaluation release cffdb8b54627e92c7dd112c1243dde739c90d35b, verbatim files in sources/l141 checked against _sources_l143.json. https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b
Paper https://arxiv.org/html/2502.06784v2 . Released checkpoint revision321e6f6e7af5d7546b637f147783fc28ab5d4a7a, SHA2563ba2b6e5c99bc0939d13debb6b06d8d0f0828148361662d54bab83f6c23943df.
F1 database SHA256ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; task775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e.
GloVe revisione5e8fec6971be8960cfaa853a77a6ddc62a265d7. Fresh full released test-cutoff snapshot, seed42 type inference; no inherited neural weights in fresh fits. Snapshot statistics are released preprocessing, not train-only numerical statistics. Graph:74063rows,169421forwardFKedges,338842directededges,20routes. Exact materialization hash and package versions in evidence/l143/prepared/prepared.json.
Runtime Python3.11,Torch2.5.1+cu124,PyG2.6.1,PyTorchFrame0.2.3,RelBench1.1.0,pyg-lib0.4.0+pt25cu124; remaining pins requirements-l117-runtime.txt.

## Frozen training protocol

Complete7453/499/760queries. Mean positionOrder in (cutoff,cutoff+60days]; independent raw-row reconstruction checks labels and eligible populations. Future participation determines label presence, not a prospective all-driver population. Model: one composite layer,128channels,four128-channel heads,512→128 projection,sum routes,per-node LayerNorm/ReLU,scalar head. Uniform temporal128/64,owner-specificcutoffs,bidirectional subgraphs,batch512,no workers. Ten complete epochs,15batches/epoch,seeds0–4,Adam.005,unclippedL1. Lazy parameters initialized with one evaluation-mode training batch before optimizer. First strict validation minimum saves checkpoint. Train-target2/98percentiles[2,31]clip at evaluation only. Final validation is resampled and may differ from selection-time validation. Test is scored only after checkpoint selection; no post-test retuning. Reused test population from earlier lessons is not a fresh confirmatory sample.

Five runs and architecture match the selected evidence target; unresolved training details are frozen reconstruction choices. No historical training recipe is claimed. Descriptive closeness abs(mean−3.798)≤.20, with1e-12rounding allowance; not an equivalence test. Mean and sample SD across exactly five fits, never pooled with replay or notebook seed100.

## Compatibility and gradient limitations

Original fresh qualifying.position categorical layout fails to load released weights. Preserve checkpoint-incompatibility.txt. Reconstruct only qualifying.position as numerical; compare ordered[number,position]means/populationSD with checkpoint buffers at1e-6relative/absolute tolerance. Keep fresh-training graph unchanged. Equal moments support this mapping but cannot prove uniqueness or recover the historical stypes cache. Compatible replay results carry CHECKPOINT_COMPATIBILITY_REPLAY.

Every primary held-out raw prediction is compared with the original architecture at identical weights and batches, atol/rtol2e-4. CPU operator differential checks also cover outputs and active input/parameter gradients, including empty edges. A separate real first-batch audit matches dropout RNG and compares original gradients; report nonfinite masks/counts and finite-entry errors. Source parity is not gradient health. The source numerical missing-value path is preserved, not repaired in this reproduction. Temporal event filtering is not recorded historical availability.

## Evidence and learner work

Independent labels/keys: evidence/l143/label-audit.json. Full training aggregate: evidence/l143/training.json. Actual scientific verification: _verify_l143_results.json. Source manifest: _sources_l143.json. Exact per-run predictions, full epoch histories, immutable run IDs, sampled traces and checkpoint hashes are retained. Selected checkpoint binaries are collected in ignored labs/results/l143, while the public package includes their hashes and executable training code.

Three live notebook exercises: verify_numeric_layout guards real compatibility materialization; first_validation_min controls checkpoint saving; evidence_verdict consumes audited complete run records. Canonical code is in relkit/reproduction_l143.py; model and trainer are fully visible inline. Default standalone execution runs neural fixtures, semantic checks and all author-prediction scoring; full fresh training is NOT_RUN in that lane. RUN_FULL_REPRODUCTION explicitly runs full fresh preprocessing, compatible replay and five fits in a pinned GPU environment. Separate pinned notebook validation executes one full seed100 plus compatible replay, excluded from primary statistics; its predictions are independently rescored. Learner PENDING_WRITTEN_DEFENSE. Actual browser/notebook status is in delivery/execution reports; liveColab/deployment NOT_CHECKED.

## Aggregate cost and stop rule

USD10cap covers all preparation,training,replays,failed attempts,validation and overhead. Current T4+2physicalCPU+16GiB rateUSD.00022572/s (https://modal.com/pricing checked2026-09-30). USD3reserved for build/startup/commit/storage and unitemized overhead. Eight1800second reservations (prepare,fivefits,replay,notebook) totalUSD3.250368, plus a600second real-gradient auditUSD.135432: planned conservative totalUSD6.3858. Pilot seed0 duration×1.25+120must fit1800before four remaining dispatches. Every attempt reserves before dispatch; duplicates refused and no automatic retries. Body-time estimates exclude overhead and are not an itemized invoice. Ledger:_budget_l143.json. Stop on aggregate-cap or forecast failure; report NOT_RUN/INCOMPLETE without shrinking the claimed task.

## Commands

From repository root, in a fresh experiment namespace for new paid execution (existing phase IDs refuse duplicates):

```bash
.venv/bin/python labs/_check_l143.py
.venv/bin/python labs/_parity_l143.py
.venv/bin/modal run modal/l143_repro.py --phase prepare
.venv/bin/python labs/_collect_l143.py prepare
.venv/bin/modal run modal/l143_repro.py --phase seed-0
.venv/bin/python labs/_collect_l143.py seed-0
.venv/bin/modal run modal/l143_repro.py --phase replay-compatible
.venv/bin/modal run modal/l143_audit.py
.venv/bin/python labs/_collect_audit_l143.py
.venv/bin/modal run modal/l143_repro.py --phase remaining
.venv/bin/python labs/_collect_l143.py full
.venv/bin/python labs/_audit_l143.py
.venv/bin/python labs/_figures_l143.py
.venv/bin/python labs/_build_l143.py
.venv/bin/python labs/_execute_l143.py
.venv/bin/modal run modal/l143_notebook_check.py
.venv/bin/python labs/_collect_notebook_l143.py
.venv/bin/python labs/_verify_l143.py
.venv/bin/python labs/_delivery_l143.py
.venv/bin/python labs/_check_pages_checkout.py
```

All verification reports describe completed checks, never intended checks. No publication/deployment was requested.

## Observed execution

Five primary fits completed all ten epochs. Validation MAE3.130064±.088643; test4.221853±.173512(sampleSD). Test comparison OUTSIDE_TOLERANCE. Checkpoint-compatible replay: validation2.836661/test3.737381. All8712labels and7554primary held-out predictions independently checked;403895primary training/evaluation query occurrences audited without future violations. Real diagnostic batch:640matching nonfinitegradiententries,maximum finite-gradient error1.1921e-7. These counts are batch-specific; other seeds can differ.

Default25-code-cell notebook PASS. Pinned25-cell notebook PASS, reusing this lesson's fresh prepared graph, running full seed100 plus compatibility replay. All2518additional predictions independently checked, excluded from primary aggregates. Conservative reservations plus overheadUSD6.3858; measured worker-body estimateUSD.076719 excludes startup/build/commit/storage, invoiceNOT_ITEMIZED. No paid retry. Initial browser-launch preview needed the established local library path; the delivery harness already supplies it. Initial deterministic packaging check caught dropped notebook execution metadata; builder now preserves matching executed-cell and notebook metadata. Neither issue changed scientific code or results.
