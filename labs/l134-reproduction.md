# Lesson134 reproduction contract

## Two declared experiments

1. RelBench arXiv2407.20060v1 Table7, rel-f1/driver-position, basic RDL. Five fresh seeds0–4; all released data, ten complete epochs. Targets validation3.193/test4.022 MAE; reported SDs.024/.119. Descriptive mean tolerance0.2MAE predeclared. CLOSE is not statistical equivalence.
2. Full-topology rel-stack user-engagement systems workload, simplified width32 GNN, constant+log-age features, two sum-SAGE heterogeneous layers, node LayerNorm/ReLU, BCE/Adam. Original task SQL reconstructed on the pinned database at the latest released training cutoff; independently checked event sets. Roots must exist at the cutoff; sort by entity ID and select first2048 without inspecting targets. B64/[8,4], B64/[16,8], B128/[16,8]. Two warmup update batches followed by one complete measured pass over the prefix. No full-task epoch, benchmark score or accuracy ranking claim.

## F1 protocol and evidence

Primary source https://arxiv.org/html/2407.20060v1. Released code commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Canonical visible model/trainer `relkit/rdl_l117.py`; original released module files and licenses in `sources/l117`. Fresh L134 operator `_run_l134.py` instruments all loader yields; its learner audit is `relkit/scale_l134.py`.

Full F1 database74063rows/338842directed edges, official task7453train/499val/760test. DatabaseSHA256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; task775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e. GloVe model revisione5e8fec6971be8960cfaa853a77a6ddc62a265d7. Separate table Frame ResNets four blocks/128channels, relative-time encoders, two heterogeneous sum-SAGE layers and seed head. Adam.005, batch512, source fanout[128,64], uniform temporal sampling, ten full epochs, first strict validation minimum, training-label2nd/98thpercentile clipping. All6295final predictions independently rescored and replayed through original model. All403895training/evaluation query occurrences checked.

Runtime Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame0.2.3,RelBench1.1.0,pyg_lib0.4.0+pt25cu124; `requirements-l117-runtime.txt` pins remaining packages. Source fidelity is distinct from historical runtime identity. Source uses [128,64] versus paper table128; original seeds and exact historical packages unknown. Type inference seed42 before training seeds; preprocessing statistics use full released database through test cap. Future-participation conditioning in released task, unavailable ingestion times and mutable feature histories remain explicit. Original numerical-encoder gradient issues documented in L131 are preserved; this lesson does not claim healthy whole-model gradients.

## Scale data and failures

Official endpoint `https://relbench.stanford.edu/download/rel-stack/db.zip` served SHA2565a97bf65a926529143e96f6413b2b0550ca55c01247f85156bfb999ee903e94e, different from historical registry9e5acfcaef041059dba346b1a876ff108fbb496ede0955dc89be6349e777a380. Attempt1 correctly stopped at hash verification. Read-only archive inspection records bytes/members; subsequent course-scale work explicitly pins those bytes. No historical archive equivalence claim. The user-engagement task archive matches its registry98141d35e6471e6a4d391461c1ccf2bfb3aa363975e1c11e992223c1b2a82b2d.

Attempt2 stopped when test-cap filtering left references to future parent rows. Recovery applies original `Dataset.validate_and_correct_db` to projected key/time tables, then independently counts every relation with SQL. It nulls96 references (64votes,5comments,27posts); it never clips IDs to existing rows. Attempt3 stopped at a future-node audit before training. Timestamp units were investigated independently; archive/task nanosecond storage and explicit UNIX-second conversion are recorded. All failed source snapshots and cost packets are retained. Attempt4 verified that some archived task roots postdate their cutoffs on the pinned database. Attempt5 reconstructs the original task SQL on that same database and independently checks the full one-cutoff cohort against event sets, then selects2048root-eligiblequeries. This is an explicit course-scale cohort, not historical task-row identity. Attempt5 found fewer than2048eligible roots at the earliest training cutoff and stopped. Attempt6 uses the latest released training cutoff, checks all labels mature before the validation boundary, and freezes the same three configurations. No result-based tuning. Final outcome is recorded in `evidence/l134/scale/scale.json` and `_scale_audit_l134_results.json`.

Only key/time columns are read from the full archive; all text materialization is avoided in this explicitly simplified scale model. Host graph, CSC construction, preprocessing and GPU tensors are distinct memory costs. Core timings sum sampler+transfer+forward/backward/optimizer, with CUDA synchronization. Audit time is excluded from the core rate and included in a second reported rate; loader construction/preprocessing/warmups and profiling-bookkeeping overhead are not in either rate. Configuration order and fixed2048query cohort are fixed, not representative random timing trials. No latency confidence interval, maximum-throughput or full rich-model memory claim.

## Commands

From the repository root:

```bash
.venv/bin/python labs/_check_l134.py
.venv/bin/python labs/_verify_l134.py
.venv/bin/python labs/_correction_check_l134.py
.venv/bin/python labs/_time_units_l134.py
.venv/bin/python labs/_prepare_l134.py
.venv/bin/modal run --detach modal/l134_repro.py --mode pilot
.venv/bin/python labs/_collect_l134.py --mode pilot
.venv/bin/python labs/_pilot_check_l134.py
.venv/bin/modal run --detach modal/l134_repro.py --mode paper
.venv/bin/python labs/_collect_l134.py --mode paper
.venv/bin/python labs/_analyze_l134.py
.venv/bin/python labs/_audit_l134.py
# Scale: a fresh reserved attempt name; completed attempts cannot be overwritten.
.venv/bin/modal run --detach modal/l134_scale.py --attempt attempt-6
.venv/bin/python labs/_collect_scale_l134.py --attempt attempt-6
.venv/bin/python labs/_analyze_scale_l134.py
.venv/bin/python labs/_figures_l134.py
.venv/bin/python labs/_build_l134.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l134.py
.venv/bin/modal run --detach modal/l134_notebook_check.py
.venv/bin/python labs/_collect_notebook_l134.py
.venv/bin/python labs/_delivery_l134.py
```

These completed-run commands deliberately refuse duplicate reservations. To repeat, create fresh evidence volumes and a new budget ledger; preserve prior artifacts. The notebook's optional full-training gate is portable visible code and OFF by default; its manual switch has no monetary guard.

## Budget and evidence boundaries

USD10 aggregate maximum: USD2 F1+validation, USD5 scale, USD3 overhead/recovery. Prices checked2026-09-27 at https://modal.com/pricing: T4.000164/s, CPU.0000131/core/s, RAM.00000222/GiB/s. F1 worker2CPU/16GiB900s, at most8slots =USD1.625184 maximum. Scale2CPU/32GiB =USD.00026124/s, each attempt reserved before dispatch against USD5. No automatic retries; scale has a persistent started marker. Failed reservations are never refunded by this conservative accounting. CPU diagnostic worker maxima and build/storage overhead fit the USD3reserve; actual billing NOT_ITEMIZED. No credits are used to raise the cap.

Whole-paper and historical identity NOT_ESTABLISHED. Full rel-stack paper benchmark NOT_RUN. Default notebook replay is not fresh training. Live Colab and deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE. Full verification details are separate machine-readable artifacts.


## Final scale outcome

Completed all three configurations on 4,247,264 nodes and 11,625,774 directed edges. Original task SQL and independent event-set reconstruction agree for 83,531 queries at 2020-07-02; 19 roots created later are excluded before selecting the first 2,048 eligible IDs. Labels cover 91 days and mature by 2020-10-01. The first 2,048 archived task rows had 2 root-cutoff violations on this archive; their labels were never used for the final workload.

Mean sampled occurrences per batch 3066.5/4913.3/9687.3; core throughput 2198.2/2148.9/4076.4 queries/s for B64[8,4],B64[16,8],B128[16,8]. Peak allocated 25.0/30.56/42.22 MiB; peak host RSS 5639.8 MiB. Preprocessing 17.01 s. All 6,144measured query occurrences pass ownership/time/source-edge checks. Warmup updates and per-batch loss/timing/memory/query IDs are retained. These timings are descriptive, single-pass, fixed-order measurements.

Final verification: all eight named evidence groups PASS (`_final_l134_results.json`). Separate portable GPU validation executed all22cells and five complete fresh fits; its validation/test means3.194139/4.144983MAE remain separate from primary results. All6295validation-run predictions independently rescored. Desktop/mobile54interactive states, keyboard/reset/noJS/print, deterministic builders and43copied-Pageslinks PASS. Total recorded worker estimate USD0.145069; conservative allocation ceilingUSD9.625184. No liveColab/deployment or clean-committed-checkout claim.
