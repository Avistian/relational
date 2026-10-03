# B13 · Synthetic relational data: reproduction contract

Approved 2026-10-03: USD0 paid execution and3600 aggregate local execution seconds. All preparation, failed attempts, reruns and delivery checks count. No cloud/API model calls. Numerical commands run through `_budget_b13.py`; exhausted budget stops as INCOMPLETE. Preparation allowance recorded separately. No change to the user's USD10 maximum for future costed proposals.

## Lane A: fresh course held-out-schema experiment

This is a Tier C diagnostic, not RDB-PFN, PluRel, RT, or a published benchmark. Source `relkit/synthetic_b13.py` is visible in the notebook. Three live learner functions are called by generation, feature construction and fitting.

Seeds0/1/2 × training diversity1/3 × feature-only/relational =12fits. Each arm has12databases with4tables,64rows/table,4standard-normal numeric attributes/row:12288generated numeric feature cells. Additional budget:3072PKcells,2304FKcells,768target cells per training arm. All training topologies have3edges. Generated arrays and noise are paired by seed across arms. FK RNG streams are fixed per edge. Key identifiers are integer row indices; allFKs resolve.

One-family training: directed chain0→1→2→3. Three-family training: four chains, four out-stars0→{1,2,3}, four in-stars{0,1,2}→3. Exact directed isomorphism enumerates all24node permutations. Test: four new diamond0→{1,2}→3databases per seed, never generated in training;256query rows per arm. Full identity=(database_seed,table3,row_id). Numeric-feature cell count is held fixed, not the attention-token count of either paper. The diamond has4edges, so this is also a connectivity-degree shift. Database count stays12; varying family count is not the paper's database-count scaling axis.

For table3, target y=.5*x0+mean(direct parents' gathered x0)+Gaussian noise withSD.1. Feature-only ridge reads the query's4attributes. Relational ridge also reads one hand-engineered parent-mean feature. This feature encodes the true target structure; it deliberately makes generalization easy. No claim of learned schema reasoning, universal relational transfer, or RFM superiority follows.

Ridge alpha1 fixed before scoring; population mean/SD fit on training only; zero scales→1; intercept unpenalized. No validation tuning/checkpoint selection. NumPy float64. Mean squared error on all256queries; report three seed values and sampleSD, not a confidence interval. Same model additionally predicts with each query-table FK column independently permuted, preserving FK counts and validity. Original truth stays fixed. No refit or new fit count. Store all3072intact and3072corrupted predictions. Independent scalar FK read, scalar MSE, SVD least-squares, hidden-target intervention and corruption rejection checks.

```sh
.venv/bin/python labs/_budget_b13.py .venv/bin/python labs/_test_b13.py
.venv/bin/python labs/_budget_b13.py .venv/bin/python labs/_run_b13.py
.venv/bin/python labs/_budget_b13.py .venv/bin/python labs/_verify_b13.py
```

## Lane B: complete saved-evidence replay of RDB-PFN v5 Table9

Origin: L200's complete selected released-checkpoint evaluation on rel-f1/driver-dnf. B13 authenticates its portable packet and independently scores all30evaluations/21060predictions. No new model inference or pretraining occurs. Paper v1 is the curriculum reading; Table9 target is explicitly v5, not v1's differently numbered table.

Unchanged source commit a95378225478daa262b85f180d482da7516b0af6; dataset d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL checkpoint eaf789a9b25ee8486d6f48997ba076f850bbc30b. RDBPFN model_eval00528; single-table model_eval00360; TabICL0.1.3/v1.1 with32estimators.11,411train/566validation/702test. No validation tuning. Ten512-row support draws: first4bytes big-endianSHA256(rel-f1-dfs-2:driver-dnf:{s})→default_rng→choice without replacement. Same support for all3arms; global modelRNG42.

Support-only median imputation; all-missing→0. Support-only population normalization and±100clamp. Numeric category codes preserved. Float32 inference; query keys and labels excluded from features. Six blocks,width96,4heads,FFN192; retained source implementation and evaluator. Complete(driverId,date)query keys. AUROC targets.7219/.6640/.7176; mean/sampleSD and paired differences; descriptive mean-distance tolerance.02 is not statistical equivalence.

Independent replay uses positive-negative pair comparisons with half credit for ties, separate from L200's rank-based implementation. It reconstructs all support index draws and authenticates keys, labels, probabilities and immutable receipts. Released label orientation complements the current raw30-dayDNF indicator; do not call these probabilities raw DNF risk without complementing both probabilities and labels. Checkpoint width96 overrides appendix128. Full rawDFS regeneration and complete historical-availability authentication remain NOT_RUN/NOT_ESTABLISHED.

```sh
.venv/bin/python labs/_budget_b13.py .venv/bin/python labs/_audit_b13.py
```

**Executable fresh inference, not executed by B13:** the portable source archive preserves repository layout, full upstream code and prepared packet. From its extracted root in a separate Python3.11environment, install `labs/l166-requirements.txt` and Torch2.5.1, then:

```sh
python labs/_fetch_l166.py --out /tmp/b13-input
python labs/_run_l166.py --input /tmp/b13-input --out /tmp/b13-fresh
```

Default runs all30evaluations. Use a new output directory; no optional reduced-seed invocation establishes the complete selected target. New execution requires its own costed ledger; current approval is replay-only. The fresh path is inherited from completed L166/L200 execution, statically packaged here, not newly runtime-tested in B13.

## Lane C: PluRel v1 Table1 full-target audit

Primary target: all18tasks (10classification AUROC,8regressionR²) comparing Real-only and Synthetic+Real, three seeds each. Six leave-one-database-out folds ×2initializations ×3seeds =36real-data training runs, then108task/arm/seed test evaluations. Each fold trains on the other five databases' forecasting/autocomplete tasks. Validation selects the highest-scoring checkpoint separately for each task. Preserve full official test queries and temporal context filtering. Report mean and standard error across3seeds. The published table's rounded values are in `sources/b13/plurel-table1.json`; reported gains can differ from subtracting rounded cells.

Synthetic+Real starts from the1024synthetic-database/4Btoken model; reconstructing it from scratch is additional synthetic pretraining, not included in the36real-data runs. The paper selected this base using validation performance across scaling configurations; repeating the complete selection process expands scope beyond this fixed-base target. Synthetic-only is preserved as published context, not counted among the two approved comparison arms. Whole Figure1 scaling grid and whole-paper reproduction remain NOT_RUN.

RT:12blocks,width256,8heads,FFN1024,context1024cells,BFSwidth128; all-MiniLM-L12-v2 name/text encoder384dimensions. Column/feature/neighbor masks; omit full attention; per-head query/key RMS normalization. Masked numeric/boolean prediction with Huber/cross-entropy objectives. AdamW lr5e-4,weight decay.1,betas(.9,.999),eps1e-8; batch128; gradient clip1.0. Preserved scripts use50001steps and evaluation every2000steps, max_eval_steps80; verify exact evaluated populations before asserting full-test parity. The training code uses OneCycleLR with20%warmup and linear annealing. Do not substitute the later leaderboard model card's cosine description.

**Artifacts actually located:** official current PluRel source9ba86ee46d143fc32bc25ee9b0a5348a49e06af2 points to paper tagv1.0.0 at2a273cfd21933ee4893dfcb862a3edaed45ac665. Complete tagged archive includes generator, Rust sampler, RT model/trainer/scripts and lockfile. Hub `stanford-star/rt-plurel` revision0cac262c0fc95353372b2cf1d5c1b0a1e649a449 inventories original synthetic checkpoints under `paper/`. No checkpoint weights are downloaded in B13.

**Unresolved historical mapping:** public training scripts hard-code seed0; all three paper seed IDs and full per-seed weights/receipts are not established. The Hub card explicitly identifies continued/finetuned task checkpoints as June2026leaderboard additions; regression selection uses validationNMAE, not Table1R². Therefore their existence does not supply the historical three-seed Table1comparison. Current root `model.safetensors` is a later model; do not silently substitute it. Exact dataset/processed snapshot identity, task aliases, evaluation caps, base selection provenance and original preprocessing populations need authentication. Author-designated paper code is available; claiming all code or weights are missing would be false.

**Runnable upstream operators, preserved without modifications:** in the extracted tagged source, `pixi install`; `pixi run torchrun --standalone --nproc_per_node=1 scripts/baseline_pretrain.py`; synthetic base: `scripts/synthetic_pretrain.py`; continued: `scripts/cntd_pretrain.py`. These seed0 source entrypoints are not an automatically complete three-seed Table1operator. Set paths, authenticate data, supply verified three-seed configuration and resolve selection details before dispatch. Full archive rather than the reading-only extracted files is required for the Rust sampler.

```sh
.venv/bin/python labs/_reproduce_b13.py --audit
```

Paper reports roughly3B200hours per pretraining run; this is a published runtime, not a measured B13run or current price quote. No fresh cloud estimate or cost admission is claimed. Training status INCOMPLETE_SOURCE_PROTOCOL_AND_BUDGET_GATE; inference NOT_RUN; full-pretraining NOT_RUN. USD0approval explicitly excludes paid dispatch. No silent smaller-model/fewer-seed replacement.

## Delivery and learner boundaries

Portable notebook defaults run the course experiment and saved RDB-PFN replay; source appendices are readable model/trainer listings, not executed paper training. Successful notebook/browser/Pages checks certify delivery, not source-result parity, liveColab, deployment, historical identity, or learner mastery. Learner PENDING_WRITTEN_DEFENSE. Revisit schema/row/cell distinctions after1/7/30days.
