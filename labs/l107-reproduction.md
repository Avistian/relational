# Lesson 107 reproduction contract

Approved scope: GCN-GRU snapshot learning, a complete Wikipedia course comparison, and the released EvolveGCN SBM Table 2 lane. The design predates training in `docs/plans/2026-09-26-lesson-107-design.md`. The lesson is prepared material, not learner mastery.

## Replay commands

From the repository root, create Python 3.12 environment and install `labs/requirements-l107-authoring.txt`. The authoring file pins the versions actually used for the local build/notebook, including Modal and Playwright. Runtime-only replay uses the separate `labs/requirements-l107-runtime.txt`. Local authoring uses the recorded newer CPU Torch/NumPy environment; authoritative comparative results use the same pinned GPU environment for all nine fits.

```bash
python3.12 -m venv /tmp/l107-replay
/tmp/l107-replay/bin/python -m pip install -r labs/requirements-l107-authoring.txt
/tmp/l107-replay/bin/python labs/_check_l107.py
/tmp/l107-replay/bin/python labs/_source_check_l107.py
/tmp/l107-replay/bin/python labs/_audit_l107.py
/tmp/l107-replay/bin/python labs/_history_witness_l107.py
/tmp/l107-replay/bin/python labs/_run_sbm_l107.py --variant H --device cpu --output /tmp/l107-sbm-fresh
/tmp/l107-replay/bin/python labs/_run_sbm_l107.py --variant O --device cpu --output /tmp/l107-sbm-fresh
/tmp/l107-replay/bin/python labs/_run_wiki_l107.py --arm 3600 --seed 0 --output /tmp/l107-wiki-fresh
/tmp/l107-replay/bin/python labs/_run_wiki_l107.py --arm 86400 --seed 0 --output /tmp/l107-wiki-fresh
/tmp/l107-replay/bin/python labs/_run_wiki_l107.py --arm tgn --seed 0 --output /tmp/l107-wiki-fresh
```

Repeat Wikipedia commands for seeds 1 and 2. CPU execution is a separate runtime realization, not bitwise replay of GPU predictions. SBM defaults to a 10,000-second in-run cutoff; raise it explicitly for a slower machine, and inspect status instead of treating a partial result as complete. The full GPU author runs use:

```bash
.venv/bin/modal run --detach modal/l107_repro.py::full_sbm
.venv/bin/modal run --detach modal/l107_repro.py::full_wiki
```

Before building the Modal image from a fresh checkout, run `.venv/bin/python labs/_prepare_l107.py` to download/authenticate both datasets.

These are paid launches, not harmless display commands; costs are bounded in `labs/_budget_l107.json`. No automatic retry or silent partial-state resume. Existing output directories reject code/config changes. For a new attempt use a fresh directory and retain the old record. Successful full runs may be inspected without rerunning.

## Pinned inputs

- EvolveGCN publication-era commit `3f4996ac2a742a69fe6ce6e378b6317518bd99bf`. Original Apache-2.0 files and license in `labs/sources/l107/original`; individual hashes in `_sources_l107.json`. Both release configurations are retained without editing the originals.
- SBM raw CSV SHA256 `bfae7d9d89400b42c2ef513eaeef56dbbcfae2745c59f40fb3746f9cf98c8c45`, 4,870,863 rows, 1,000 nodes, times shifted to 0–49. Automatically download/authenticate if absent. Full-data adjacency/degree/normalization parity with original functions covers all 50 snapshots.
- Wikipedia raw SHA256 `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09`, all 157,474 raw records. Data from [JODIE authors](https://snap.stanford.edu/jodie/), cite Kumar, Zhang & Leskovec (KDD 2019). A fresh parser checks every field in the processed cache; `_data_check_l107_results.json` pins that cache. TGN implementation comes from L102's visible source port; it is fully included in the notebook.

## Protocol and deviations

| Dimension | SBM released replay | Wikipedia course comparison |
|---|---|---|
| Task | Predict snapshot t+1, history t−5…t | Discriminate observed event pair from supplied negative at its question time |
| Split | Source index windows: train 5…33, val 34…38, test 39…48; target one later | Released 70/15/15 timestamp cutoffs and held-node training exclusion; union of train/val/test is permitted history; excluded pre-validation events never ingested |
| Features | One-hot out-degree, dimension 162 obtained by source scan through t=48 | Same 172 raw edge features; snapshot per-node incident means plus log-count/activity; TGN consumes event messages; zero static node features |
| Access | Only current/past input graph values, but future-aware degree schema retained | Strict past; only completed snapshots; a=t assumed; event minibatches never split tied timestamps |
| Architecture | 2 evolving layers; H 50/50, O 51/51; pair head 100/565 | 2 GCN layers + node GRU (32 hidden), or 1-layer TGN attention (32 memory, 10 neighbors, no dropout) |
| Objective | Released weighted CE, mean by number of pairs; H [.1,.9], O [.15,.85] | Equal class-weight CE/BCE; each window/event batch averaged, optimizer step counts differ |
| Optimizer | Separate Adam for encoder/head; LR .01/.005 | Adam .001 for each arm; no tuning on test |
| Schedule | Max 100 epochs, validation after epoch 5, stop after >50 failures | Exactly 10 epochs, best validation pooled AP |
| Candidates | Source smart negatives for train, 4× draw cap, unique first-occurrence order; all ordered pairs for val/test | One frozen negative per event, identical across arms, seed 100+run for train, 1/2 for val/test; accidental positives possible and retained |
| RNG | Released seed 1234; single explicit NumPy stream replaces eight worker streams | Explicit model seeds 0/1/2; same GPU runtime for reported comparison |
| Selection | Strict validation MAP improvement, test evaluated at improvements; test values never select | Strict validation AP improvement; full checkpoint including TGN pending messages |
| Metrics | Snapshot AP (called MAP); mean of all positive reciprocal ranks per node, then node/snapshot means | Pooled test AP/AUROC over exactly matched positives/negative destinations |
| Recurrent state | Evolve weights restart at learned initial W for every query window | Snapshot node state persists with one-window truncated gradient; TGN source-style detached message/state boundaries |
| Runtime deviations | Dense one-hot features, modern Torch, chunked head gradient sums; original output/gradient and source-transform checks | Shared modern Torch 2.8/NumPy 2.2.6 GPU runtime; separate local CPU engineering runs not used to select results |

Released `egcn_o.py` uses a GRU-style update rather than the paper's LSTM. Original model code also leaves RReLU in training mode during evaluation; preserve this and save each test snapshot's RNG states. EvolveGCN-O has an unused top-k scorer: retain it so initialization draws stay aligned. Full all-pairs metrics include missing self-pairs as negatives; training rejects self-pair negatives. The release's degree-schema scan uses future snapshots; it is not a clean production protocol.

The optimized negative sampler preserves original draw count, source-then-destination draw ordering, exclusions, first-appearance ordering and draw-cap shortfalls; an original-source candidate/RNG fixture verifies this. Parallel historical data-loader RNG streams are not recreated. Pair-head chunking performs one accumulated gradient update per entire window and backpropagates the accumulated embedding gradient through all six slices; it is not extra optimizer steps or truncated EvolveGCN history. Floating-point reduction order differs.

## Interpretation and boundaries

Paper Table 2 targets: H MAP .1947 / MRR .0141; O MAP .1989 / MRR .0138. Predeclared CLOSE means absolute MAP difference ≤.015 and MRR difference ≤.003, compared separately. These tolerances are diagnostics, not statistical equivalence tests. One released seed supplies no independent-seed uncertainty estimate. For Wikipedia, report all three seeds and sample SD; no broad cross-dataset superiority claim.

The O history witness uses mean-slope deterministic activation to test structural dependence. It is separate from the stochastic source-score lane. A same-seed stochastic comparison alone can confound content dependence with different numbers of random draws.

Other datasets, baseline families, hyperparameter search, paper LSTM-O and paper ablations: NOT_RUN. Historical training/worker/runtime identity: NOT_ESTABLISHED. Live Colab and deployment: NOT_CHECKED until tested separately. See `_analysis_l107_results.json` for observed statuses rather than inferring them from these commands.

Cached Wikipedia arrays are authenticated field-by-field by `auth_l107.py`, even when the raw-file identity string is unchanged. The full raw reparse and uploaded cache were checked independently; this boundary validation and stricter existing-output rejection were added after author training without changing any model/trainer computation or reported prediction.

The Wikipedia node-ID universe and validation/test destination universe are fixed from the release. Snapshot states begin at zero but update for all allocated nodes on every slice, including not-yet-observed and isolated nodes. This is a known-catalog comparison, not a test of runtime entity discovery; TGN memories instead stay untouched until their observed messages arrive. No raw excluded pre-validation event features enter either arm.

Configuration interpretation: the release replaces YAML `feats_per_node: 100` with the observed degree-schema dimension 162; the YAML `k_top_grcu: 200` is unused in these implementations, whose summary k equals output width (50 for H, 51 for O).

State reconstruction differs across course systems: snapshot validation/test replays permitted prior snapshots under the selected frozen weights; TGN validation starts from the end-of-training online memory, and test restores the selected validation-end memory including pending messages. Thus trainer/state construction is another controlled-and-documented system choice, not an isolated window-width causal comparison.

## Audit completed author evidence

After both paid training calls finish, run the original-model verification on the saved checkpoints and random states. This is a separate bounded paid call per variant, already included in the budget:

```bash
.venv/bin/modal run --detach modal/l107_repro.py::replay
.venv/bin/python labs/_download_l107.py --lane wiki
.venv/bin/python labs/_download_l107.py --lane sbm
.venv/bin/python labs/_cost_l107.py
.venv/bin/python labs/_analyze_l107.py
```

`replay --variant H` or `--variant O` checks one completed variant separately; use these instead of the combined replay, not in addition. Replay uses the original sparse-feature encoder and original classifier, the selected trained weights, each saved stochastic RNG state and a different classifier chunk size. The report counts every compared probability. This establishes implementation agreement at the selected checkpoints, not historical training identity. The metric audit independently reconstructs all final scores and checks matched Wikipedia questions/candidates.

MRR tie behavior is runtime-sensitive: the original code uses NumPy's default unstable `argsort`, followed by reversal. The local newer ARM runtime produced differences up to roughly 2.2×10⁻⁷ on H despite identical saved probabilities. Authoritative metric verification therefore runs both the unedited original metric methods and independent formulas in the pinned NumPy 2.2.6 x86 environment. The removed `np.float` alias is restored to its historical meaning, `float`, for the original method. A separate local audit proves that both orderings lie inside the exact minimum/maximum MRR attainable within equal-score groups. AP agrees across these environments. Do not silently replace this source tie policy with a new metric convention.

After H missed the predeclared MAP tolerance, an additional no-copy initialization audit checked full SBM dimensions. `_initialization_l107.py` verifies all 501,440 initial H/O encoder/head parameter values and the post-construction random states against the original constructors, in the current local runtime. This rules out a constructor draw-order discrepancy in that environment; it does not establish historical RNG identity or explain the accuracy gap. The threshold and training schedule were not changed after seeing test scores.

To repeat the frozen GPU experiment without overwriting author evidence, pass the same fresh run ID to `full_sbm`, `full_wiki`, and `replay`, for example `--run-id reproduction-1`. A repeated ID still rejects existing training outputs. Download that namespace with `_download_l107.py --run-id reproduction-1 --lane sbm --output /tmp/l107-reproduction-1` (and `--lane wiki` separately). Omitting the ID refers to the original author namespace. These options were added after author training; the canonical model/trainer hashes are unchanged. Budget every new launch separately before running it.

For browser verification in a fresh authoring environment, install its Chromium build with `/tmp/l107-replay/bin/python -m playwright install chromium` and provide the required system browser libraries. The tested machine uses its existing browser-library directory recorded in `_delivery_l107.py`. Use `/tmp/l107-replay/bin/modal` in place of `.venv/bin/modal` for the fresh environment and authenticate that client to your own Modal account. Authoring dependency pins reflect the actually tested utility versions; the remote image independently pins the named experiment runtime. Neither environment is represented as the historical 2019 runtime.
