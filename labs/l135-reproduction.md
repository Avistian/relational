# Lesson 135 — frozen tuning and selected RelBench reproduction

## Claims and protocol

The complete selected published lane is RelBench arXiv2407.20060v1 Table7, basic RDL on **rel-f1/driver-position**. Five fresh default fits, seeds0–4, ten full epochs, all7,453 training queries per epoch,499 validation queries and760 test queries. Published validation/test MAE3.193/4.022 (reported SD.024/.119). Predeclared descriptive tolerance: absolute mean difference≤.2 MAE on each split. CLOSE is not statistical equivalence.

The course extension searches six configurations: lr.001/.005/.01 × fanouts[32,16]/[128,64], seeds100,101, ten full epochs each. Rank the mean selected validation MAE over both seeds; choose first minimum epoch within each fit and stable configuration ID on exact cross-configuration ties. Search has no test loader, requests no test table, and has a task guard rejecting test access. `_freeze_l135.py` requires all12 fits and records their hashes before final dispatch. Final fits use fresh seeds0–4 for default and frozen winner, with validation-selected checkpoints. Test results never enter selection. If winner equals default, the same condition is reused and explicitly identified.

The official test split was evaluated in earlier course lessons. Fresh fits do not create a new untouched holdout; the test lock protects selection within this declared study.

This is an exhaustive **six-point declared grid**, not an exhaustive search over all hyperparameters, not a reproduction of a paper tuning procedure (the paper used defaults), and not a cross-database comparison. Full paper/all-task reproduction: NOT_RUN. Exact historical training identity: NOT_ESTABLISHED.

| Element | Frozen implementation |
|---|---|
| Upstream | RelBench commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; MIT |
| Model | Visible `relkit/rdl_l117.py`: per-table Frame encoders, relative-time encoders, two128-channel typed GraphSAGE layers; sum neighbor and relation aggregation; seed head |
| Default | Adam.005, fanouts[128,64], batch512, mean L1 loss, ten epochs |
| Sampling | Uniform incoming temporal neighborhoods; inclusive node timestamp≤root cutoff at every hop; disjoint query ownership |
| Features | Released database up to test cutoff; feature statistics are not train-only; frozen GloVe sentence-transformer revision |
| Output | Training-label2nd/98th percentile clipping for validation/test; first minimum validation checkpoint |
| Runtime | Python3.11, Torch2.5.1+cu124, PyG2.6.1, Frame0.2.3, RelBench1.1.0, pyg_lib0.4.0+pt25cu124; `requirements-l117-runtime.txt` |
| Archive identity | DatabaseSHA256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482; task775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Source distinction | Paper table lists128 neighbors; source divides by hop to use[128,64]. Released preprocessing and source behavior preserved, not silently corrected |

The new trainer exposes only learning rate and fanout, gates test access and adds batch audits. It preserves model dimensions, initialization, optimizer type, loss, epoch count, clamping and checkpoint selection. Original-model parity uses the same selected weights and sampled batches. This validates forward implementation; it does not prove the unavailable historical training run used identical RNG/library/hardware settings.

All batch audits check timestamps against original global rows, root cutoffs, edge identities and query ownership. They do not establish unrecorded ingestion times or historical feature availability. SQL joins independently validate all13 foreign-key relations; archived query labels/IDs/times are checked for every saved artifact.

## Commands

From the relational root, the executed author sequence was:

```bash
.venv/bin/python labs/_check_l135.py
.venv/bin/python labs/_verify_l135.py
.venv/bin/python labs/_prepare_l135.py
.venv/bin/modal run --detach modal/l135_repro.py --mode pilot
.venv/bin/python labs/_collect_l135.py --mode pilot
.venv/bin/python labs/_pilot_check_l135.py
.venv/bin/modal run --detach modal/l135_repro.py --mode search
.venv/bin/python labs/_collect_l135.py --mode search
.venv/bin/python labs/_freeze_l135.py
.venv/bin/modal run --detach modal/l135_repro.py --mode final
.venv/bin/python labs/_collect_l135.py --mode final
.venv/bin/python labs/_analyze_l135.py
.venv/bin/python labs/_audit_l135.py
.venv/bin/python labs/_figures_l135.py
.venv/bin/python labs/_build_l135.py
.venv/bin/python labs/_execute_l135.py
.venv/bin/modal run --detach modal/l135_notebook_check.py
.venv/bin/python labs/_collect_notebook_l135.py
.venv/bin/python labs/_build_l135.py
.venv/bin/python labs/_delivery_l135.py
```

Collection follows completion, not merely dispatch. Existing budgets, frozen decisions and output directories prevent accidental overwrite/relaunch. For a new independently budgeted run, use `_new_run_l135.py` below to create a fresh isolated experiment with a unique Modal volume. Do not delete prior evidence to make an old command rerun.

```bash
.venv/bin/python labs/_new_run_l135.py --directory /tmp/l135-repeat-001
cd /tmp/l135-repeat-001
# Use the original workspace's .venv/bin/python and .venv/bin/modal as executables.
# The fresh copy contains the protocol, runners and newly pinned budget.
```

Local pinned GPU alternatives for a single fit (no automatic dollar enforcement):

```bash
python labs/_run_l135.py --config lr005-full --seed 999 --epochs 1 --output /tmp/l135-smoke
python labs/_run_l135.py --config lr005-full --seed 100 --epochs 10 --output /tmp/l135-search-fit
python labs/_run_l135.py --config lr005-full --seed 0 --epochs 10 --test --output /tmp/l135-paper-fit
```

A single fit is not the full five-seed target or the complete tuning study. The portable notebook's `RUN_FULL_REPRODUCTION = True` runs the complete search, freeze and final comparison with visible code, in a compatible GPU runtime. Default notebook execution replays author evidence; it is not fresh training. Exact Colab frontend behavior: NOT_CHECKED.

## Budget

USD10 aggregate cap; USD3 reserved for overhead. Current checked Modal resource rate: T4.000164/s +2 physical CPU cores*.0000131/s +16GiB*.00000222/s =USD.00022572/s. Per-fit timeout900s, retries0. Pilot+12 search+10 final maximum commitmentsUSD4.672404. A separate900s notebook validation reservation addsUSD.203148. Runtime estimates do not equal invoices; billing total remains NOT_ITEMIZED. A failed attempt retains its maximum reservation. Additional launches are rejected when reserved costs+overhead exceedUSD10. Never use credits to enlarge this allowance.

Sources: https://modal.com/pricing and the timed pilot in `_pilot_l135_results.json`. The pilot deliberately overestimates a full fit as fifteen times its one-epoch worker lifetime. Primary workers include cold preprocessing and checks; training-only timing is also saved. Equal epoch budgets do not establish equal compute across fanouts.

## Evidence and interpretation

`evidence/l135/frozen.json` records the complete search and decision. `summary.json` records all seed metrics, paired tuned-minus-default differences, conditional95% t interval, independently rescored prediction counts, batch audits, exact artifact hashes and measured worker-resource estimates. `predictions.npz` contains labels, predictions, entities and query cutoffs; selected weights are retained separately in `results/l135` and the Modal volume. Weight hashes are in each result.

The interval conditions on this task, fixed split and selected winner. It excludes variation over datasets, time splits and repeated search decisions. Published seed SDs and our sample seed SDs are descriptive, not evidence of equivalence. A new seed cannot undo test reuse. Deployment, live Colab and learner mastery are separate from successful code execution.
