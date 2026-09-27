# Lesson 136 — leaderboard evaluation replay and selected training reproduction

## Executed claims

1. **COMPLETE_EVALUATION_REPLAY:** all nine regression tasks for Kapso entry393, GNN entry380 and RT-PluRel fine-tuned entry397. 2,480,739 predictions, 27 task-entry pairs. Independent keyed alignment agrees exactly with pinned upstream join functions. Every NMAE agrees with the submitted number within1e-10. Reconstructed aggregates: Kapso0.24761040628893738; GNN0.3129864901946843; RT-PluRel0.2813920374009386. `labs/evidence/l136/leaderboard.json` contains every task score, row count, CSV hash and official task-file hash.
2. **COMPLETE_SELECTED_TRAINING:** historical RelBench arXiv2407.20060v1 Table7 basic RDL on rel-f1/driver-position, five fresh seeds0–4, ten epochs, all7453 train queries per epoch,499 validation and760 test queries. Validation3.187926990±.025788405/test4.097695229±.099604882 MAE (sample SD). Paper means3.193/4.022; inherited L135 descriptive absolute-mean tolerance.2 → CLOSE/CLOSE. Not statistical equivalence. All6794 saved pilot/final predictions independently rescored, checkpoint hashes verified, original-model maximum output difference2.86102294921875e-6. SQL verifies all13 forward foreign-key relations and corresponding reverse edges; all74063 released database rows accounted for.
3. **Portable default notebook:** three live audit functions align and rescore all2280 F1 submitted predictions; reconstruct27 task scores from the labeled complete author audit; independently rescore6794 primary training predictions. Default execution does not train or directly replay every task.
4. **Portable full-evaluation gate:** all22 code cells executed in an isolated directory with a copied, pinned78MB official task cache and source archives. Independently rescored all2,480,739 predictions; exact primary audit reproduced. Full-training switch stayed off. See `_notebook_replay_l136_results.json`.
5. **Portable full-training gate:** isolated pinned GPU execution and separately collected evidence are recorded in `_notebook_gpu_l136_results.json`. Five validation fits are not pooled into the primary experiment. Live Colab is a different frontend and remains NOT_CHECKED.

## Pinning and score contract

- Leaderboard/source commit: `584a03d518b2b655580ea8e1cfbbb26bec0a2841` (MIT).
- Official hosted task revision: `d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600`, dataset `stanford-star/relbench-v1`.
- Attached prediction archives: GitHub submission issues393,380,397, URL+SHA256 in `_sources_l136.json`. Do not substitute a current training script for the unidentified generating run.
- Exact canonical regression task list comes from the pinned `LEADERBOARD_TASKS` constant. Entity+cutoff keys must form a bijection with test queries. Finite values required. NMAE uses the official hosted train-target sample-SD constant. Aggregate is the arithmetic mean of exactly nine task NMAEs, not ranks or a pooled-row mean.
- Independent scorer uses Python `math.fsum`; source oracle executes pinned key coercion, validation, alignment and `make_nmae` functions. This is not a claim that the full current submission CLI, its packaging workflow, or an external submission was executed.
- The current leaderboard page's extracted text and repository data disagreed on entry availability; the captured repository bytes define this lesson's snapshot. The entry ranking is not a timeless claim.

Machine-readable setup comparison: `evidence/l136/config-diff.json`. Consolidated contract: `_protocol_l136.json`; dispatch-time source fingerprints remain in the original budget ledger.

## Source discrepancies and unknowns

- `rel-stack/post-votes` normalization: hosted.5104313497537508 versus recomputed pinned raw-train sample SD.5104212106119147. Preserve the official constant for exact board replay. Cause NOT_ESTABLISHED; do not silently fit another denominator.
- Prediction validation proves identity/metric/coverage. It does not certify training-time feature legality, selection, equal budgets, or every row's actual information availability.
- Kapso's released evaluation-protocol document distinguishes frozen and rolling database regimes. The document is author evidence, not independent confirmation that a specific attached CSV came from a particular described run.
- Exact Kapso search/training reproduction: NOT_ESTABLISHED/NOT_RUN. The project supplies a campaign harness, but this lesson does not re-run its agent search or prove the historical best programs' identity.
- RT-PluRel training/pretraining reproduction: NOT_RUN. Fresh historical RDL fits are not the current GNN entry's historical training identity.
- Complete selected training is not whole-paper reproduction. All other paper tasks and whole-paper training remain NOT_RUN.

## Historical training protocol

Upstream released code commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; visible model in `relkit/rdl_l117.py` and trainer in `relkit/tuning_train_l135.py`, unchanged from their audited source paths. New run uses a unique L136 evidence volume, timestamps and ledger.

Per-table Frame encoders, relative-time encodings, two128-channel typed GraphSAGE layers with sum neighbor/relation aggregation, seed head; Adam.005, fanouts[128,64], batch512, mean L1 loss,10epochs. First minimum validation checkpoint; train-label2nd/98th-percentile clipping at evaluation. Sampling is stochastic, so checkpoint replay validation MAE may differ slightly from the selection trace; preserve both.

Released full database up to test cutoff supplies feature statistics. They are not train-only. Historical node sampling uses inclusive node-time≤root cutoff, documented separately from today's public rule excluding information at the prediction time. This is a historical reproduction, not a certified current submission. Real ingestion timestamps are not supplied.

Database archiveSHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; task archiveSHA256 `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e`. Text embedding revision in `sources/l117/text_model.json`. Runtime Python3.11, Torch2.5.1+cu124, PyG2.6.1, Frame0.2.3, RelBench1.1.0, pyg_lib0.4.0+pt25cu124; remaining pins in `requirements-l117-runtime.txt`. The isolated notebook runner additionally installs IPython8.31.0 for display support.

## Exact commands

From the relational repository root. Existing author budgets refuse repeated dispatch/overwrite; use the fresh-directory operator below for a new experiment.

```bash
.venv/bin/python labs/_acquire_l136.py
.venv/bin/python labs/_check_l136.py
.venv/bin/python labs/_replay_l136.py
.venv/bin/python labs/_prepare_l136.py
.venv/bin/modal run --detach modal/l136_repro.py --mode pilot
.venv/bin/python labs/_collect_l136.py --mode pilot
.venv/bin/python labs/_pilot_check_l136.py
.venv/bin/modal run --detach modal/l136_repro.py --mode final
.venv/bin/python labs/_collect_l136.py --mode final
.venv/bin/python labs/_analyze_l136.py
.venv/bin/python labs/_audit_l136.py
.venv/bin/python labs/_figures_l136.py
.venv/bin/python labs/_build_l136.py
.venv/bin/python labs/_execute_l136.py
.venv/bin/python labs/_execute_replay_l136.py
.venv/bin/modal run --detach modal/l136_notebook_check.py
.venv/bin/python labs/_collect_notebook_l136.py
.venv/bin/python labs/_build_l136.py
.venv/bin/python labs/_verify_l136.py
.venv/bin/python labs/_delivery_l136.py
```

Wait for each remote phase to complete before collection. Full-evaluation replay is CPU-only and downloads task tables, not whole databases. All published archive scores are checked, not just the aggregate. The native standalone notebook includes the complete operator under `RUN_FULL_LEADERBOARD_REPLAY=True`; that lane was executed separately using a copied pinned cache. `RUN_FULL_REPRODUCTION=True` separately runs all five full historical fits using the visible code. Neither notebook switch alone enforces a dollar limit.

Create an independently budgeted historical rerun without altering primary evidence:

```bash
.venv/bin/python labs/_new_run_l136.py --directory /tmp/l136-repeat-001
cd /tmp/l136-repeat-001
# Call the original workspace's .venv/bin/modal / .venv/bin/python binaries.
# Run pilot, collect, pilot check, final, collect, analyze, audit as above.
```

This creates source-pinned operators and an empty budget ledger; it launches no paid work. A new experiment requires its own budget authorization. Single-fit local GPU command (not the complete five-seed experiment):

```bash
python labs/_run_l136.py --seed 0 --epochs 10 --output /tmp/l136-seed0
```

## Budget and failures

Authorized aggregate capUSD10. Current rates checked2026-09-27: T4.000164/s +2physical CPU cores*.0000131/s +16GiB*.00000222/s =USD.00022572/s. Each worker timeout900s, automatic retries0. Dispatch reserves worst-case worker cost; requires reservations+USD3overhead≤USD10. Five-fit timing projection passed before final dispatch. Prices: https://modal.com/pricing . Available credits do not enlarge the cap.

Primary pilot+five fits reserveUSD1.218888. First paid portable check failed before training because the minimal GPU image lacked IPython; its entireUSD.203148 reservation is retained, with failure artifact `evidence/l136/notebook-validation-failed.json`. The corrected image adds IPython; itsUSD.203148 reservation is separate. Maximum worker commitmentsUSD1.625184 +USD3overhead =USD4.625184, below cap. An earlier local dispatch failed to import a misnamed budget helper before worker launch; no worker reservation or training occurred. All failures are retained rather than treated as successful executions.

Primary measured worker estimateUSD.065589094. Successful portable-check estimate is recorded in `_notebook_gpu_l136_results.json`. Failed worker usage and startup/storage costs are not itemized; the conservative reservations and overhead remain the budget bound, not a billing invoice. Billing total NOT_ITEMIZED.

## Delivery and learner boundary

Source narrative, HTML, reference, three live student TODOs, executed solution, six portable figures, visible full model/trainer, audit operators, task/CSV/source hashes and navigation ship together. `_delivery_l136_results.json` records actual browser, notebook, deterministic rebuild and copied-Pages checks. Clean Git-index Pages check recorded separately. Live Colab/deployment NOT_CHECKED; no external leaderboard submission requested or performed. Learner mastery PENDING_WRITTEN_DEFENSE.
