# L140 — Two-task RDL checkpoint reproduction contract

Approved 2026-09-30: five fresh full released-protocol fits each for RelBench v1 Table6 `rel-amazon/user-churn` and `rel-trial/study-outcome`. Paper AUROC targets in percent: Amazon validation70.45±.06/test70.42±.05; trial validation68.18±.49/test68.60±1.01. Predeclared descriptive mean tolerance±1percentage point. This is not a statistical equivalence test.

Source9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Same pinned runtime as L138/L139, `requirements-l117-runtime.txt` plus Torch2.5.1/CUDA12.4 and pyg-lib0.4.0+pt25cu124. Two128-channel relation GraphSAGE layers; Frame row encoders; GloVe300-dimensional mean-word features revisione5e8fec6971be8960cfaa853a77a6ddc62a265d7; relative time; sum across relations; LayerNorm/ReLU; linear root head; BCE loss. Seeds0–4. First strict validation AUROC maximum selects the checkpoint. Re-evaluation samples fresh neighborhoods, so selection and re-evaluated validation scores can differ.

| Task | Neighbor aggregation | Fanouts | Adam LR | Epochs | Train queries / epoch |
|---|---|---|---|---|---|
| Amazon | sum |128/64|.005|10|1,024,512 (released2,001-batch cap)|
| Trial | mean |64/32|.0001|20|11,994 (complete training table)|

Both use batch512, uniform temporal sampling, full released database/task populations and all source features. Cached graph preprocessing is reused from L138/L139, with archive hashes, provenance records and current full graph checksums verified. Full fresh materialization is provided separately. The first recorded full-file Amazon graph digest is from this L140 preflight; L138 supplies the materialization source, provenance and row census, not an older digest to compare. Amazon reused graph36,176,795,562bytes; trial17,883,804,206bytes. Every run starts new weights, has a new UUID and directory, and saves checkpoint hashes, histories, predictions and audits.

## Reproduction boundaries

Amazon's checksum-matched training archive contains4,708,383queries versus paper4,732,555:24,172 fewer. Held-out counts match. This gap is retained beside numerical results, not repaired by a close score. Source, label and population audits from L138/L139 are explicitly inherited and checksum-linked; L140 independently verifies fresh predictions and current data/graph identity. Trial counts match; historical training seeds/RNG/environment identity remain unknown for both tasks.

Preprocessing statistics use the released full test-cutoff snapshot. Temporal sampling admits timestamps≤the owning query cutoff at every hop. Trial analysis timestamps are inferred from completion, and association timestamps from starts. Untimestamped metadata, mutable attributes and actual arrival histories prevent a prospective-availability claim. No ingestion-history reconstruction is possible from these archives. Test sets have appeared in prior lessons; no new pristine holdout or causal benefit is claimed. Other28RelBench tasks, LightGBM baselines and whole-paper training are not run here. Learner status stays PENDING_WRITTEN_DEFENSE.

## Exact commands

From the repository root, using the existing author environment:

```bash
.venv/bin/python labs/_check_l140.py
.venv/bin/modal run --detach modal/l140_repro.py --phase preflight
.venv/bin/python labs/_collect_l140.py preflight
.venv/bin/modal run --detach modal/l140_repro.py --phase pilots
.venv/bin/python labs/_collect_l140.py pilots
.venv/bin/modal run --detach modal/l140_repro.py --phase remaining
.venv/bin/python labs/_collect_l140.py full
.venv/bin/python labs/_verify_l140.py
.venv/bin/python labs/_figures_l140.py
.venv/bin/python labs/_build_l140.py
.venv/bin/python labs/_execute_l140.py
.venv/bin/python labs/_delivery_l140.py
```

Collect only after committed worker completion. Existing phases refuse repeat dispatch; do not erase reservations or run markers. For a new authorized experiment, use a new output volume and budget ledger. First full seed0 per task doubles as timing pilot and counts toward the five; remaining seeds1–4 require pilot seconds×1.25+120 below the task timeout. No automatic retries. Failed attempts consume their reservation.

For a new local pinned environment:

```bash
uv venv /tmp/l140-runtime --python 3.11
uv pip install --python /tmp/l140-runtime/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /tmp/l140-runtime/bin/python -r labs/requirements-l117-runtime.txt
uv pip install --python /tmp/l140-runtime/bin/python pyg_lib==0.4.0+pt25cu124 --find-links https://data.pyg.org/whl/torch-2.5.1+cu124.html
/tmp/l140-runtime/bin/python labs/_full_l140.py --task amazon --seed 0 --prepared-root /path/new-amazon --materialize --output /path/new-run/amazon/seed-0
/tmp/l140-runtime/bin/python labs/_full_l140.py --task trial --seed 0 --prepared-root /path/new-trial --materialize --output /path/new-run/trial/seed-0
```

Repeat seeds1–4 without `--materialize` against the verified directories. The fresh materializer refuses an existing destination. Full Amazon preparation needs about128GiB host RAM; cached training uses64GiB. Free Colab capacity is not assumed. The standalone notebook contains the same model/materializer/trainer, portable evidence, and `RUN_FULL_REPRODUCTION=False` by default. Its full gate validates exact runtime versions; enabling it does not enforce billing. Default notebook execution is artifact re-scoring plus a synthetic full neural forward/backward check, not full training.

## Budget and evidence

USD10 aggregate; USD7 worker reservations and USD3 overhead reserve. Current Modal rates checked2026-09-30: T4.000164/s, CPU.0000131/core/s, RAM.00000222/GiB/s. T4+2CPU+64GiB costs.00033228/s. Amazon five×2300s and trial five×600s reserveUSD4.81806; CPU preflight2cores/8GiB/600s reservesUSD.026376. Notebook checks and any explicitly bounded retry reserve before dispatch and must fit the remainder. Worker-body measurements exclude startup and volume commits; invoice is NOT_ITEMIZED. Credits do not enlarge the cap.

`evidence/l140/training.json`: all ten run identities, checkpoint hashes, independently reconciled metrics and exact verdicts. `_verify_l140_results.json`: source AST parity, unchanged epoch loop, independent sklearn/rank arithmetic, schedule/identity/audit checks and budget ceiling. `_execution_l140_results.json`: standalone notebook execution. `_delivery_l140_results.json`: actual browser/copy checks. Pinned-runtime and gated full-notebook checks are recorded separately. Live Colab and deployment remain NOT_CHECKED. No publication requested.

## Portable gate validation

The additional validation operator executes all default notebook checks and then one complete `trial` fit (seed100) through the notebook's live full-training gate:

```bash
.venv/bin/modal run --detach modal/l140_notebook_check.py --task trial --attempt 1
.venv/bin/python labs/_collect_notebook_l140.py
```

Its predictions are excluded from the predeclared primary five-seed summaries. The Amazon notebook training branch is exposed and AST-checked against the author trainer; an additional full Amazon notebook fit is not part of the required validation. Both aggregation modes execute small forward/backward fixtures. Full notebook validation reuses verified materialization, so it does not claim fresh text preprocessing. Actual gate execution status is in `_notebook_trial_l140_results.json`.


## Measured primary results

All ten primary runs completed. Independently reconciled3,817,310held-out predictions; all selected checkpoints passed original-model comparisons on512real query roots (rtol1e-5/atol1e-6). Five fresh runs per task:

| Task | Validation AUROC % ± sample SD | Test AUROC % ± sample SD | Numerical verdict | Paper/release evidence |
|---|---:|---:|---|---|
| Amazon user-churn |70.4547 ± .0724|70.3356 ± .1347|CLOSE / CLOSE|GAPPED:24,172 fewer training queries|
| Trial study-outcome |67.9720 ± .4558|68.6378 ± .7115|CLOSE / CLOSE|No known released population-count gap|

This is complete execution of the two selected released protocols, with descriptive score agreement. It is not whole-paper reproduction or historical identity. Source-level epoch loop and six model/graph primitives pass AST comparisons. Five deliberately faulty report implementations are rejected; the tolerance comparison includes an eight-machine-epsilon guard solely for an inclusive decimal boundary.

Notebook validation packaging history: the first image definition was rejected before dispatch; the next reserved attempt failed during remote helper import and its app was stopped. That entire reservation is retained. The corrected operator imports the reservation helper only in its local entrypoint. These failures did not change the primary trainer or notebook code.

A reused materialization must include `prepared.json` with its graph digest, or the output base must contain the checked L140 `preflight.json` (as the author operator does for the older Amazon cache). Fresh `--materialize` creates the required metadata automatically. Archive endpoint HEAD checks are recorded separately from content-hash verification.

Final recorded worker-body estimate: USD2.813687; it excludes unitemized startup/commit and failed bootstrap overhead. Conservative reservations including the failed notebook startup and USD3 overhead reserve: USD8.243172 of USD10. No primary training retry. A successful notebook run emitted an ignored IPython exit-hook cwd warning after saving evidence; wrapper cleanup was corrected and checked locally.
