# L151 — classification portfolio and complete selected reproduction

Approved: rel-trial/study-outcome, RelBench v1 Table6 RDL. Source9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; licensed files and URL/SHA256 manifest in sources/l151. Target validation68.18±.49/test68.60±1.01 AUROC percentage points. Frozen descriptive tolerance±1point for reference mean; no equivalence or historical identity claim.

## Frozen protocol

Fresh materialization seed42 and GloVe300 revision e5e8fec6971be8960cfaa853a77a6ddc62a265d7. Full database archive9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a; task archive20eb922c1a8f894563f4b4c900c912e396688d2bd71eeb6b13f19429aa74a649. No teaching caps. Raw labels independently rebuilt and source SQL reconciled for11994train/960val/825test queries.

Study starts by cutoff; qualifying Primary outcome analysis in(cutoff,cutoff+365days], p in[0,1], modifier not'>'. Positive iff minimum numeric p≤.05; no qualifying analysis means no query. Source treatment of other modifiers preserved. Retrospective completion cohort and inferred dates retained. Timestamped neighbors checked against each owner's query cutoff; untimestamped feature-arrival legality NOT_ESTABLISHED. Released snapshot feature-statistics scope preserved.

Two128-channel mean-neighbor GraphSAGE layers, summed relation outputs, batch norm/ReLU; fanouts64/32 uniform temporal sampling; batch512, BCE logits, Adam. Full20epochs,24batches/11994queries each. Released2001batch cap never reached. First strict maximum validation AUROC chooses checkpoint. Final validation/test use newly sampled neighborhoods, so final validation may differ from selection validation.

Reference: lr.0001, seeds0..4. Separate course search:.00005/.0001/.0002, seed100; train/val task tables only, no test loader. Max checkpoint-selection validation, lower-rate exact tie break; immutable candidate-result hashes frozen before selected seeds10..14. Search and notebook seed1000 excluded from primary means. Past course test exposure makes search exploratory. No post-test tuning. Distinct reference/selected seed sets prevent interpreting their difference as a paired causal learning-rate effect.

Runtime: Python3.11, torch2.5.1/cu124, PyG2.6.1, pytorch-frame.2.3, relbench1.1.0, numpy1.26.4, pandas2.2.3, sentence-transformers3.3.1, pyg-lib.4.0+pt25cu124; requirements-l117-runtime.txt. Historical RNG/environment identity NOT_ESTABLISHED.

## Reproduce and audit

From repository root:

```bash
.venv/bin/python labs/_check_l151.py
.venv/bin/modal run --detach modal/l151_repro.py --phase prepare
.venv/bin/python labs/_collect_l151.py prepare
.venv/bin/modal run --detach modal/l151_repro.py --phase ref-0
.venv/bin/python labs/_collect_l151.py ref-0
.venv/bin/modal run --detach modal/l151_repro.py --phase reference
.venv/bin/modal run --detach modal/l151_repro.py --phase search
.venv/bin/python labs/_collect_l151.py reference
.venv/bin/python labs/_collect_l151.py search
.venv/bin/python labs/_freeze_l151.py
.venv/bin/modal run --detach modal/l151_repro.py --phase selected
.venv/bin/python labs/_collect_l151.py selected
.venv/bin/modal run --detach modal/l151_audit.py
.venv/bin/python labs/_verify_l151.py
.venv/bin/python labs/_figures_l151.py
.venv/bin/python labs/_build_l151.py
.venv/bin/python labs/_execute_l151.py
.venv/bin/modal run --detach modal/l151_notebook_check.py
.venv/bin/python labs/_delivery_l151.py
```

Author reservations/phase markers reject redispatch. These commands document the completed sequence, not a safe way to overwrite it. For an independent run use the standalone notebook full gate in a new output directory, or create a new explicitly budgeted author volume/ledger. Do not delete reservations to retry. Standalone RUN_FULL_REPRODUCTION=True runs13freshfits; PREPARED_ROOT=None freshly materializes. Default gate off audits author evidence plus a synthetic full neural forward/backward, not benchmark training. Full execution requires substantial host RAM (author64GiB); live Colab NOT_CHECKED. Notebook itself has no billing guard.

## Cost and evidence boundaries

USD10 aggregate cap includes preparation, all seeds, retries, notebook checks and overhead. Initial fourteen900s fits plus1800s preparation/audit allocation at.00033228/sec totalsUSD4.784832;USD3overhead =>USD7.784832. Separate600s gradient audit uses contingency USD.199368. Pilot must satisfy1.25*seconds+120<900 before remaining training dispatch. No automatic retries. The first independent gradient-audit dispatch failed during remote module import (missing l151_repro). Its USD.199368 reservation and failure log are retained; a separately reserved packaging fix repeats no training. All attempts retained; resource estimates exclude unitemized startup/build/storage/commit costs and are not invoices. Current ledger:_budget_l151.json.

Source parity is not gradient health. Report all first-batch nonfinite-gradient counts and the separate real-checkpoint gradient differential. Historical identity and feature-arrival legality NOT_ESTABLISHED. Whole paper NOT_RUN; fresh manual-FE and LightGBM comparators NOT_RUN. Published LightGBM70.09±1.41 is context only. Author execution leaves learner PENDING_WRITTEN_DEFENSE. Publication/deployment not requested; live Colab and deployment NOT_CHECKED.

## Measured result and delivery

Five reference fits: validation67.696425±.482609/test69.258775±.748950 AUROC points (mean±sampleSD). Test is CLOSE to the published68.60 under the frozen±1point rule. Three candidates selected lr.0001 from checkpoint-selection validation scores68.045336/68.288815/68.072588 for rates.00005/.0001/.0002. Five selected refits scored test68.457012±1.350223. The chosen recipe equals reference; do not call their unpaired seed difference a tuning gain or pool the tracks.

All20,730primary/search predictions independently aligned and rescored;13unique runs,20full epochs each;3,388,770query occurrences with zero future-timestamp violations. Fresh graph5,434,924rows matches the earlier current-release graph hash exactly, not a historical paper graph claim. All13initial gradient checks report256nonfinite entries. Independent source comparison at seed0selected weights matches813gradient tensors, including256nonfinite entries; maximum finite-entry error8.38e-9.

Default standalone26-code-cell notebook PASS; four portable figures,16browser interventions at1200/375widths, keyboard/reset,noJS,print,29copiedPages links and deterministic regeneration PASS. The first pinned notebook app launch used a stale volume name and stopped before worker dispatch; corrected without repeating training. Exact final pinned-runtime/cost/clean-checkout status is recorded in _notebook_l151_results.json, evidence/l151/reproduction.json and _checkout_l151_results.json. LiveColab/deployment remain NOT_CHECKED.

The first source audit packaging failure is retained under evidence/l151/gradient-packaging-failure.*, the pre-dispatch notebook volume failure under notebook-volume-preflight-failure.*. No model training was retried. The portfolio-entry.json export records author evidence ownership and leaves the written defense empty.

Final reconciliation: pinned26-code-cell notebook PASS with one additional full seed1000,1785independently keyed/scored predictions and checkpoint hash verified; excluded from primary means. Conservative reservations plus overhead USD8.183568; recorded worker-body estimate USD0.966480 excludes unitemized startup/build/commit/storage and the failed pre-body audit. Invoice NOT_ITEMIZED. Complete Git-index Pages build PASS; no push/deployment.
