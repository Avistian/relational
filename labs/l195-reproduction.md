# L195 — complete thesis stress-test replay

Approved scope: all five L149 basic-GNN and five engineered-feature LightGBM runs on validation/test; all thirty L182 RDB-PFN comparison runs via L190's authenticated packet; complete L194 twenty-one-task report. USD 0 cloud/API, 1800 aggregate local execution seconds including failed attempts, numerical preparation, builds and verification. No source repair, new training/inference, paid dispatch or deployment.

## Named experiments and frozen protocol

L149: rel-f1/driver-position, 499 validation / 760 test queries, seeds 0–4. Basic GraphSAGE, ten full 7,453-query epochs; source 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. Original user-study SQL+LightGBM, ten validation trials per seed and selected refit; source 445bb7a3b1230f49f8e5890ae81754d3e365680f. Complete entity/cutoff keys; independent SQLite scoring uses common archive float64 targets. Original GNN training cast targets to float32; replay retains stored prediction precision and checks original scores with tolerance 1e-6. Independent common-target metric checks use 1e-12. These are complete pipeline comparisons, not isolated architecture effects. Full original contracts are in packet/contracts/l149.md.

L149 global query-weighted advantage = FE absolute error minus GNN absolute error. Average five run differences per query, then 2,000 whole-driver bootstrap draws, original seed 137; percentile 2.5/97.5 interval. This reverses L149's loss-penalty sign. Preserve both splits; no test slice selection. Conditional on fixed models/split; shared race/time dependence and new-database uncertainty not covered. Test reused: exploratory, not pristine confirmation. Run indices do not imply shared training randomness.

L182: RDB-PFN v5 Table 9 rel-f1/driver-dnf, RDBPFN/RDBPFN_single/TabICLv1.1 × support seeds 0–9 × all 702 test queries with 512 support rows. Exact released key/label/support identities and original NPZ/receipts from L190 retained. Original source/data/checkpoint identities are retained in packet/l182/original-input-manifest.json, checked against both original receipt digests; the full L182 contract is also preserved. Checkpoint weight files are not copied or loaded. Rank AUROC independently checked by pair-counting, half credit for ties, tolerance 1e-12. No model loading or raw relational reconstruction. Support variability is not cross-database uncertainty; repeated inference on the same data is not an independent replication.

L194: authenticate entire packet and original report; independently parse RDBLearn v1 Tables 1–3 against all 21 task identities, fixed AutoGluon+DFS comparator. Reference signs 17/3/1 are descriptive. All fresh scores stay null; 567 validation / 63 test slots remain unexecuted. Source 0.1.2 commit b5b03ebf8091547285a6e06cba53d2d1a40cb171 / FastDFS 0.2.1; full inherited source, split, support, backend/depth/seed contract in packet/contracts/l194.md. Complete report does not establish complete model reproduction: INCOMPLETE_SOURCE_PREPROCESSING_GATE. Data/availability/checkpoint/regression-normalization/executor gaps remain. No downstream executor is invented by this replay.

## Provenance and exact commands

`packet/origins.json` maps every copied input to original workspace path, SHA256 and whether an earlier manifest independently pinned it. L149 GNN predictions and receipts are checked against the original training manifest; FE prediction hashes are checked against original run receipts. Some FE receipt/report files lack an earlier available manifest; these are explicitly current-snapshot-pinned, not retroactively authenticated historical bytes. L182 files are chained through L190; L194 report through its artifact manifest and packet through its input manifest. Hashes certify bytes, not historical truth.

From repository root, using the existing environment (NumPy; author tools nbformat/nbclient/nbconvert/Matplotlib/BeautifulSoup/Playwright):

```bash
python3 labs/_budget_l195.py .venv/bin/python labs/_replay_l195.py
python3 labs/_budget_l195.py .venv/bin/python labs/_test_l195.py
python3 labs/_budget_l195.py .venv/bin/python labs/_verify_l195.py
python3 labs/_budget_l195.py .venv/bin/python labs/_build_l195.py
python3 labs/_budget_l195.py .venv/bin/python labs/_execute_l195.py
python3 labs/_budget_l195.py .venv/bin/python labs/_delivery_l195.py
```

`_prepare_l195.py` is the author snapshot step requiring original lesson inputs; it refuses changed existing packet bytes. The portable notebook embeds that whole packet and uses only NumPy plus Python standard library. `_budget_l195.py` kills child processes at the remaining aggregate cutoff and accounts failures. Historical completed author budget reservations are not authorization for new paid runs.

## Evidence boundary and delivery

33,650 stored prediction rows rescored; COMPLETE_SELECTED_REPLAY. No fresh model evaluation, pretraining, raw-label/DFS reconstruction, whole-paper parity, new-database test, causal architecture/prior isolation, economic undervaluation or learner mastery. Interactive margin sweep is retrospective sensitivity; WITHIN_MARGIN is interval geometry, not a formal equivalence test. All stronger claims remain NOT_ESTABLISHED; fresh training NOT_RUN; learner PENDING_WRITTEN_DEFENSE. LiveColab and deployment NOT_CHECKED.

Checks write `_verify_l195_results.json`, `_execution_l195_results.json`, `_delivery_l195_results.json`, `_checkout_l195_results.json`. Local isolated-index Pages checks do not deploy. The budget file includes failed attempts. A future decisive experiment requires new held-out data, admitted sources, valid temporal availability and a predeclared practical margin and feasible budget; it remains NOT_RUN.
