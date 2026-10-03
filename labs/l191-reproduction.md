# L191 RelBench Published-Table Reconstruction

Approved scope: USD 0 new cloud/API spending;1,800 aggregate local execution seconds including preparation, failures, validation, builds and delivery. The standing approximately USD 10 cap does not authorize spending in this USD 0 lane. `labs/_budget_l191.py` records every attempt and kills its process group at the remaining cutoff. A budget stop means INCOMPLETE. No model training or inference occurs.

## Frozen target and exact commands

Source: https://arxiv.org/html/2604.12596v1,14April 2026. Tables 3, 4, 7, 8;18/4/17/6 method rows;12/5/9/2 task columns;401 task scores plus90 published summaries. Every source byte and policy file is SHA256-authenticated in `evidence/l191/input-manifest.json`. The full source and extracted decimal strings are embedded in both notebooks.

From the repository root, in its existing environment:

```bash
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_test_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_replay_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_verify_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_figures_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_build_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_execute_l191.py
.venv/bin/python labs/_budget_l191.py .venv/bin/python labs/_delivery_l191.py
```

Budget is cumulative across reruns. `labs/_prepare_l191.py` creates the original frozen packet once and refuses overwrite. Reproduction uses the saved packet, not a fresh source download. A new source version requires a new experiment packet and comparison contract. The notebook requires Python 3.10+ standard library only; its external links are optional. Authoring/check tools use the recorded versions in `sources/l191/environment.json`.

## Numerical protocol

- No model/splitter seeds, optimizer, fitting, or train/test reconstruction are used: this is deterministic published-table arithmetic. Python binary64 operations are independently checked with rational arithmetic (`fractions.Fraction`).
- Keep AUROC on the printed 0–100 scale. Task gap is Kumo minus comparator. MAE task gap is comparator minus Kumo.
- Regression aggregate is the equally weighted mean of per-task MAE/LightGBM MAE, with a shared per-task denominator. Report aggregate differences in normalized units. Never pool AUROC and MAE or average heterogeneous raw MAEs.
- Eligible rows require an inspected open implementation and the same published table. The bounded foundation pool comprises RDBLearn/Griffin; supervised comprises LightGBM/GraphSAGE/RelGNN/RelGT. This access policy does not certify historical protocol matching, checkpoint openness or all backend licenses. Other rows are preserved and explicitly excluded from this limited access audit.
- Best single method is the complete-coverage row with best aggregate. Aggregate ties use absolute tolerance 1e-12. Taskwise oracle selects the best displayed eligible value per task and keeps exact ties. Both are retrospective test summaries, not validation-selection policies.
- Missing candidate cells exclude the candidate from whole-table selection; no eligible comparator yields NO_ELIGIBLE_COMPARATOR. Frozen source has no missing task cells. Unexpected table dimensions/non-numeric tokens fail closed, rather than silently changing the experiment.
- Published summaries remain separate. A cell with p decimals represents its printed value ±0.5×10^-p; propagate those intervals through positive ratios and averaging. Also allow half a unit of the printed aggregate precision. These conservative rounding bounds are not confidence intervals.
- Rank all displayed methods separately per task; ties share average occupied rank. Average those ranks equally across task columns. Published rank pools and hidden precision are not recoverable; a difference is DIFFERS_DISPLAYED, not proof of a paper error.

## Findings and deviations

401 task scores and90 summary cells independently parsed; all 45 method rows reconstructed. Five aggregate values are OUTSIDE_ROUNDING_BOUND.34 mean ranks differ under the displayed-row midrank rule. See `evidence/l191/report.json` and `_verify_l191_results.json` for every result, including matching rows. Causes of discrepancies are NOT_ESTABLISHED. No significance/seed uncertainty is inferred from rounding.

The paper supplies at most 10k context examples from training plus validation for Kumo; baseline results can originate from different cited studies. Same-table inclusion is attribution, not an independent equal-information-access audit. The current-source ledger is a bounded check as of2 October 2026; OpenRFM and newer RDBLearn results are not spliced into April tables. Kumo GitHub/API 404 and leaderboard endpoint 401 attempts are retained; failed retrieval does not establish nonexistence. A supplemental raw-master README retrieval also returned 404 and is retained under `sources/l191`.

## Full model reproduction gate

Fresh Kumo inference: NOT_RUN. Historical model reproduction: NOT_ESTABLISHED. Current global SOTA: NOT_ESTABLISHED. Required before a future execution: immutable checkpoint/service identity; full database/task hashes and complete(entity,cutoff)query IDs; preprocessing/feature availability; context selection and all seed/draw identities; validation-selection rules; predictions and evaluation code; service permissions/pricing and total bounded cost. The paper's script link is recorded, but a complete historical runnable protocol has not been recovered. Do not invent a Kumo trainer or present a replacement model as source reproduction.

Live Colab and deployment remain NOT_CHECKED. Offline solution execution, local browser tests and clean-index Pages staging are distinct delivery checks, recorded separately. Learner remains PENDING_WRITTEN_DEFENSE.
