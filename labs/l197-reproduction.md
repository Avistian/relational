# L197-LANDSCAPE-EVIDENCE-AUDIT

Approved scope: complete saved-evidence reconstruction and replay, USD0 cloud/API, aggregate 1800 local execution seconds including failures, preparation and numerical validation. No model training, checkpoint inference, API calls, repair or deployment. A cutoff leaves INCOMPLETE, never silently fewer tasks/seeds.

Inputs are frozen in evidence/l197/packet with SHA256 and original relative paths in input-manifest.json. The original reports are reference outputs, not substitutes for recomputation. Exact original executable sources are included. Python/NumPy details appear in environment.txt. The local source tree has uncommitted predecessor work; the packet hashes identify the actual inputs independently of repository HEAD.

## Complete declared protocol

- KumoRFM-2 v1 (2604.12596v1), Tables 3/4/7/8: 401 task cells, 90 summaries, 45 method rows, 28 task columns. Parse original HTML, check every extracted cell, compute all original pools, midranks and rounding bounds. AUROC table units are percent; MAE aggregates divide each task by its LightGBM baseline. Foundation pool RDBLearn/Griffin; supervised pool LightGBM/GraphSAGE/RelGNN/RelGT, only where present. Report no comparator explicitly; retain five aggregate and 34 rank discrepancies.
- L195 full packet: five L149 seeds, both GNN and engineered-feature pipelines, 499 validation plus 760 test queries per seed; thirty L182 runs (three arms × ten support draws × 702 test queries). Total 33,650 prediction rows. Original selection receipts and complete entity/cutoff identities remain frozen. Use float64 scoring, common saved FE targets, disclose original GNN float32-target differences. Bootstrap whole drivers using original 2,000 draws, seed 137. This conditional interval does not cover all race/time dependencies or cross-database generalization.
- Full embedded L194 report: authenticate every input, reconstruct all 21 published reference rows/signs, preserve every fresh score as null, all completed seeds as zero, and INCOMPLETE_SOURCE_PREPROCESSING_GATE. Published signs 17/3/1 are not fresh scores. No new L196 source diagnostic is executed.

Independent verification uses BeautifulSoup plus rational arithmetic/rankdata for the tables, SQLite key joins for regression, explicit positive-negative comparisons for AUROC, explicit driver blocks for the original bootstrap and Decimal parsing for the 21 reference rows. It runs frozen independent verifiers in a disposable directory and never modifies predecessor artifacts. Missing, corrupt and extra inputs and three wrong student implementations are rejected.

## Commands from repository root

    .venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_audit_l197.py
    .venv/bin/python labs/_budget_l197.py .venv/bin/python labs/_verify_l197.py

The solution notebook embeds all inputs and displays executable audit functions. Its learner functions control coverage/admission/report and essay submission. Standard-library table reconstruction plus NumPy is enough for the core; independent checks also need SciPy/BeautifulSoup. See the ZIP README and requirements for standalone execution. Student TODO cells deliberately fail until implemented. Default portable mode is replay, not training.

## Evidence boundaries

COMPLETE_SELECTED_EVIDENCE_AUDIT is limited to declared reconstruction and saved predictions. Reused evidence is not an independent replication. Full model reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE; fresh training NOT_RUN; historical identity, general superiority, architectural causality and economic undervaluation NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE. Local notebook/browser/build checks do not establish live Colab or deployment, both NOT_CHECKED.
