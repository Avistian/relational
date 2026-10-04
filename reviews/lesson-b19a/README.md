# B19a author review

[Lesson](../../lessons/b19a-predictive-distributions.html) · [Student notebook](../../labs/b19a-predictive-distributions.ipynb) · [Executed solution](../../labs/html/b19a-predictive-distributions.html) · [Reproduction contract](../../labs/b19a-reproduction.md) · [B23 metric contract](../../labs/b19a-metric-contract.md).

## Outcome and evidence

Complete selected published-table reconstruction: ScoringBench v3 Tables 1/4/5 (CRPS/R2/CRLS). All 38 models, 97 retained datasets, 114 model rows and 798 displayed fields match at printed precision (six numeric fields and one magnitude label per row; CI endpoints counted separately). Full-precision released leaderboard fields match with atol/rtol 1e-10. The original model roster and original missing-data policy are preserved. All 18,480 historical-release fold records are retained; no imputed substitute scores.

The source manifest authenticates 84 artifacts, including the 38 Git-LFS-authenticated Parquet files and three leaderboard JSON files. Historical source/output commits are frozen. Full raw union is 100 dataset names; source policy drops Abalone, Santander_transaction_value and isolet, leaving 97 common datasets. Every present model/dataset pair has all five folds. Dataset-level effects are separately available for TabICLv2 versus quantile CatBoost; units are not silently pooled.

This is saved-score reconstruction, not fresh model inference/training or raw predictive-distribution rescoring. Original checkpoint, row-level split and historical environment identities are not authenticated by numerical table agreement. Full paper/pretraining/fresh fits remain NOT_RUN. A fresh-run switch explicitly refuses execution outside the approved scope.

Historical issues retained: paper stratification claim versus ordinary shuffled KFold; loader fits imputation/encoding on full features before split; approximate leading CRPS rank in prose differs from Table 1; rank origin differs between prose sections. The replay preserves these choices without presenting them as recommended methodology or measuring their score impact. A new contract fits preprocessing on training data only.

## Teaching and implementation

B19's comparison identity leads causally to metric identity. Worked CRPS/interval calculations precede the complete finite diagnostic. Nine forecast/outcome rows, 15 mass triples and 18 unit/translation interventions are exhaustive for the declared course scope. The truth forecast has minimum expected CRPS in the finite candidate family, while an exactly correct point forecast can win on a single observation. Nine calibration residuals and four distinct test rows demonstrate the calibration boundary; no theorem is inferred from that deterministic fixture.

Three student functions are visible and load-bearing: crps, interval_score and calibrate. Independent CDF integration checks the pairwise-expectation CRPS implementation. Nine deliberately corrupted diagnostic packets and three malformed score packets are rejected. The source hash test rejects changed primary bytes. Three individual blank student tasks and three plausible wrong implementations are rejected.

Portable solution executes 15 code cells from an empty directory with exact diagnostic and complete mean-rank/median replay equality. It contains all selected score records and three embedded PNG figures, and requires only Python standard library for its numerical work. Repository verification additionally authenticates extraction and recalculates full autorank table statistics. Notebook replay alone does not perform those additional checks. The 97 paired per-dataset effects are saved.

## Delivery

Desktop 1200px and mobile 375px: all 22 distribution/calibration control states agree with numerical evidence; keyboard selection/reset, print/no-JS, 40 local links, home/gallery manifest navigation and notebook images pass. CDF/score/pipeline figures and mobile board renders inspected. Scrollable visuals have hints and keyboard focus. Shared retrieval/prediction/teachback checks pass. Final prose review corrected a single-observation example; lesson/notebooks regenerated and delivery checks rerun.

Copied Pages workflow passes under a temporary index, including the inherited uncommitted lesson packages required by the existing workflow; real index remains unchanged. The current package remains local and uncommitted. No push/deployment or live Colab test was requested. Package seal excludes mutable accounting and self-referential build metadata; see exact final machine-readable counts in labs/_pages_b19a_results.json and labs/evidence/b19a/seal.json.

## Cost and status

Approved USD0 paid; 3600-second aggregate local numerical cap. Ledger includes failed RED attempt, dependency preparation, verification, repeated builds and a conservative 60-second pre-wrapper preparation allowance. Final actual accounting is in labs/evidence/b19a/local-budget.json. No model/cloud dispatch occurred.

Learner PENDING_WRITTEN_DEFENSE. Next action for the learner: implement the three operations and submit the metric/calibration contract before B23. Author delivery does not establish learner mastery, new empirical superiority or whole-paper reproduction.

Final accounting: **309.502/3600 seconds**, USD0 paid. Final copied build:125 byte-identical B19a files and40 local links. Seal:126 files. Full solution parity, numerical audit, rendered delivery and shared pedagogy checks PASS.
