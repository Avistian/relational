# L185: complete original synthetic causal experiment

Experiment `L185 relational-shortcut-intervention-v1`, approved 2026-10-02. The scope is the curriculum's Tier C shortcut demonstration, not an empirical causal benchmark or Pearl/RCD reproduction.

## Frozen protocol

Five seeds 0–4. Each has 1,000 companies ×20 customers. Company-disjoint train/validation/test groups: 600/200/200 companies, yielding 12,000/4,000/4,000 rows. NumPy SeedSequence spawns independent streams for traits, action assignment, outcome noise, and splitting. Traits arrive at0, actions/query at1, outcomes at2; full query key `(customer_id,cutoff)`.

U~Bernoulli(.5), B=U XOR Bernoulli(.05), A~Bernoulli(.1+.8U), E~Uniform(0,1), Y=1[E<.05+.85U+.05A]. U and B are shared company variables; A and E vary by customer. U is observed. E is evaluator-only. No interference; all common causes of A and Y are measured by U in the declared SCM.

Fit train-only conditional-frequency tables for B, A, and (U,A); global training mean fallback for unseen cells. Score all three on validation and test; no search or model selection. No pretraining, checkpoint or optimizer exists for these fixed counting estimators. Use held-out test data for the separate causal estimates. Compare unadjusted action association, U-standardized action effect, paired potential-outcome action and badge effects, and gains of all-treated/all-badged policies over observed outcomes. Pair intervention outcomes using identical U,E; replacing B leaves U,A unchanged. Float64 arithmetic. Report mean and ddof1 standard deviation across seeds, not IID customer confidence intervals.

Population action ATE=.05; badge ATE=0. With balanced demand, observed action risk difference=.73, observed badge risks=.0995/.9005; either badge intervention risk=.5. Test-specific all-treated expected gain is mean(.05*(1-A)), distinct from ATE. Simulated realized effects fluctuate around expectations.

## Run locally

From the repository root, use the project's virtual environment or create one and install the exact versions in `labs/sources/l185/requirements.txt`.

```bash
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_test_l185.py
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_run_l185.py
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_verify_l185.py
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_figures_l185.py
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_build_l185.py
.venv/bin/python labs/_budget_l185.py .venv/bin/python labs/_execute_l185.py
.venv/bin/python labs/_delivery_l185.py
.venv/bin/python labs/_seal_l185.py
```

All author numerical invocations pass through the persistent 600-second guard. Failed tests and retries count; no cloud/API calls. The guard locks the ledger and terminates a command at the remaining allowance. It refuses new numerical jobs after exhaustion. Do not delete the ledger to evade the cap. A learner's separate notebook session is not new author execution and is not remotely metered; it has the same small complete experiment.

The standalone notebooks require no repository clone or dataset download: source and portable figures are embedded. Install the documented versions for exact reference parity. Later package versions require revalidation. The solution executes from an empty temporary directory. A builder rerun overwrites its execution outputs; execute it again before delivery.

## Audit and files

`evidence/l185/seed-N/` retains four normalized input tables, complete held-out predictions and paired potential outcomes. `run-manifest.json` hashes the experiment source, frozen design, inputs, outputs, environment, and report. The independent verifier recomputes training frequencies, rank-based AUROC, complete query sets, intervention truth tables and adjustment, then compares a fresh five-seed run exactly. Numerical bounds frozen pre-run: paired and adjusted action estimates within .025 of .05 per seed; badge effects exactly0; independent metric difference below1e-12; exact table/report rerun.

`_test_l185.py` was run with intentional missing implementations first (RED), then passed after implementation. The first full independent replay failed because a nonsemantic pandas row index was omitted on Parquet serialization; comparison now normalizes the index after verifying semantic keys. No seed, data size, parameter or scientific protocol changed. All attempts are charged in the ledger.

Source definitions: Pearl (2009), DOI10.1214/09-SS057, §§2,3.2,3.3,3.4. Source PDF hash and scope in `sources/l185/source-ledger.json`. Maier et al.2013 is optional relational context only. Numerical parameters, data, experiment and tolerances are original teaching choices; no named published empirical target is available in the curriculum contract.

Status: COMPLETE_SYNTHETIC_EXPERIMENT. Paper reproduction and real-world effectiveness NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Browser, notebook execution and clean-index staging have separate reports. Live Colab and deployment remain NOT_CHECKED.

Author publication check: the full Git-index Pages workflow was blocked by an unrelated staged frontier-lesson sections with missing artifacts (L186 on the first attempt, L187 on the second). `_checkout_l185.py` records that failure and separately verifies copied L185 artifacts/links from index bytes. PASS_PACKAGE_ONLY must not be reported as a successful whole-site build. Stage updated L185 artifacts and shared L185 navigation before running that checker. No other lesson work was removed to force a pass.
