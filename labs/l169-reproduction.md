# L169 · Complete selected context-size reproduction

Approved scope: RDB-PFN v5 Tables6–10; rel-f1/driver-dnf and rel-trial/study-outcome; contexts64,128,256,512,1024; three fixed arms RDBPFN,RDBPFN_single,TabICLv1.1; ten support seeds. Full702/825 test queries.300evaluations,60reused512-context evaluations from L166/L168 and240fresh evaluations. Every probability is rescored by full entity/date identity.

Source: https://arxiv.org/html/2603.03805v5 . Code a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL eaf789a9b25ee8486d6f48997ba076f850bbc30b . Checkpoint hashes and per-file hashes in evidence/l169/input-manifest.json. Paper means parsed from original table cells before fresh dispatch;0.02AUROC descriptive tolerance, no equivalence claim.

Sampling uses original SHA256-derived task/seed integer and NumPy default_rng choice without replacement separately for each context. It is NOT a nested-prefix intervention. Match support identities across models at each size; report seed-indexed differences across sizes with this caveat.32TabICL estimators. No test selection, refitting pretrained parameters or target gradient steps. Full train candidates and test queries are preserved.

Labels complement raw task labels; keep released orientation. Candidate-label horizons precede every test query. Source MAX timestamp checks are reused; complete DFS regeneration NOT_RUN. Historical availability/identity, independent checkpoint training lineage and exact target-schema exclusion NOT_ESTABLISHED. Whole paper and fresh pretraining NOT_RUN; context response does not establish a pretraining scaling law.

## Commands

Use the repository environment for evidence replay:

```bash
.venv/bin/python labs/_check_l169.py
.venv/bin/python labs/_audit_l169.py
.venv/bin/python labs/_verify_l169.py
.venv/bin/python labs/_figures_l169.py
.venv/bin/python labs/_build_l169.py
.venv/bin/python labs/_execute_l169.py
.venv/bin/python labs/_delivery_l169.py
```

Fresh inference is explicit. Install Python3.11, torch2.5.1+CUDA12.4, numpy1.26.4,pandas2.2.3,scikit-learn1.6.1,pydantic1.10.26,pyyaml6.0.2,tabicl0.1.3 in a separate environment; source imports live under labs/sources/l166/upstream/model_pretrain. The portable notebook embeds the original source and the visible complete runner.

```bash
.venv/bin/python labs/_fetch_l169.py --out /tmp/l169-input
# Requires the declared GPU environment; all means all240 fresh evaluations.
python labs/_run_l169.py --input /tmp/l169-input --source labs/sources/l166/upstream --out /tmp/l169-new-attempt --phase all
```

The managed author lane uses modal/l169_repro.py with pilot/remaining phases. USD10total including retries/checks,USD3overhead reserve,21600aggregate worker-second ceiling. Reserve each timeout plus30s lifecycle allowance before launch; fixed source hashes reject drift.600s pilot,7200s main timeout. Measured1024-context six-run pilot projected2373.2s for the remaining234runs with3×margin.

Both long synchronous client waits ended with RPC deadlines and cancelled their remote inputs, including the first detached retry. All partial logs, per-run records and cost receipts are retained. The second attempt saved225complete per-run artifacts. A nonblocking `spawn` call through modal/l169_tail.py completed the remaining9runs plus6sentinels; support/key/label arrays match exactly; all four RDB-PFN repeats match exactly, while two TabICL repeats differ by at most0.0002791in probability and0.00000488AUROC. Exact TabICL repeatability FAIL; no post-hoc tolerance converts it into a pass. Only the9missing results enter the experiment.

`_assemble_l169.py` creates an explicitly labeled author-assembled234-run receipt from the225saved original per-run records and9tail records. Original paths and record hashes are preserved and checked; this is not represented as an original whole-worker receipt. `remaining-1` is excluded entirely. Every failed attempt, bounded retry and sentinel counts toward the aggregate budget. No automatic retries. Source-parity checks restrict the tail runner change to job scheduling; predictor operations stay byte-identical.

Author reservation ledgers are immutable attempt histories. For a future paid rerun, start a separately budgeted ledger and new output names; use nonblocking submission and bounded polling rather than a long synchronous wait. The underlying cause of the RPC deadline is not established; this run verifies the bounded completion path, not a general transport fix.

The default notebook performs only CPU saved-evidence replay. RUN_FRESH=False; fresh code is opt-in and manual execution has no cloud billing guard. A notebook replay verifies the saved experiment; it does not become a newly executed inference run. Delivery, liveColab, deployment and learner mastery are separate statuses.
