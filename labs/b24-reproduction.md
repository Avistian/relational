# B24 reproduction contract — B24-RESEARCH-DEFENSE-AUDIT

Approved 2026-10-04, including GitHub Pages publication. Scope is complete B23 saved-evidence replay and a research-defense lab. No new inference or fitting.

## Frozen inputs and computation

RDB-PFN arXiv2603.03805v5 Table9, rel-f1/driver-dnf; all30 published-checkpoint evaluations and10 course logistic fits from B23. Ten paired support draws of512training rows;702fixed queries,72features; all28,080probabilities retained. Exact (driverId,date) keys, labels, support identities and preprocessing unchanged. Selected source code a95378225478daa262b85f180d482da7516b0af6, data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad, TabICL weights eaf789a9b25ee8486d6f48997ba076f850bbc30b remain inherited, not re-downloaded. Archive digest frozen in evidence/b24/source-identity.json. The portable B24 packet embeds the entire original B23 portable archive byte-for-byte.

Arms: RDBPFN/model_eval00528, RDBPFN_single/model_eval00360, TabICL0.1.3/v1.1 (32estimators), and B23's fixed C1 logistic baseline. Historical fresh inference used float32/RNG42; no B24 RNG or selection. B23 cloud versions Python3.11/torch2.5.1/NumPy1.26.4/sklearn1.6.1; logistic artifact generated with local NumPy2.5.0/sklearn1.9.0. B24 replays the float32 scaler convention and records current runtime separately.

Independent average-rank AUROC, complete grid/key/support/label/probability checks, pair-by-draw effects, sample SD; independently reconstruct saved baseline coefficients' probabilities without refitting. Compare all40metrics and every model/draw with B23 within1e-12; baseline probabilities within1e-10. Published means .7219/.6640/.7176; inherit descriptive .02 closeness boundary from B23, not a new equivalence test. Rank AUROC versus B23's pair-count arithmetic differs below1.2e-16; reconstructed logistic vectors exactly equal.

## Evidence boundaries and deviations

B24 COMPLETE_SAVED_EVIDENCE_REPLAY. B23 inherited COMPLETE_SELECTED_RELEASE_EVALUATION. Current historical checkpoint identity/raw feature availability NOT_ESTABLISHED; checkpoint width96 versus paper128; released label orientation complements current raw reconstruction. Full DFS regeneration, fresh B24 inference, pretraining, untouched task and whole-paper benchmark NOT_RUN. Course logistic is not tuned trees; pretraining budget not equalized. One shared test population and previously used task. Archive hashes establish byte identity, not historical truth.

The defense checker validates supplied scores/flags and two structural test contracts. It does not score prose, prove task novelty or certify information availability. All illustrative inputs are labeled; learner PENDING_WRITTEN_DEFENSE. Missing historical reproduction blocks claims requiring it even if selected release replay completes.

## Execution and verification

From repository root:

```bash
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_audit_b24.py
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_verify_b24.py
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_build_b24.py
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_execute_b24.py
.venv/bin/python labs/_budget_b24.py .venv/bin/python labs/_delivery_b24.py
```

Portable student/solution notebooks embed evidence, independent audit and visible load-bearing functions. NumPy is the sole replay dependency. No account, repo checkout, model checkpoint download or training required. B23's original fresh operator stays at modal/b23_repro.py under its separate authorization/budget; do not dispatch it as a B24 replay.

USD0cloud/API;3600aggregate local numerical/check seconds including120seconds initial preparation allowance, all failures/retries and delivery. Wrapper reserves remaining time and kills descendants at cutoff. Ledger evidence/b24/local-budget.json. Publication verification and any final accounting allowance included; no silent truncation. Live Colab NOT_CHECKED.
