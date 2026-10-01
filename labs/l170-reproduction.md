# L170 — Complete saved-evidence replay for the FM design checkpoint

Approved scope: RDB-PFN v5 Tables 6–10, rel-f1/driver-dnf and rel-trial/study-outcome; contexts 64/128/256/512/1024; RDBPFN, RDBPFN_single and TabICLv1.1; ten support seeds. All 702/825 test queries per evaluation. 300 evaluations and 229,050 predictions. All are reused in L170; original provenance is 240 fresh L169 evaluations plus 60 from L166/L168.

Sources, query keys, supports, labels, original per-run receipts and diagnostics are authenticated through the original L169 pins. The L170 manifest also freezes the original audit helpers, report, Griffin budget boundary and versioned reading snapshots. Every probability is independently rescored; paired effects preserve full task/context/seed identity. Task-level macro averages assign each of the two tasks one equal weight. No significance or unseen-database guarantee is inferred from support-seed SD.

Code: RDBPFN a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL eaf789a9b25ee8486d6f48997ba076f850bbc30b. Original preprocessing, checkpoints, 32 TabICL estimators and independent-per-context sampler retained. Paper-distance tolerance .02 AUROC is inherited descriptive context, not statistical equivalence. Sources: https://arxiv.org/html/2603.03805v5 . Full original fresh-inference lane: [L169 reproduction protocol](l169-reproduction.md), [_run_l169.py](_run_l169.py), [Modal operator](../modal/l169_repro.py). No fresh lane is launched by L170.

## Execute

From the repository root, use the environment's Python:

```bash
.venv/bin/python labs/_budget_l170.py .venv/bin/python labs/_check_l170.py
.venv/bin/python labs/_budget_l170.py .venv/bin/python labs/_replay_l170.py
.venv/bin/python labs/_budget_l170.py .venv/bin/python labs/_verify_l170.py
.venv/bin/python labs/_figures_l170.py
.venv/bin/python labs/_build_l170.py
.venv/bin/python labs/_budget_l170.py .venv/bin/python labs/_execute_l170.py
.venv/bin/python labs/_delivery_l170.py
```

The standalone notebook embeds the evidence packet and audit functions, needs Python3 + NumPy, and defaults to no network, new inference or cloud calls. Author independent checking additionally uses sklearn. See the execution receipt for tested versions. Lesson construction/browser checks use the existing lab environment.

## Budget and deviations

USD0 new cloud/API cost. Enforced aggregate 600-second numerical replay/check/notebook-execution limit in `evidence/l170/local-budget.json`, including failed attempts. The wrapper records an outstanding reservation before launch and refuses to continue after an interrupted unresolved attempt. Rendering and browser delivery are outside this experimental compute bound. A timeout means stop, not silently shrink the grid.

- Released labels complement pinned raw labels. Preserve orientation; do not reinterpret the event silently.
- Historical identity/availability, exact target-schema exclusion and training lineage NOT_ESTABLISHED. Full DFS regeneration NOT_RUN.
- Exact TabICL repeatability FAIL in two original sentinel reruns; four RDB-PFN runs match exactly. Small numerical differences remain visible without score-based selection.
- Griffin full selected experiment INCOMPLETE_BUDGET_GATE. RDBLearn pipeline NOT_RUN: the RDB-PFN paper's DFS + TabICL arm is only a conceptual comparator.
- Whole-paper reproduction and fresh pretraining NOT_RUN. Context response is not a pretraining scaling law.
- All claim checkboxes are hypothetical declarations. READY_FOR_REVIEW means necessary checklist items are present, not that a scientific claim is true.
- Learner PENDING_WRITTEN_DEFENSE; live Colab and deployment NOT_CHECKED.
