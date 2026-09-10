# L059: selection versus evaluation, v2

[Lesson](../lessons/0059-validation-set-overfitting.html) · [Run/download notebook](0059-validation-set-overfitting.ipynb) · [Read-only preview](html/0059-validation-set-overfitting.html#lab-exercises) · [Reference](../reference/0059-validation-set-overfitting.html)

## Exactly what can be rerun

The historical `_verify_l059_results.json` remains byte-identical and its measured `benchmark_core.null_search` remains unchanged. Its saved seed is a repetition index; RNG seed=59+index. The notebook replays all 1,000 repeat-budget rows using the student's selector. The new operator `relkit/validation_audit_l059.py` adds the paper's Gaussian mixture, KRR with an unpenalized intercept, exact deleted residuals, internal/external outer evaluation, and same-training-size fresh comparisons. Source and data are visible; no cloud or checkpoint is needed.

From the repository root, using the course environment:

```bash
.venv/bin/python labs/_check_l059_v2.py
.venv/bin/python labs/_source_check_l059_v2.py
.venv/bin/python labs/_run_l059_v2.py --output labs/data/cache/l059-rerun.json --repetitions 30
.venv/bin/python labs/_run_l059_v2.py --output labs/data/cache/l059-precision-rerun.json --repetitions 1000
.venv/bin/python labs/_figures_l059_v2.py
.venv/bin/python labs/_build_l059.py
```

The last two commands regenerate only 059. They use the committed author evidence, not the fresh cache outputs. To create new author evidence, pass a deliberately chosen output path and review differences; never relabel an old evidence hash. The local teacher solution is built under `labs/solutions/` and remains ignored by Git. Run it with the course Jupyter kernel. Student TODOs and written EXIT are intentionally incomplete.

## Protocol and measurement target

- Data: paper§3.1 four equal-probability Gaussian components, variance.04 per coordinate,64 rows per independent repetition;4,096 new evaluation rows after all selections freeze.
- Candidate grid: λ=.01/.1/1; η1 andη2=.25/2/16,27 ARD recipes in fixed order; raw known-distribution coordinates, no learned normalization.
- Inner selector: exact LOO PRESS/n, including the intercept. First minimum wins.
- Outer: four equal shuffled label-blind folds; internal selection reruns on 48 rows. External selection uses all 64 labels first and therefore contaminates each outer-held block.
- Matched fresh evaluation: the same 48-row model is scored on new rows. The separately recorded selected-LOO gap compares 63-row deleted fits with a64-row full fit and includes that size difference.
- Independent units:30/1,000 synthetic dataset draws; four overlapping folds averaged first. Seed ranges 5900–5929/5900–6899. Longer-run first 30 are the same observations, not a confirmation set.
- Uncertainty: paired Student-t95 interval over repeat-level differences and Monte Carlo SE. Multiple contrasts are exploratory. A correct run need not show an interval excluding zero.

[30-repeat author evidence](_verify_l059_v2_results.json) · [1,000-repeat evidence](_verify_l059_closer_results.json) · [Checks](_check_l059_v2_results.json) · [Source inventory](_sources_l059_v2.json).

## Fidelity ledger

[Cawley–Talbot full paper](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) §§2–6 supplies the equations, generator and evaluation distinction. This 29-page paper has no appendix. Eq 3 and the LOO shortcut are checked against independent deleted-row refits and equation residuals. The original HTTP GKM URL serves the toolbox; HTTPS fails certificate verification. We pinned the archive hash and inspected the KRR/RBF/LOO/criterion/cross-validation/simplex implementation. Unchanged selected files with original GPLv 2-or-later notices are in `sources/l059/`. Eighteen translated Cholesky/Schur source-algebra cases match the local augmented solve; [source-check report](_source_check_l059_v2_results.json). Native MATLAB execution, optimizer/runtime parity and identity with the exact 2010 experiment revision are NOT_ESTABLISHED.

The original Figure 2 uses iterative ARD optimization, inner four-fold MSE and 1,000 independent 64-row datasets. The lab uses a fixed 27-candidate grid and analytic LOO; matching the count does not align its optimizer. Figures 4–8 use smoothed error surfaces, fixed 256-row fitting data and varying validation draws. Table 8 uses thirteen real benchmarks, RBF, ten outer folds and different released partitions. Those paper experiments remain NOT_RUN / INCOMPARABLE. The 1,000-repeat track is a measured precision extension of this protocol, not a paper-result reproduction.

For a closer future replication, execute the pinned GKM implementation with original benchmark partitions, recover optimizer initialization/stopping/settings, use the relevant figure's criterion and split structure, preserve every trajectory, and compare the same metric with its reported aggregation. Do not combine Figure 2 and Table 8 into a single invented target.

## Delivery and EXIT

The early launcher and preview expose run/download/exercises/reference/evidence/reproduction links without JavaScript. Notebook figures are inline PNGs with full-size links. Live browser and copied-Pages checks are recorded in `reviews/lesson-quality-audit-047-070/059.md`; live Colab and deployment are separate checks.

The required notebook precision track runs the same live student definitions. EXIT saves `data/cache/l059-student/exit.json`, including both experiments, call counts, actual source text, evidence identity, an 80–180 word interpretation, and a frozen lesson 060 decision ledger. There is no promise of resumable edited-code identity. No paid/cloud job is launched by any command above.
