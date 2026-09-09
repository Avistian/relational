# L056 reproduction contract

Scope: **evaluator implementation and analysis of released scores**, not a new model.
The curriculum deliverable is leaderboard methodology literacy. No model training is
needed for this unit; hardware escalation and model-training presets are inapplicable.

- Paper: TabArena, arXiv 2506.16791 **v1**, §§2–3; initial IID focus.
- Artifacts: `tabarena-2025-06-12`, CatBoost, LightGBM, RealMLP CPU, TabM_GPU.
- Files: `data/l056/*.parquet`, 253,845 bytes total. SHA256 and public URLs:
  `_sources_l056.json`. Committed result artifacts let the lab run offline after setup.
- Coverage: 51 real datasets; 816 outer splits (17 × 30 + 34 × 9), three regimes,
  four methods = 9,792 rows. These are released model measurements, not new fits.
- Metrics: existing `metric_error`: 1−ROC-AUC, log loss, RMSE. No mixed-unit average.
- Failure policy: require exact common keys; reject duplicates, missingness, nonfinite
  values, mismatched metrics and any imputation. Do not select methods/datasets by test score.
- Audit: average tied ranks within a split, splits within a dataset, datasets equally.
  Pairwise wins follow the same weights. Bootstrap paired dataset gaps with 2,000 draws,
  RNG seed 56; report percentile intervals. Dataset curation limits inference beyond this suite.
- Versions: measured environment in `_verify_l056_results.json`; runtime prints its own.
- Split SDs are descriptive, not IID standard errors; the Lite view has one split, so
  its within-dataset SD is undefined (JSON null). No new model seeds are invented.
- Supplementary Friedman/Nemenyi first ranks each dataset's **mean errors**. This is
  a separate summary, not interchangeable with mean split ranks. Three exploratory
  regime tests; .05/3 gives .0167 if using a Bonferroni family threshold.

Run from the repository root:

```bash
.venv/bin/python labs/_check_l056.py
.venv/bin/python labs/_verify_l056.py
.venv/bin/python labs/_figures_l056.py
.venv/bin/python labs/_build_l056.py
.venv/bin/python labs/_execute_l056.py
.venv/bin/python labs/_delivery_check_l056.py
```

The notebook first analyzes the one-outer-split view (51 datasets), then **requires**
the all-split analysis with the exact same live student functions. Both author analyses
are executed. Figure and source parity are part of the delivery evidence.

## Conclusion ledger

| Bucket | Status | Precisely what it establishes |
|---|---|---|
| Verified here | PASS | Reanalysis of four public method artifacts; all keys/hashes; 51 datasets; 9,792 rows |
| Source primitive | MATCH | Dataset-balanced rank-to-win totals agree with pinned current upstream implementation, max absolute difference <2e−15 |
| Larger result analysis | RUN | All 816 outer splits after the one-split view; no training required |
| Paper claims | CITED | v1 Figure 1 reports its full-pool Elo ranking; no substitution of our four-arm ranks |
| Exact Figure 1 numbers | INCOMPARABLE | Different method pool, summary statistic, evaluator revision; artifact bytes pinned now, not proven identical to those used for v1 |
| Original training | NOT_RUN | Cached errors cannot verify preprocessing, fitted weights, tuning trajectories or training reproducibility |
| Official full leaderboard | NOT_RUN | A separate official reconstruction track below |
| Browser / live Colab / deployment | NOT_CHECKED | Static delivery checks are not these runtime checks |

## Next reproduction track: full official leaderboard

The official frozen-paper example is preserved at `sources/l056/official_paper_leaderboard.py`.
The repository revision is `e7cc6b049f9be11a6df29eb2560d9ccb2399d95c`.
For a separate analysis environment, follow that revision's README installation commands:

```bash
git clone https://github.com/autogluon/tabarena.git tabarena-l056
cd tabarena-l056
git checkout e7cc6b049f9be11a6df29eb2560d9ccb2399d95c
uv venv --seed --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow -e './packages/tabarena[plot]'
python examples/reproducibility/run_generate_main_leaderboard_neurips2025.py
```

These commands are sourced from the pinned repository but **not executed here**. Its
dated collection name includes camera-ready replacements for some methods (see
`contexts/tabarena/methods.py`), so do not assume it recovers v1 exactly. The current
Elo implementation also documents changes to solver tolerance and weighting. Reconcile
method roster, data artifacts, evaluator code, imputation, anchor and bootstrap against
the exact paper version before claiming recovery. `USE_LEGACY_ELO_SOLVER_TOL` only
addresses one solver setting; it is not a complete historical fidelity switch.

If fresh training is the later objective, restore original per-method configurations,
feature/preprocessing rules, complete inner and outer evaluation, hardware and time
budgets. Score-only files cannot establish these. This is explicitly outside L056's
evaluator scope; L057 builds cross-family ensembles and L059 examines validation overfit.

## Source licensing

Pinned upstream `elo_utils.py` and the official example retain the upstream Apache-2.0
license in `sources/l056/LICENSE`. Benchmark result artifacts are attributed to TabArena;
their source locations and exact downloaded bytes are recorded separately. No raw
dataset records or model predictions are redistributed by this lesson.
