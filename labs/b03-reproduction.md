# B03 · PFN and the TabPFN generations

**Named experiment:** B03-TABPFNV2-BLOOD-OFFICIAL-SPLITS. **Status: INCOMPLETE_SOURCE_PROTOCOL. Zero benchmark runs.**

The approved target is default historical TabPFN v2 classification on blood-transfusion, all ten official benchmark splits. No newly generated splits replace the original experiment. The complete paper benchmark and original synthetic pretraining remain NOT_RUN.

## Recovered source and missing identities

The Nature paper (DOI 10.1038/s41586-024-08328-6, Detailed evaluation protocol) describes ten 90/10 repetitions. Its Zenodo record 13981285 contains `TabPFN-main.zip` (SHA256 `936f4a9433bb89ec660dd0343ed5dca61c842ae9c60e271dc82c87a3a99a2db5`) and `TabPFN-original-stale.zip` (`0a995999b235d47eba66488c42cd565d7646008589ba1693af94f167dd4cd80a`). Both were downloaded and inspected on 2026-10-03.

The first is an inference package with examples. The second retains `tabpfn/scripts/tabular_evaluation.py`, but imports absent `tabpfn.datasets`, including `get_benchmark_for_task` and `load_openml_dataset`. This is where task IDs, row processing and split generation would be recovered. The script sets N_SPLITS=5; a notebook helper uses range(0,10), and evaluation utilities contain an unresolved official-split comment. None authenticates the paper's ordered ten splits. The GitHub path-history API for `tabpfn/datasets.py` returned an empty list; that scoped lookup is not proof that no release exists elsewhere.

OpenML dataset 1464 has multiple classification tasks. Retrieved task 10101 and task 145836 metadata both describe ten-fold cross-validation. A shared dataset and split proportion do not identify the paper's task. We have not recovered the exact per-split reference values or a complete historical configuration-to-checkpoint mapping. The later 2.0.9 classifier wheel and v2 checkpoint are authenticated for the **separate diagnostic only**.

## Executable source gate

From repository root:

```bash
.venv/bin/python labs/_reproduce_b03.py
.venv/bin/python labs/_reproduce_b03.py --run
```

The first prints the independently reconstructed source gate. The second exits nonzero before model execution. It is a runnable preflight, **not a complete working benchmark runner**. Original archive bytes and upstream evaluation source are preserved in `labs/sources/b03` and the portable reproducer. To unblock: recover the missing original loader and exact benchmark roster; authenticate all ten ordered train/test row sets, preprocessing, seed/view/temperature recipe and checkpoint; recover the numeric reference; add tests of those identities; then implement dispatch and budget admission. There is deliberately no override flag.

## Separate fresh course diagnostic

Pinned `tabpfn==2.0.9`, scikit-learn 1.6.1 and v2 classifier SHA256 `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`. Uses sklearn Iris, all 150 rows, one stratified 75/75 split with seed 0. All six permutations of three class labels are applied to support labels; every prediction is restored to original class-column meanings. Four default preprocessing/ensemble views, model seed 0, CPU float32, fit_preprocessors, one thread. No HPO, feature subsampling, missing-value interventions or class-cap workaround. Tolerance fixed at atol 1e-6, rtol 0 before execution. Seven predict calls: six permutations plus one identical-query repeat. The declared class-label intervention is unrelated to benchmark score reproduction.

The worker accepts only support X/y and query X at inference. Repeated prediction after altering an unused copy of scoring labels checks that scorer data has no path into this worker; it does not establish universal absence of leakage or query-feature independence. Historical L064 contains separate counterexamples to query-feature independence.

`_diagnostic_b03.py` retains raw tables, labels, row IDs, all450 probability rows and75 repeated rows, config, versions and checkpoint identity. `_audit_b03.py` authenticates the input identities and uses scalar loops to realign columns, calculate maximum errors, accuracy and log loss. The notebook repeats this audit using learner functions. Accuracy is 94.67% in every permutation, although aligned probabilities change by up to 0.071616888. Equal accuracy is not equivariance. One fixture gives no population uncertainty estimate.

To rerun the diagnostic in the already prepared historical environment:

```bash
PYTHONPATH=labs/data/cache/l064-source/official .venv/bin/python labs/_budget_b03.py .venv/bin/python labs/_diagnostic_b03.py
```

The worker refuses to overwrite evidence. Use a copied checkout with a fresh evidence destination for a rerun. For a new environment follow `l064-reproduction.md` / `requirements-foundation-v2.txt`; use the explicit checkpoint URL in `source-lock.json`. A moving `pip install tabpfn` is not this protocol. Current author Python/NumPy/Torch versions are recorded; this is not a historical hardware/runtime reconstruction or speed benchmark. The existing L064 visible model is included for architectural study; the fresh diagnostic executes original release code, not that educational implementation.

## Budget and evidence

USD10 approved aggregate ceiling; USD8 dispatch cutoff + USD2 reserve. **Actual cloud/API USD0**: source failure prevented paid admission. Local numerical work, failed checks and retries are accounted in local-budget.json under 3600 seconds, with a preparation/bookkeeping allowance. Full original pretraining, tuned competitors, current-generation evaluation and paper benchmark remain NOT_RUN. Source parity and diagnostic completion never imply those claims.

Portable solution execution, browser/mobile, copied Git-index Pages build, live Colab and deployment are separately recorded delivery checks. Learner remains PENDING_WRITTEN_DEFENSE. No publication requested.
