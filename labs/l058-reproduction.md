# L058: frozen TALENT results and tiny-benchmark selection

[Lesson](../lessons/0058-surveys-meta-benchmarks.html) · [Read-only preview](html/0058-surveys-meta-benchmarks.html) · [Notebook](0058-surveys-meta-benchmarks.ipynb) · [Reference](../reference/0058-surveys-meta-benchmarks.html)

## What was reproduced

A fresh reanalysis of released rounded mean scores; no predictive models were trained and no raw training arrays were downloaded. Original paper training and Table 7 reproduction: **INCOMPARABLE / NOT_ESTABLISHED**. Operator agreement is separate: the complete six-method ranks match the authors' `calculate_ranks_for_each_dataset` after explicit metric orientation, with spreadsheet input replaced by the exact parsed numeric frame. The upstream script imputes missing ranks; this lab does not. The checked historical script uses `DataFrame.applymap`; a declared pandas3 compatibility alias maps that call to `DataFrame.map` with the identical elementwise callback.

Source: [TALENT arXiv v3, §§4–5 and8](https://arxiv.org/html/2407.00956v3), [pinned release](https://github.com/LAMDA-Tabular/TALENT/tree/1b973adffa4203c3f62c4949957b1ba3efbe60d3). Sources are historical even though the repository notice points to newer corrected results. The old Figure3 binary/regression counts conflict with §4; checked local files have120/80/100 rows. Original model/split/seed data are not reconstructed from Markdown SDs.

## Exact local tracks

| Track | Tasks and methods | Randomness | Status |
|---|---|---|---|
| Archived initial058 | 300 tasks, six methods; deliberately XGB-favored45 | Original operator and sort unchanged | Retained `_verify_l058_results.json` |
| Fresh six-model audit | Same300 and six methods; complete coverage | 2,000 paired dataset draws, seed58 | `_verify_l058_v2_results.json` |
| Tiny selection | 116 binary/60 multiclass/100 regression complete tasks; 13 classifier/12 regression methods | 5 cyclic method holdouts × seeds58,59,60;1,000 proposals | Same v2 artifact |
| Closer random search | Identical276 complete tasks; select17/9/15 tasks | Same nested streams;10,000 proposals | `_verify_l058_closer_results.json` |
| Paper Table7 | Original corrected source population, subsets and method splits | Original selector/seeds unavailable here | Not regenerated |

Four binary and20 multiclass tasks are excluded from the tiny panel because the historical TabPFN result is missing. Complete-case filtering changes the target population. All excluded IDs and selected IDs are recorded. The selection budget matches the paper only in the closer track; there are41 selected tasks locally, not45. The six-model comparison remains complete300.

Seen ranks are recomputed among seen columns before selection; unseen ranks are recomputed independently after the task IDs freeze. The selected subset minimizes mean absolute difference between its mean ranks and the full eligible-panel means, matching the detailed §8.2 prose. Equation5 displays unnormalized sums; its surrounding prose describes averages. The selection text/Table7 caption use MAE; the applications paragraph calls the metric MSE. We declare the local objective instead of claiming exact equation or table parity. Tiny Benchmark1's balanced tree/DNN/tie diagnostic is explained, not recreated. Its15/12/18 allocation is different from Tiny Benchmark2's stated15% task-type sampling.

Classification reserves3 of13 methods and regression reserves2 of12, using five cyclic rotations. Classification held-outs overlap; two regression methods are never held out. No independence or exhaustive-CV claim follows. Seen and unseen MAE have different rank scales; compare selected-versus-random changes within each pool. Search seeds are not model-training seeds. Dataset bootstrap intervals are percentile intervals conditional on reported means, not t intervals, seed intervals or future-domain guarantees.

## Execute and rebuild

From the repository root with `requirements-labs.txt` installed:

```bash
.venv/bin/python labs/_check_l058_v2.py
.venv/bin/python labs/_source_check_l058_v2.py
.venv/bin/python labs/_verify_l058_v2.py --output /tmp/l058-fresh.json
.venv/bin/python labs/_verify_l058_v2.py --trials 10000 --output /tmp/l058-closer.json
.venv/bin/python labs/_figures_l058.py
.venv/bin/python labs/_build_l058.py
.venv/bin/python labs/_execute_l058.py
```

The lab itself runs both budgets through the student's live functions, without a subprocess that imports completed reference TODOs. Its source definitions and result artifacts are saved in `labs/data/cache/l058-student/`. It makes no resume promise. The teacher notebook is locally executed and gitignored. Rebuilding notebooks clears execution; `--keep-notebooks` preserves it while rebuilding prose/preview. No paid/cloud job or pretrained model download is required. Live Colab is not validated by local execution.

The legacy `_verify_l058.py` / `_run_foundation.py --lesson58` path remains the historical initial analysis and does not regenerate the new tiny-track evidence. Do not use its `closer` preset as evidence for this repaired track. The lesson-specific commands above are canonical.

## Source and evidence inventory

- `_sources_l058.json`: exact original source URLs/hashes, unchanged.
- `_sources_l058_audit.json`: additional official evaluator/training source inventory and reading record.
- `relkit/talent_audit_l058.py`: canonical evaluator; inlined into the notebook, with five student functions left blank.
- `_source_check_l058_v2_results.json`: complete-panel rank agreement only.
- `_verify_l058_v2_results.json`, `_verify_l058_closer_results.json`: independently measured new artifacts with source/module/version identities.
- `_execution_l058_results.json`: actual local solution execution, including the10,000-candidate track.
- `../reviews/lesson-quality-audit-047-070/058.md`: substantive review and performed checks. Browser, copied Pages, live Colab and publication are distinct delivery claims.
