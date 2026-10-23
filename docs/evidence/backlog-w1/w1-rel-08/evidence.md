# REL-08 blocked audit

Status: BLOCKED. No checker implementation, TDD pass, separate review, or hosted CI is claimed.

Source audited: `00f3dfd721dc66bbab3f9672705fd0f60fe8a416`.

## Actual notebook validation

Shared interpreter: `/home/avist/Projects/relational/.venv/bin/python`.
Executed `nbformat.validate(json.loads(path.read_text()))` for every `Path.cwd().rglob("*.ipynb")` result, without notebook conversion or writes.
Actual results: notebooks=410, invalid=0, warnings=464.

Directory counts:
```json
{
  "labs": 221,
  "labs/evidence/l149": 1,
  "labs/solutions": 148,
  "labs/sources/b04a/upstream/ticl/analysis": 1,
  "labs/sources/b04a/upstream/ticl/notebooks": 19,
  "labs/sources/b05/paper-era/notebooks": 1,
  "labs/sources/b19": 4,
  "labs/sources/b19b/gift/notebooks": 1,
  "labs/sources/b21": 1,
  "labs/sources/l129/f1": 1,
  "labs/sources/l129/f1/driver-position": 1,
  "public-test/labs": 11
}
```

Validation errors:
```json
[]
```

Warnings (retained, not suppressed into a clean claim):
```json
[
  [
    "labs/0006-missingness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0006-missingness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0007-class-imbalance.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0007-class-imbalance.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0008-metrics-calibration.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0008-metrics-calibration.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0009-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0009-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0010-baseline-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0010-baseline-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0011-decision-trees-partitions.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0011-decision-trees-partitions.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0012-bagging-random-forest.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0012-bagging-random-forest.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0013-boosting-intuition.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0013-boosting-intuition.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0014-xgboost.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0014-xgboost.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0015-lightgbm.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0015-lightgbm.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0016-catboost.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0016-catboost.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0017-hyperparameter-search.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0017-hyperparameter-search.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0018-ensembling-stacking.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0018-ensembling-stacking.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0019-when-trees-win.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0019-when-trees-win.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0020-q2-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0020-q2-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0021-data-splits-in-the-wild.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0021-data-splits-in-the-wild.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0022-label-leakage-patterns.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0024-grinsztajn-benchmark.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0025-inductive-bias-smoothness.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0026-inductive-bias-rotation.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0027-inductive-bias-uninformative-features.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0028-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0029-manual-fe-vs-automl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0030-q3-checkpoint.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0031-embeddings-for-categoricals.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0032-tabtransformer-preview.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0033-when-to-stop-feature-engineering.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0034-relational-data-without-rdl.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0035-what-joins-destroy.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0036-revisit-your-homework-pipeline.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0037-document-a-baseline-package.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0040-year-1-exit-exam.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0040-year-1-exit-exam.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0042-mlp-resnet-baselines.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0043-tabnet.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0044-node.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0045-tabtransformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/0046-ft-transformer.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ],
  [
    "public-test/labs/LAB-TEMPLATE.ipynb",
    "MissingIDFieldWarning",
    "Cell is missing an id field, this will become a hard error in future nbformat versions. You may want to use `normalize()` on your notebooks before validations (available since nbformat 5.1.4). Previous versions of nbformat are fixing this issue transparently, and will stop doing so in the future."
  ]
]
```

## Exact pair inventory

Student=221, HTML=222, solutions=148. Pairing uses labs/<stem>.ipynb, labs/html/<stem>.html, labs/solutions/<stem>.ipynb.

This is inventory, not an approved exception baseline. Existing evidence: labs/README.md documents the template and private ignored solutions; .gitignore names distributable solution exceptions; labs/_execute_l113.py writes the extra solution render; notebooks.html links foundation-sequence as a materials directory. No broad allowances were added.

### student without HTML: 1

- `labs/LAB-TEMPLATE.ipynb`

### HTML without student: 2

- `labs/html/0113-scaling-ogb-solution.html`
- `labs/html/foundation-sequence.html`

### student without solution: 73

- `labs/0006-missingness.ipynb`
- `labs/0007-class-imbalance.ipynb`
- `labs/0008-metrics-calibration.ipynb`
- `labs/0009-feature-engineering.ipynb`
- `labs/0010-baseline-checkpoint.ipynb`
- `labs/0011-decision-trees-partitions.ipynb`
- `labs/0012-bagging-random-forest.ipynb`
- `labs/0013-boosting-intuition.ipynb`
- `labs/0014-xgboost.ipynb`
- `labs/0015-lightgbm.ipynb`
- `labs/0016-catboost.ipynb`
- `labs/0017-hyperparameter-search.ipynb`
- `labs/0018-ensembling-stacking.ipynb`
- `labs/0019-when-trees-win.ipynb`
- `labs/0020-q2-checkpoint.ipynb`
- `labs/0021-data-splits-in-the-wild.ipynb`
- `labs/0022-label-leakage-patterns.ipynb`
- `labs/0023-statistical-comparison.ipynb`
- `labs/0024-grinsztajn-benchmark.ipynb`
- `labs/0025-inductive-bias-smoothness.ipynb`
- `labs/0026-inductive-bias-rotation.ipynb`
- `labs/0027-inductive-bias-uninformative-features.ipynb`
- `labs/0028-mlp-resnet-baselines.ipynb`
- `labs/0029-manual-fe-vs-automl.ipynb`
- `labs/0030-q3-checkpoint.ipynb`
- `labs/0031-embeddings-for-categoricals.ipynb`
- `labs/0032-tabtransformer-preview.ipynb`
- `labs/0033-when-to-stop-feature-engineering.ipynb`
- `labs/0034-relational-data-without-rdl.ipynb`
- `labs/0035-what-joins-destroy.ipynb`
- `labs/0036-revisit-your-homework-pipeline.ipynb`
- `labs/0037-document-a-baseline-package.ipynb`
- `labs/0040-year-1-exit-exam.ipynb`
- `labs/0042-mlp-resnet-baselines.ipynb`
- `labs/0043-tabnet.ipynb`
- `labs/0044-node.ipynb`
- `labs/0045-tabtransformer.ipynb`
- `labs/0046-ft-transformer.ipynb`
- `labs/0047-saint.ipynb`
- `labs/0048-dcnv2.ipynb`
- `labs/0049-excelformer-trompt.ipynb`
- `labs/0050-q1-checkpoint.ipynb`
- `labs/0051-why-trees-still-win.ipynb`
- `labs/0052-tabr-retrieval.ipynb`
- `labs/0053-realmlp-strong-defaults.ipynb`
- `labs/0054-tabm-parameter-efficient-ensembling.ipynb`
- `labs/0055-tabred-temporal-splits.ipynb`
- `labs/0056-tabarena-benchmark-literacy.ipynb`
- `labs/0057-cross-family-ensembling.ipynb`
- `labs/0058-surveys-meta-benchmarks.ipynb`
- `labs/0059-validation-set-overfitting.ipynb`
- `labs/0060-broad-model-comparison.ipynb`
- `labs/0061-prior-data-fitted-networks.ipynb`
- `labs/0062-tabpfn-v1.ipynb`
- `labs/0063-synthetic-scm-prior.ipynb`
- `labs/0064-tabpfn-v2.ipynb`
- `labs/0065-tabpfn-query-embeddings.ipynb`
- `labs/0066-tabicl-column-row-attention.ipynb`
- `labs/0067-local-pfn-retrieval-finetuning.ipynb`
- `labs/0068-pfns-under-temporal-shift.ipynb`
- `labs/0069-tabpfn-open-environment-failures.ipynb`
- `labs/0070-foundation-model-checkpoint.ipynb`
- `labs/0071-vime-masked-tabular-ssl.ipynb`
- `labs/0072-scarf-subtab-contrastive-views.ipynb`
- `labs/0073-when-ssl-helps.ipynb`
- `labs/0074-carte-cross-table-transfer.ipynb`
- `labs/0075-pytorch-frame-row-encoder.ipynb`
- `labs/0076-encoder-predictor-stack.ipynb`
- `labs/0077-single-table-ceiling.ipynb`
- `labs/0078-message-passing-preview.ipynb`
- `labs/0079-neural-tabular-decision-guide.ipynb`
- `labs/0080-year-2-exit-exam.ipynb`
- `labs/LAB-TEMPLATE.ipynb`

### solution without student: 0


## Delivery blockers

1. The installed canonical TDD gate in `/home/avist/agent-stack/lead/bin/lead-guard` rejects any `tests/` delta between RED and GREEN (lines 258-260), including the checker seam explicitly authorized by spec contract 6 and Test seams. No workaround or weakened gate was attempted.
2. Required exported verification/environment roots are outside this worker sandbox. Exact failed commands and output are in verification.log. The verification command did not run its tests; environment check could not retain proof.

No builder-done handoff: its committed verification prerequisite is unmet. Lead must resolve these blockers and launch any review through counted lead-spawn.
