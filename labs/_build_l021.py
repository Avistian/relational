"""Build labs/0021-data-splits-in-the-wild.ipynb (student, blanks) and solutions/ (filled).

Run:  .venv/bin/python labs/_build_l021.py
Then execute the solution to verify:  jupyter nbconvert --execute ...
"""
from __future__ import annotations

import json
from pathlib import Path

LABS = Path(__file__).resolve().parent
SOL = LABS / "solutions"
SOL.mkdir(exist_ok=True)

CELLS = []


def md(src):
    CELLS.append(("md", src))


def code(src):
    CELLS.append(("code", src))


def todo(sol, stu):
    CELLS.append(("todo", sol, stu))


# ---------------------------------------------------------------- header
md("""# Lab 021 — Data splits in the wild: the temporal optimism gap

**Lesson:** [`lessons/0021-data-splits-in-the-wild.html`](../lessons/0021-data-splits-in-the-wild.html) · **Phase / Year:** Year 1 · Q3

**Paper (preview):** Rubachev et al. 2024, *TabReD: Analyzing Pitfalls and Filling Gaps in Tabular Deep Learning Benchmarks*, [arXiv:2406.19380](https://arxiv.org/abs/2406.19380) — abstract + §1 (why temporal splits) and §5.4 (random vs time-based splits change rankings and shrink XGBoost's margin).

**Dataset tier:** C — synthetic (mechanism isolation). We generate a **time-ordered stream** whose label rule *drifts* over time, so the random-vs-temporal gap is real and reproducible. This is *not* a benchmark — it isolates one mechanism. (Real temporal datasets need timestamp metadata that most public benchmarks lack — TabReD's central complaint.)

**Skill you are practising:** build a **temporal split** (train past → test future), the time-series CV analogue (`TimeSeriesSplit`), and a per-time-bucket **drift diagnostic** — then read off how much a random split *overstates* deployment accuracy.

**Exit criteria:** EXIT TICKET prints, for logistic regression and HistGradientBoosting, the random 5-fold CV, the `TimeSeriesSplit` CV, and the temporal holdout — with the optimism gap — plus your one-sentence takeaway. Reproduction target: random ≈ 0.846, temporal ≈ 0.758 (logistic).

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** and **numpy** only (no boosters, no network). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — the assumption a random split makes

`train_test_split(..., shuffle=True)` and `KFold(shuffle=True)` assume the rows are **i.i.d.** — exchangeable — so which rows land in test versus train does not matter. In production that is often false: data arrives over **time** and the world **drifts**. A deployed model always trains on the *past* and predicts the *future*; a random split breaks that arrow of time by scattering test rows across the whole history, letting the model train on the very period it is scored on.

**Temporal split.** Order rows by time, cut once: `train = first frac`, `test = the rest`. The model must extrapolate forward into a period it never saw — exactly deployment.

**Time-series CV (`TimeSeriesSplit`).** The cross-validated version: an *expanding window* where every fold trains on a block of the past and validates on the block immediately after — never shuffled.

```
fold 1:  train [##....]  test [#]
fold 2:  train [###...]  test [#]
fold 3:  train [####..]  test [#]
```

**Concept drift.** The feature→label *rule* changes over time (distinct from the class balance changing). Our stream rotates the rule from feature `x0` to `x1`: `corr(x0,y)` fades 0.72→0.12 while `corr(x1,y)` rises 0.10→0.71, with prevalence flat ≈ 0.50.

**Toy micro-example (not this lab's answer).** Rows in time order with labels `[0,0,1,1,0,0,1,1]`. A random 50% test set (say rows 1,3,5,7) leaves the model rows 0,2,4,6 — every test row has an *adjacent* train row from the same moment. A temporal split (train 0–3, test 4–7) gives the model no row from the test period. If the rule flips at row 4, only the temporal split reveals the failure.

Full write-up + the split-geometry and drift widgets: [Lesson 021](../lessons/0021-data-splits-in-the-wild.html).""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code('''# PROVIDED — a time-ordered stream whose label rule DRIFTS, plus models + a random-CV baseline.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

SEED = 0

def make_stream(n=6000, d=6, drift=np.pi / 2, noise=0.6, seed=SEED):
    """Rows sorted by time t in [0,1). The label rule ROTATES with time:
    early rows are labelled by x0, late rows by x1 (pure concept drift;
    the feature distribution itself is stationary)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    X = rng.normal(size=(n, d))
    theta = t * drift
    logit = np.cos(theta) * X[:, 0] + np.sin(theta) * X[:, 1] + 0.4 * X[:, 2]
    logit = 1.6 * logit + rng.normal(scale=noise, size=n)
    y = (logit > 0).astype(int)
    return X, y, t

X, y, t = make_stream()

def model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    return HistGradientBoostingClassifier(random_state=SEED)

def random_cv(name):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)   # <-- the reflex: shuffle=True
    return cross_val_score(model(name), X, y, cv=cv, scoring="accuracy").mean()

print(f"stream ready: n={len(X)}, features={X.shape[1]}, prevalence={y.mean():.3f}")
print(f"rows are time-ordered (t strictly increasing): {bool(np.all(np.diff(t) > 0))}")''')

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — the temporal holdout — TODO (crucial fragment)

**Goal:** score a model on a **temporal** holdout — train on the earliest `frac` of the time-ordered rows, test on the rest — and compare it to the random 5-fold CV.

**Why it matters:** this single number is the honest estimate of deployment accuracy. The gap between it and the random-CV number is the size of the illusion a random split creates.

**You implement:** the cut. The rows are already time-ordered, so `cut = int(len(X) * frac)`; train on `X[:cut], y[:cut]`, and return the accuracy on `X[cut:], y[cut:]`. Use the model's own `.fit(...).score(...)`.

**Hint boundary:** slice by position (`[:cut]` / `[cut:]`) — do **not** shuffle. `estimator.score(Xte, yte)` returns accuracy.""")

todo(
    '''# TODO — implement the temporal holdout score
def temporal_holdout(name, frac=0.8):
    cut = int(len(X) * frac)
    m = model(name).fit(X[:cut], y[:cut])
    return m.score(X[cut:], y[cut:])

split = {}
for name in ("logistic", "hist_gbdt"):
    r_cv = random_cv(name)
    t_ho = temporal_holdout(name)
    split[name] = {"random_cv": r_cv, "temporal_ho": t_ho}
    print(f"  {name:10s}: random-CV {r_cv:.3f}   temporal-holdout {t_ho:.3f}   optimism gap {r_cv - t_ho:+.3f}")''',
    '''# TODO — implement the temporal holdout score
def temporal_holdout(name, frac=0.8):
    cut = int(len(X) * frac)
    m = model(name).fit(____, ____)      # train on the EARLIEST frac of rows (past)
    return m.score(____, ____)           # score on the remaining rows (future)

split = {}
for name in ("logistic", "hist_gbdt"):
    r_cv = random_cv(name)
    t_ho = temporal_holdout(name)
    split[name] = {"random_cv": r_cv, "temporal_ho": t_ho}
    print(f"  {name:10s}: random-CV {r_cv:.3f}   temporal-holdout {t_ho:.3f}   optimism gap {r_cv - t_ho:+.3f}")''',
)

code('''# CHECK — do not edit
assert abs(split["logistic"]["random_cv"] - 0.846) < 0.02, "logistic random-CV should be ~0.846."
assert abs(split["logistic"]["temporal_ho"] - 0.758) < 0.03, "logistic temporal-holdout should be ~0.758."
for name in ("logistic", "hist_gbdt"):
    gap = split[name]["random_cv"] - split[name]["temporal_ho"]
    assert gap > 0.05, f"{name}: random CV should be clearly optimistic vs the temporal holdout."
print(f"Task 1 ok — random CV overstates deployment accuracy by "
      f"{split['logistic']['random_cv'] - split['logistic']['temporal_ho']:+.3f} (logistic). "
      f"That gap is the illusion a random split creates.")''')

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — the honest CV: TimeSeriesSplit — TODO (crucial fragment)

**Goal:** replace the shuffled `KFold` with `TimeSeriesSplit`, the expanding-window CV that always trains on the past and validates on the next block.

**Why it matters:** a single temporal holdout is one noisy number; `TimeSeriesSplit` gives a cross-validated estimate *without* leaking the future. It lands between the optimistic random CV and the harsh last-block holdout, because its earlier folds test on less-drifted periods.

**You implement:** construct `TimeSeriesSplit(n_splits=5)` and pass it as `cv=` to `cross_val_score` (accuracy). Do **not** shuffle.

**Hint boundary:** `TimeSeriesSplit(n_splits=5)`; then `cross_val_score(model(name), X, y, cv=..., scoring="accuracy").mean()`.""")

todo(
    '''# TODO — cross-validate with a time-respecting splitter
def timeseries_cv(name):
    tscv = TimeSeriesSplit(n_splits=5)
    return cross_val_score(model(name), X, y, cv=tscv, scoring="accuracy").mean()

for name in ("logistic", "hist_gbdt"):
    ts = timeseries_cv(name)
    split[name]["ts_cv"] = ts
    print(f"  {name:10s}: random-CV {split[name]['random_cv']:.3f}  "
          f"TimeSeriesSplit {ts:.3f}  temporal-holdout {split[name]['temporal_ho']:.3f}")''',
    '''# TODO — cross-validate with a time-respecting splitter
def timeseries_cv(name):
    tscv = ____                          # TimeSeriesSplit with 5 folds
    return cross_val_score(model(name), X, y, cv=____, scoring="accuracy").mean()

for name in ("logistic", "hist_gbdt"):
    ts = timeseries_cv(name)
    split[name]["ts_cv"] = ts
    print(f"  {name:10s}: random-CV {split[name]['random_cv']:.3f}  "
          f"TimeSeriesSplit {ts:.3f}  temporal-holdout {split[name]['temporal_ho']:.3f}")''',
)

code('''# CHECK — do not edit
# TimeSeriesSplit must never let a fold train on rows that come AFTER its test rows.
folds = list(TimeSeriesSplit(n_splits=5).split(X))
assert all(tr.max() < te.min() for tr, te in folds), "Every fold must train strictly before it tests."
for name in ("logistic", "hist_gbdt"):
    lo, ts, hi = split[name]["temporal_ho"], split[name]["ts_cv"], split[name]["random_cv"]
    assert lo < ts < hi + 1e-6, f"{name}: TimeSeriesSplit should sit between the temporal holdout and the random CV."
print(f"Task 2 ok — ordering holds: temporal {split['logistic']['temporal_ho']:.3f} < "
      f"TimeSeriesSplit {split['logistic']['ts_cv']:.3f} < random {split['logistic']['random_cv']:.3f}. "
      f"Only the random CV leaks the future.")''')

# ---------------------------------------------------------------- Task 3
md(r"""## Task 3 — make the drift visible — TODO (crucial fragment)

**Goal:** show *why* the temporal split is harder — the feature→label rule drifts. Split the time-ordered rows into 5 equal buckets and, in each, measure the correlation of `x0` and `x1` with the label, plus the prevalence.

**Why it matters:** if `corr(x0,y)` fades while `corr(x1,y)` rises but prevalence stays flat, the leak is **concept drift** (the rule moved), not a shifting class balance. Naming which kind of drift you have is the diagnostic skill.

**You implement:** the per-bucket correlation. For a bucket slice `s:e`, the correlation of feature column `j` with the label is `np.corrcoef(X[s:e, j], y[s:e])[0, 1]`.

**Hint boundary:** `np.corrcoef(a, b)` returns a 2×2 matrix; the cross-correlation is entry `[0, 1]`. Compute it for `j = 0` and `j = 1`.""")

todo(
    '''# TODO — per-bucket drift diagnostic
nb = 5
edges = np.linspace(0, len(X), nb + 1).astype(int)
corr_x0, corr_x1, prev = [], [], []
for b in range(nb):
    s, e = edges[b], edges[b + 1]
    corr_x0.append(np.corrcoef(X[s:e, 0], y[s:e])[0, 1])
    corr_x1.append(np.corrcoef(X[s:e, 1], y[s:e])[0, 1])
    prev.append(y[s:e].mean())
    print(f"  bucket {b} [{t[s]:.2f},{t[e-1]:.2f}]: corr(x0,y) {corr_x0[-1]:+.2f}   corr(x1,y) {corr_x1[-1]:+.2f}   prevalence {prev[-1]:.3f}")''',
    '''# TODO — per-bucket drift diagnostic
nb = 5
edges = np.linspace(0, len(X), nb + 1).astype(int)
corr_x0, corr_x1, prev = [], [], []
for b in range(nb):
    s, e = edges[b], edges[b + 1]
    corr_x0.append(____)                 # corr of column 0 with the label on this bucket
    corr_x1.append(____)                 # corr of column 1 with the label on this bucket
    prev.append(y[s:e].mean())
    print(f"  bucket {b} [{t[s]:.2f},{t[e-1]:.2f}]: corr(x0,y) {corr_x0[-1]:+.2f}   corr(x1,y) {corr_x1[-1]:+.2f}   prevalence {prev[-1]:.3f}")''',
)

code('''# CHECK — do not edit
assert corr_x0[0] - corr_x0[-1] > 0.4, "corr(x0, y) should FADE strongly from the first bucket to the last."
assert corr_x1[-1] - corr_x1[0] > 0.4, "corr(x1, y) should RISE strongly from the first bucket to the last."
assert max(prev) - min(prev) < 0.05, "Prevalence should stay ~flat — this is concept drift, not a class-balance shift."
print(f"Task 3 ok — corr(x0,y) {corr_x0[0]:+.2f}->{corr_x0[-1]:+.2f}, corr(x1,y) {corr_x1[0]:+.2f}->{corr_x1[-1]:+.2f}, "
      f"prevalence flat ({min(prev):.2f}-{max(prev):.2f}). The RULE drifted, not the base rate.")''')

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why does the random CV report a higher number than the temporal holdout, and which one would you quote to a stakeholder?""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 021 ===")
for name in ("logistic", "hist_gbdt"):
    d = split[name]
    print(f"{name:10s}: random-CV {d['random_cv']:.3f}  TimeSeriesSplit {d['ts_cv']:.3f}  "
          f"temporal-HO {d['temporal_ho']:.3f}  optimism gap {d['random_cv'] - d['temporal_ho']:+.3f}")
print(f"drift: corr(x0,y) {corr_x0[0]:+.2f}->{corr_x0[-1]:+.2f}, corr(x1,y) {corr_x1[0]:+.2f}->{corr_x1[-1]:+.2f}, "
      f"prevalence flat")
print()
print("takeaway:",
      "the random CV is optimistic because its shuffled folds put deployment-period rows into training, "
      "so under concept drift the model learns the current rule it is then scored on; the temporal holdout "
      "forbids that peek (train on the past, test on the future), so it is the honest deployment estimate and "
      "the number I would quote to a stakeholder.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 021 ===")
for name in ("logistic", "hist_gbdt"):
    d = split[name]
    print(f"{name:10s}: random-CV {d['random_cv']:.3f}  TimeSeriesSplit {d['ts_cv']:.3f}  "
          f"temporal-HO {d['temporal_ho']:.3f}  optimism gap {d['random_cv'] - d['temporal_ho']:+.3f}")
print(f"drift: corr(x0,y) {corr_x0[0]:+.2f}->{corr_x0[-1]:+.2f}, corr(x1,y) {corr_x1[0]:+.2f}->{corr_x1[-1]:+.2f}, "
      f"prevalence flat")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md(r"""## Stretch (optional, ungraded) — push the mechanism

1. **No drift, no gap.** Regenerate the stream with `drift=0.0` (the rule never rotates). The random and temporal numbers should now nearly coincide — proving the gap came from *drift*, not from the split mechanic itself.
2. **Covariate shift too.** Add a slow trend to a feature over time (e.g. `X[:, 3] += 2 * t`) and see whether it widens the gap — a second, separate kind of drift.
3. **A time proxy.** Shuffle the rows, then recover the temporal split by sorting on a column that increases with time. Real datasets rarely hand you a clean `t` — often an incrementing ID is your only clock.
4. **Grouped + temporal.** Give each row a repeating `entity_id` and confirm that a `GroupKFold` alone still leaks across time — you need *both* group and time discipline (callback to L004).""")

code('''# STRETCH — ungraded.
# X2, y2, t2 = make_stream(drift=0.0)   # no concept drift
# cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
# r = cross_val_score(LogisticRegression(max_iter=1000), X2, y2, cv=cv).mean()
# cut = int(len(X2) * 0.8)
# h = LogisticRegression(max_iter=1000).fit(X2[:cut], y2[:cut]).score(X2[cut:], y2[cut:])
# print("no-drift: random", round(r, 3), " temporal", round(h, 3), " gap", round(r - h, 3))''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l021-{i:02d}"
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": entry[1].splitlines(keepends=True)})
        elif kind == "todo":
            src = entry[1] if solution else entry[2]
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})
        else:
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": entry[1].splitlines(keepends=True)})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


(LABS / "0021-data-splits-in-the-wild.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0021-data-splits-in-the-wild.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0021-data-splits-in-the-wild.ipynb and solutions/0021-data-splits-in-the-wild.ipynb")
