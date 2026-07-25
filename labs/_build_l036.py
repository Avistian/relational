"""Build Lab 036 (Revisit your homework pipeline — audit, fix one real defect, re-measure) — student + solution.

Tier **A** — the learner's OWN prior work: the ReAction L&D response-prediction submission at
`~/Projects/homework` (persons / situations / responses parquet; 119,498 rows, 5,587 labels, 5 classes,
person-grouped repeated measures). A seeded **Tier-C synthetic stand-in** with the same structure is
generated automatically when that directory is absent, so the notebook still runs on Colab.

Implementation scope (standard #18): L036 assigns no new paper — the artifact under study *is* the learner's
own pipeline. So the crucial fragments implement the **audit instruments** rather than an architecture:
  * Task 1 — `group_straddle_report(splits, groups)`: a reusable detector for group leakage in ANY splitter,
    applied to the outer person-grouped CV (expect: clean) and to the split that
    `CalibratedClassifierCV(cv=5)` builds internally (expect: leak — the library default has never heard of
    `person_id`).
  * Task 2 — build a person-grouped INNER calibration CV and re-measure log-loss + ECE before/after.
  * Task 3 — implement the Nadeau & Bengio (2003) corrected resampled t-test (the L023 paper method) and a
    leave-one-fold-out winner check, applied to the submission's own saved per-fold metrics, to ask whether
    the shipped model was chosen by signal or by noise.

The lab uses a CHEAPER LightGBM than the submission (120 trees, not 400) so both re-measurements finish in
about a minute; the before/after comparison is internally consistent because BOTH arms use it. The lesson
quotes the full-config (400-tree) measurement.

Run: .venv/bin/python labs/_build_l036.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0036-revisit-your-homework-pipeline.ipynb \
    labs/solutions/0036-revisit-your-homework-pipeline.ipynb
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from _colab import bootstrap_cells  # noqa: E402


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


SETUP = r'''# PROVIDED — load the pipeline under audit. Just run.
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

# Point this at your own submission. If it is missing (e.g. on Colab), a seeded synthetic
# stand-in with the SAME structure is built instead, so every cell below still runs.
HOMEWORK = Path(os.environ.get("HOMEWORK_DIR", Path.home() / "Projects" / "homework"))
CLASSES = ("completed_effective", "completed_ineffective", "declined", "dropped_out", "partial")


def load_real(root):
    """The learner's own three tables, joined to one row per situation."""
    d = root / "data"
    persons = pd.read_parquet(d / "persons.parquet")
    situations = pd.read_parquet(d / "situations.parquet")
    responses = pd.read_parquet(d / "responses.parquet")
    df = situations.merge(persons, on="person_id", how="left", validate="m:1")
    df = df.merge(responses[["situation_id", "response_category"]], on="situation_id",
                  how="left", validate="1:1")
    return df


def load_synthetic(n_persons=4000, seed=0):
    """Tier-C stand-in: same shape (repeated measures per person, ~5% labelled, 5 classes)."""
    rng = np.random.default_rng(seed)
    n_sit = rng.integers(1, 8, size=n_persons)            # situations per person
    person_id = np.repeat(np.arange(n_persons), n_sit)
    n = len(person_id)
    # person-level attributes (constant within a person — this is what makes grouping matter)
    p_age = rng.normal(42, 11, n_persons)
    p_burn = rng.uniform(0, 1, n_persons)
    p_eng = rng.normal(0, 1, n_persons)
    df = pd.DataFrame({
        "situation_id": np.arange(n),
        "person_id": person_id,
        "age": p_age[person_id],
        "burnout_composite": p_burn[person_id],
        "engagement_mean": p_eng[person_id],
        "workload_index": rng.uniform(0, 1, n),
        "manager_support_score": rng.normal(0, 1, n),
        "context_domain": rng.choice(["formal_training", "on_the_job", "external_event"], n,
                                     p=[0.75, 0.18, 0.07]),
        "region": rng.choice(["DE", "FR", "PL", "ES", "NL"], n),
    })
    # a weak, person-driven signal (so person identity genuinely carries information)
    lin = (0.9 * df["burnout_composite"] + 0.5 * df["engagement_mean"]
           + 0.4 * df["workload_index"] + rng.normal(0, 1.6, n))
    df["response_category"] = pd.qcut(lin, 5, labels=list(CLASSES)).astype(object)
    # only ~4.7% of rows are labelled, as in the real submission
    unlabelled = rng.random(n) > 0.047
    df.loc[unlabelled, "response_category"] = np.nan
    # structural missingness, so imputation is doing real work
    df.loc[rng.random(n) < 0.30, "burnout_composite"] = np.nan
    df.loc[rng.random(n) < 0.07, "age"] = np.nan
    return df


if (HOMEWORK / "data" / "situations.parquet").exists():
    df = load_real(HOMEWORK)
    SOURCE = "REAL (your own submission)"
else:
    df = load_synthetic()
    SOURCE = "SYNTHETIC stand-in (homework/ not found — structure is the same)"

ID_COLS = {"situation_id", "person_id"}


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


# Detect by dtype rather than by hard-coded name, so the real tables (Arrow-backed
# string columns) and the synthetic stand-in (plain object) both classify correctly.
NUMERIC = [c for c in df.columns if _is_num(df[c]) and c not in ID_COLS]
CATEGORICAL = [c for c in df.columns
               if not _is_num(df[c]) and c not in ID_COLS | {"response_category"}]
# Keep the audit fast: the real submission has 50 opaque ctx_feat_* columns and 24 engagement
# items; a 40-column sample is plenty to reproduce the fold mechanics under audit.
NUMERIC = NUMERIC[:40]
# Plain object dtype keeps SimpleImputer/OneHotEncoder happy on either source.
df[CATEGORICAL] = df[CATEGORICAL].astype(object)

mask = df["response_category"].notna().to_numpy()
y = df.loc[mask, "response_category"].astype(object).reset_index(drop=True)
groups = df.loc[mask, "person_id"].to_numpy(dtype=object)

print(f"source          : {SOURCE}")
print(f"rows (all)      : {len(df):,}")
print(f"rows (labelled) : {mask.sum():,}   label rate {mask.mean():.3%}")
print(f"distinct persons among labelled rows: {pd.Series(groups).nunique():,}")
vc = pd.Series(groups).value_counts()
print(f"persons with >1 labelled row       : {(vc > 1).sum():,}"
      f"  ({vc[vc > 1].sum():,} rows = {vc[vc > 1].sum() / mask.sum():.1%} of the labelled set)")
print(f"features        : {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical")
print(f"classes         : {sorted(y.unique())}")'''

ENCODE = r'''# PROVIDED — the encoder + the outer CV, matching the submission's design. Just run.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

RANDOM_STATE, N_SPLITS = 0, 5

encoder = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
], remainder="drop", sparse_threshold=0.0)

# As in the submission: the encoder is fit once, on ALL rows, before any CV.
# (Whether that matters is the stretch cell at the end — for now we hold it fixed
# so the ONE thing changing between the two arms is the calibration split.)
X_all = encoder.fit_transform(df).astype(np.float32)
X = X_all[mask]

outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
outer_splits = list(outer.split(X, y, groups=groups))
print("encoded matrix:", X_all.shape, "| labelled:", X.shape)
print("outer folds   :", [len(te) for _, te in outer_splits], "test rows each")'''

T1_MD = r'''## Task 1 — build the audit instrument: does *any* split straddle a group?

**Goal.** Write one reusable function, `group_straddle_report(splits, groups)`, that takes any iterable of
`(train_idx, test_idx)` pairs plus a group array and reports which groups appear on **both** sides of a split.
Then point it at two splitters: the submission's **outer** CV, and the split that
`CalibratedClassifierCV(cv=5)` constructs **inside** each outer training fold.

**Why it matters.** The submission's outer CV is right: `StratifiedGroupKFold(groups=person_id)` keeps every
situation of a person on one side, and the notebook even asserts it per fold. But calibration is a *second,
nested* split, and it is created by the library from the integer `cv=5` — which has never heard of
`person_id`. A grouping discipline that is enforced in the code you wrote and dropped in the code you called
is the most common way a careful pipeline still leaks. This function is the instrument that finds it, and it
works on any splitter you pass it later.

**Hint boundary.** For each split, compare the *sets* of group labels on the two sides; a group in the
intersection has straddled. Return both the per-split counts and the set of offending groups. For the inner
split, remember what `CalibratedClassifierCV` does with an integer `cv` on a classifier: it builds a
`StratifiedKFold` over the rows it was handed — which are the rows of one outer *training* fold.'''

T1_CODE = r'''# TODO — Task 1: the reusable group-straddle detector, applied to outer vs inner splits.

def group_straddle_report(splits, groups):
    """For each (train, test) split, find group labels present on BOTH sides.

    Returns dict with:
      'per_split'  : list of straddling-group counts, one per split
      'offenders'  : sorted list of every group label that straddled at least once
      'n_straddling_rows' : how many rows belong to an offending group
    """
    groups = np.asarray(groups)
    per_split, offenders = [], set()
    for tr, te in splits:
        # ____ : the group labels on each side of this split
        g_tr = ____
        g_te = ____
        # ____ : groups appearing on both sides
        both = ____
        per_split.append(len(both))
        offenders |= both
    n_rows = int(np.isin(groups, list(offenders)).sum()) if offenders else 0
    return {"per_split": per_split, "offenders": sorted(offenders), "n_straddling_rows": n_rows}


# (a) the OUTER cv the submission wrote itself
outer_report = group_straddle_report(outer_splits, groups)

# (b) the INNER cv CalibratedClassifierCV(cv=5) builds inside ONE outer training fold.
tr0, te0 = outer_splits[0]
# ____ : reproduce that internal split — a StratifiedKFold(5, shuffle=True, random_state=0)
#        over the TRAINING-fold rows only, and express its indices in terms of the ORIGINAL
#        row positions so the person labels line up.
inner_cv = ____
inner_splits_local = list(inner_cv.split(X[tr0], y.iloc[tr0]))
inner_splits_global = [(tr0[a], tr0[b]) for a, b in inner_splits_local]
inner_report = group_straddle_report(inner_splits_global, groups)

print("OUTER  (StratifiedGroupKFold, groups=person_id)")
print("  straddling persons per fold:", outer_report["per_split"])
print("INNER  (what CalibratedClassifierCV(cv=5) does, inside outer fold 0)")
print("  straddling persons per fold:", inner_report["per_split"])
print(f"  distinct persons straddling : {len(inner_report['offenders'])}")
print(f"  rows belonging to them      : {inner_report['n_straddling_rows']}")'''

T1_SOL = r'''# SOLUTION — Task 1
def group_straddle_report(splits, groups):
    groups = np.asarray(groups)
    per_split, offenders = [], set()
    for tr, te in splits:
        g_tr = set(groups[tr])
        g_te = set(groups[te])
        both = g_tr & g_te
        per_split.append(len(both))
        offenders |= both
    n_rows = int(np.isin(groups, list(offenders)).sum()) if offenders else 0
    return {"per_split": per_split, "offenders": sorted(offenders), "n_straddling_rows": n_rows}


outer_report = group_straddle_report(outer_splits, groups)

tr0, te0 = outer_splits[0]
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_splits_local = list(inner_cv.split(X[tr0], y.iloc[tr0]))
inner_splits_global = [(tr0[a], tr0[b]) for a, b in inner_splits_local]
inner_report = group_straddle_report(inner_splits_global, groups)

print("OUTER  (StratifiedGroupKFold, groups=person_id)")
print("  straddling persons per fold:", outer_report["per_split"])
print("INNER  (what CalibratedClassifierCV(cv=5) does, inside outer fold 0)")
print("  straddling persons per fold:", inner_report["per_split"])
print(f"  distinct persons straddling : {len(inner_report['offenders'])}")
print(f"  rows belonging to them      : {inner_report['n_straddling_rows']}")'''

T1_CHECK = r'''# CHECK — do not edit
assert set(outer_report) == {"per_split", "offenders", "n_straddling_rows"}, "return the 3 documented keys"
assert len(outer_report["per_split"]) == N_SPLITS, "one entry per outer split"
assert sum(outer_report["per_split"]) == 0, (
    "the OUTER person-grouped CV must be clean — if not, the detector is wrong")
assert len(inner_report["per_split"]) == 5, "the inner CalibratedClassifierCV split has 5 folds"
assert sum(inner_report["per_split"]) > 0, (
    "the INNER cv=5 split ignores person_id, so some person MUST straddle it; "
    "check you split only the training-fold rows and mapped indices back to original positions")
assert inner_report["n_straddling_rows"] >= len(inner_report["offenders"]), "row count must cover offenders"
print("CHECK 1 passed — the detector clears the outer CV and catches the library-default inner split.")
print(f"  the fix has to protect {len(inner_report['offenders'])} persons "
      f"/ {inner_report['n_straddling_rows']} rows in outer fold 0 alone.")'''

T2_MD = r'''## Task 2 — fix the one defect, then re-measure before/after

**Goal.** Give `CalibratedClassifierCV` a **person-grouped** inner split, then run the submission's outer CV
twice — once with the library default, once with your fix — and print log-loss and top-label ECE for both.

**Why it matters.** This is the whole discipline of the lesson: a finding is a *hypothesis* until you
re-measure. You already know the defect is real (Task 1 counted the straddling persons). What you do **not**
know is which direction it moves the numbers, or by how much. Predict first, then measure — and be willing to
find that the effect is inside the noise. Note especially *where* this defect does **not** reach: the outer
test fold is untouched, so the reported ECE was never inflated. What the mis-split damages is the calibrator
you ship.

**Hint boundary.** `cv` accepts an *iterable of (train, test) index pairs*, not just an integer — so you can
hand it splits you generated yourself with the groups of that training fold. Indices must be **local to the
matrix you pass to `fit`** (0..len(train)-1), not global row positions. Everything else — estimator,
`method="isotonic"`, the outer folds, the metrics — must stay identical between the two arms, or you are
measuring something else.

**About the absolute numbers.** This notebook runs a deliberately cheaper model than the submission — 120
trees and a 40-column feature sample — so both arms finish in about a minute. Expect a log-loss near **1.56**
rather than the lesson's 1.4248, and a *better*-looking ECE (a weaker model is less overconfident). The level
is not comparable to the lesson; the comparison is, because both arms use the identical model. Expect the
delta to be small **and expect its sign to be unreliable**: the full 400-tree audit measured −0.0016 log-loss
and −0.0003 ECE, and a run at this size can easily land on the other side of zero. That *is* the finding — the
defect is real, its measured cost is not distinguishable from fold noise, and it is worth fixing because it is
one line and because the exposure grows as labels arrive.'''

T2_CODE = r'''# TODO — Task 2: person-grouped inner calibration CV, then measure both arms.
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# Cheaper than the submission's 400 trees so this finishes in ~a minute. Both arms use it,
# so the comparison is internally valid. n_jobs is capped deliberately: on a 12-core box
# n_jobs=-1 cost 210 s per fit vs 9 s at n_jobs=6 (thread thrashing on ~4.5k rows).
def base_model():
    return LGBMClassifier(objective="multiclass", n_estimators=120, learning_rate=0.05,
                          num_leaves=31, min_child_samples=50, reg_lambda=1.0,
                          random_state=RANDOM_STATE, n_jobs=4, verbose=-1)


def ece_top(probs, y_idx, n_bins=15):
    """Top-label expected calibration error — the submission's headline ship-gate metric."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    acc = (pred == y_idx).astype(float)
    edges, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(total)


def run_cv(grouped_calibration):
    """The submission's outer CV; only the INNER calibration split changes."""
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    y_idx_all = y.map(cls_idx).to_numpy()
    lls, eces = [], []
    for tr, te in outer_splits:
        if grouped_calibration:
            # ____ : person-grouped inner splits over the TRAINING rows, with indices LOCAL
            #        to X[tr] (0..len(tr)-1) so CalibratedClassifierCV can use them directly.
            inner = ____
        else:
            inner = 5                      # the library default under audit
        model = CalibratedClassifierCV(estimator=base_model(), method="isotonic", cv=inner)
        model.fit(X[tr], y.iloc[tr])
        order = [list(model.classes_).index(c) for c in CLASSES]
        probs = model.predict_proba(X[te])[:, order]
        lls.append(log_loss(y_idx_all[te], probs, labels=list(range(len(CLASSES)))))
        eces.append(ece_top(probs, y_idx_all[te]))
    return {"log_loss": (float(np.mean(lls)), float(np.std(lls, ddof=1))),
            "ece_top": (float(np.mean(eces)), float(np.std(eces, ddof=1))),
            "per_fold_ece": eces}


before = run_cv(grouped_calibration=False)
after = run_cv(grouped_calibration=True)

# Verify the fix actually removed the straddling it was meant to remove.
tr0 = outer_splits[0][0]
fixed_inner_local = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                         .split(X[tr0], y.iloc[tr0], groups=groups[tr0]))
fixed_report = group_straddle_report([(tr0[a], tr0[b]) for a, b in fixed_inner_local], groups)

print(f"{'arm':<26} {'log-loss':>18} {'ECE top':>18}")
for name, r in (("before (cv=5 default)", before), ("after  (grouped inner)", after)):
    print(f"{name:<26} {r['log_loss'][0]:>9.4f} ± {r['log_loss'][1]:.4f} "
          f"{r['ece_top'][0]:>9.4f} ± {r['ece_top'][1]:.4f}")
print(f"\ndelta  log-loss {after['log_loss'][0] - before['log_loss'][0]:+.4f}"
      f"   ECE {after['ece_top'][0] - before['ece_top'][0]:+.4f}")
print(f"straddling persons in the fixed inner split: {sum(fixed_report['per_split'])}")'''

T2_SOL = r'''# SOLUTION — Task 2
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss


def base_model():
    return LGBMClassifier(objective="multiclass", n_estimators=120, learning_rate=0.05,
                          num_leaves=31, min_child_samples=50, reg_lambda=1.0,
                          random_state=RANDOM_STATE, n_jobs=4, verbose=-1)


def ece_top(probs, y_idx, n_bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    acc = (pred == y_idx).astype(float)
    edges, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(total)


def run_cv(grouped_calibration):
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    y_idx_all = y.map(cls_idx).to_numpy()
    lls, eces = [], []
    for tr, te in outer_splits:
        if grouped_calibration:
            inner = list(
                StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                .split(X[tr], y.iloc[tr], groups=groups[tr])
            )
        else:
            inner = 5
        model = CalibratedClassifierCV(estimator=base_model(), method="isotonic", cv=inner)
        model.fit(X[tr], y.iloc[tr])
        order = [list(model.classes_).index(c) for c in CLASSES]
        probs = model.predict_proba(X[te])[:, order]
        lls.append(log_loss(y_idx_all[te], probs, labels=list(range(len(CLASSES)))))
        eces.append(ece_top(probs, y_idx_all[te]))
    return {"log_loss": (float(np.mean(lls)), float(np.std(lls, ddof=1))),
            "ece_top": (float(np.mean(eces)), float(np.std(eces, ddof=1))),
            "per_fold_ece": eces}


before = run_cv(grouped_calibration=False)
after = run_cv(grouped_calibration=True)

tr0 = outer_splits[0][0]
fixed_inner_local = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                         .split(X[tr0], y.iloc[tr0], groups=groups[tr0]))
fixed_report = group_straddle_report([(tr0[a], tr0[b]) for a, b in fixed_inner_local], groups)

print(f"{'arm':<26} {'log-loss':>18} {'ECE top':>18}")
for name, r in (("before (cv=5 default)", before), ("after  (grouped inner)", after)):
    print(f"{name:<26} {r['log_loss'][0]:>9.4f} ± {r['log_loss'][1]:.4f} "
          f"{r['ece_top'][0]:>9.4f} ± {r['ece_top'][1]:.4f}")
print(f"\ndelta  log-loss {after['log_loss'][0] - before['log_loss'][0]:+.4f}"
      f"   ECE {after['ece_top'][0] - before['ece_top'][0]:+.4f}")
print(f"straddling persons in the fixed inner split: {sum(fixed_report['per_split'])}")'''

T2_CHECK = r'''# CHECK — do not edit
assert sum(fixed_report["per_split"]) == 0, (
    "the fixed inner split must have ZERO straddling persons — pass groups=groups[tr] to the inner splitter")
for name, r in (("before", before), ("after", after)):
    assert 0.0 < r["ece_top"][0] < 0.5, f"{name}: implausible ECE"
    assert 1.0 < r["log_loss"][0] < 1.61, (
        f"{name}: log-loss should beat the uniform-prior floor -log(1/5)=1.6094 but stay above 1.0")
assert len(after["per_fold_ece"]) == N_SPLITS, "ECE must be measured per outer fold"
# The point is that you MEASURED the direction rather than assumed it: both arms exist and differ.
assert before["ece_top"][0] != after["ece_top"][0], "the two arms should not be byte-identical"
d_ece = after["ece_top"][0] - before["ece_top"][0]
noise = max(before["ece_top"][1], after["ece_top"][1])
print("CHECK 2 passed — the fix removes the straddling, and you have a measured before/after.")
print(f"  ECE moved {d_ece:+.4f}; one fold's ECE std is {noise:.4f} "
      f"({'INSIDE the noise band' if abs(d_ece) < noise else 'larger than fold noise'}).")
print("  Note which number did NOT move for a leakage fix: the outer test folds were never")
print("  contaminated, so this defect degraded the shipped calibrator, not the reported metric.")'''

T3_MD = r'''## Task 3 — audit the *decision*: was the shipped model chosen by signal or by noise?

**Goal.** Implement the **Nadeau & Bengio (2003) corrected resampled t-test** — the L023 paper method — and a
leave-one-fold-out winner check, then apply both to the submission's own saved per-fold log-losses.

**Why it matters.** Not every audit finding is a leak. The submission shipped **M2a** over **M1** on a mean
log-loss gap of about 0.003 nats and reported a paired `ttest_rel`. But the folds share training data, so they
are *not* independent, and the naive test's variance is too small: Nadeau & Bengio inflate it by
`1/n + n_test/n_train`. Two questions follow. Does the report's one "significant" claim survive the
correction? And is the winner stable — if you drop a single fold, does the deployed artifact change? A model
selected by noise is a defect that leaves every metric intact and still hands you the wrong artifact.

**Hint boundary.** For paired per-fold differences `d`, the naive statistic is
`mean(d) / (sd(d)/sqrt(n))`; the corrected one replaces the denominator with
`sqrt((1/n + n_test/n_train) * var(d))`, keeping `n-1` degrees of freedom. Use `scipy.stats.t.sf` for a
two-sided p-value. For the stability check, recompute each model's mean log-loss with one fold removed and
see which model is the argmin.'''

T3_CODE = r'''# TODO — Task 3: corrected resampled t-test + leave-one-fold-out winner.
from scipy import stats

ART = HOMEWORK / "artifacts"
if (ART / "cv_folds_M1_lgbm_iso.csv").exists():
    per_fold = {name: pd.read_csv(ART / f"cv_folds_{fn}.csv")["log_loss"].to_numpy()
                for name, fn in [("M0", "M0_logreg"), ("M1", "M1_lgbm_iso"),
                                 ("M2a", "M2a_lgbm_km"), ("M2b", "M2b_lgbm_prop"),
                                 ("M2c", "M2c_lgbm_km_prop")]}
    n_test_mean = float(pd.read_csv(ART / "cv_folds_M1_lgbm_iso.csv")["n_test"].mean())
    N_LABELLED = int(mask.sum())
else:
    # Stand-in: the five per-fold vectors the submission recorded (report.md §3.1/§3.2).
    per_fold = {
        "M0":  np.array([1.4888, 1.4978, 1.4626, 1.4690, 1.4826]),
        "M1":  np.array([1.4138, 1.4946, 1.4018, 1.4073, 1.4065]),
        "M2a": np.array([1.4254, 1.4679, 1.4065, 1.4051, 1.4028]),
        "M2b": np.array([1.4166, 1.4926, 1.4020, 1.4089, 1.4067]),
        "M2c": np.array([1.4222, 1.4962, 1.4018, 1.4076, 1.4042]),
    }
    n_test_mean, N_LABELLED = 1117.0, 5587
n_folds = len(per_fold["M1"])
n_train_mean = N_LABELLED - n_test_mean


def paired_tests(a, b, n_test, n_train):
    """Naive paired t-test vs Nadeau & Bengio corrected resampled t-test on d = a - b."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean_d, sd = d.mean(), d.std(ddof=1)
    # ____ : the naive statistic — treats the n fold differences as independent
    t_naive = ____
    # ____ : the corrected statistic — inflate the variance by (1/n + n_test/n_train)
    factor = ____
    t_corr = ____
    return {
        "mean_d": float(mean_d), "sd_d": float(sd), "factor": float(factor),
        "t_naive": float(t_naive), "p_naive": float(2 * stats.t.sf(abs(t_naive), df=n - 1)),
        "t_corr": float(t_corr), "p_corr": float(2 * stats.t.sf(abs(t_corr), df=n - 1)),
        "folds_favouring_a": int((d < 0).sum()),
    }


print(f"correction factor 1/n + n_test/n_train = "
      f"{1 / n_folds:.3f} + {n_test_mean / n_train_mean:.3f} = "
      f"{1 / n_folds + n_test_mean / n_train_mean:.3f}   (naive uses 1/n = {1 / n_folds:.3f})\n")
print(f"{'comparison':<12} {'mean Δ':>9} {'p naive':>9} {'p corrected':>12}  folds favouring first")
results = {}
for a, b in [("M1", "M0"), ("M2a", "M1")]:
    r = paired_tests(per_fold[a], per_fold[b], n_test_mean, n_train_mean)
    results[f"{a}-{b}"] = r
    print(f"{a + ' vs ' + b:<12} {r['mean_d']:>+9.4f} {r['p_naive']:>9.4f} {r['p_corr']:>12.4f}"
          f"        {r['folds_favouring_a']}/{n_folds}")

# ____ : leave-one-fold-out winner — for each dropped fold, which model has the lowest
#        mean log-loss over the remaining folds?
loo_winners = ____

print(f"\nshipped winner on all {n_folds} folds: "
      f"{min(per_fold, key=lambda m: per_fold[m].mean())}")
print("leave-one-fold-out winners:", loo_winners)
print(f"the winner is {'STABLE' if len(set(loo_winners)) == 1 else 'UNSTABLE'} "
      f"under dropping a single fold")'''

T3_SOL = r'''# SOLUTION — Task 3
from scipy import stats

ART = HOMEWORK / "artifacts"
if (ART / "cv_folds_M1_lgbm_iso.csv").exists():
    per_fold = {name: pd.read_csv(ART / f"cv_folds_{fn}.csv")["log_loss"].to_numpy()
                for name, fn in [("M0", "M0_logreg"), ("M1", "M1_lgbm_iso"),
                                 ("M2a", "M2a_lgbm_km"), ("M2b", "M2b_lgbm_prop"),
                                 ("M2c", "M2c_lgbm_km_prop")]}
    n_test_mean = float(pd.read_csv(ART / "cv_folds_M1_lgbm_iso.csv")["n_test"].mean())
    N_LABELLED = int(mask.sum())
else:
    per_fold = {
        "M0":  np.array([1.4888, 1.4978, 1.4626, 1.4690, 1.4826]),
        "M1":  np.array([1.4138, 1.4946, 1.4018, 1.4073, 1.4065]),
        "M2a": np.array([1.4254, 1.4679, 1.4065, 1.4051, 1.4028]),
        "M2b": np.array([1.4166, 1.4926, 1.4020, 1.4089, 1.4067]),
        "M2c": np.array([1.4222, 1.4962, 1.4018, 1.4076, 1.4042]),
    }
    n_test_mean, N_LABELLED = 1117.0, 5587
n_folds = len(per_fold["M1"])
n_train_mean = N_LABELLED - n_test_mean


def paired_tests(a, b, n_test, n_train):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean_d, sd = d.mean(), d.std(ddof=1)
    t_naive = mean_d / (sd / np.sqrt(n))
    factor = 1.0 / n + n_test / n_train
    t_corr = mean_d / np.sqrt(factor * sd ** 2)
    return {
        "mean_d": float(mean_d), "sd_d": float(sd), "factor": float(factor),
        "t_naive": float(t_naive), "p_naive": float(2 * stats.t.sf(abs(t_naive), df=n - 1)),
        "t_corr": float(t_corr), "p_corr": float(2 * stats.t.sf(abs(t_corr), df=n - 1)),
        "folds_favouring_a": int((d < 0).sum()),
    }


print(f"correction factor 1/n + n_test/n_train = "
      f"{1 / n_folds:.3f} + {n_test_mean / n_train_mean:.3f} = "
      f"{1 / n_folds + n_test_mean / n_train_mean:.3f}   (naive uses 1/n = {1 / n_folds:.3f})\n")
print(f"{'comparison':<12} {'mean Δ':>9} {'p naive':>9} {'p corrected':>12}  folds favouring first")
results = {}
for a, b in [("M1", "M0"), ("M2a", "M1")]:
    r = paired_tests(per_fold[a], per_fold[b], n_test_mean, n_train_mean)
    results[f"{a}-{b}"] = r
    print(f"{a + ' vs ' + b:<12} {r['mean_d']:>+9.4f} {r['p_naive']:>9.4f} {r['p_corr']:>12.4f}"
          f"        {r['folds_favouring_a']}/{n_folds}")

loo_winners = [
    min(per_fold, key=lambda m: np.delete(per_fold[m], drop).mean())
    for drop in range(n_folds)
]

print(f"\nshipped winner on all {n_folds} folds: "
      f"{min(per_fold, key=lambda m: per_fold[m].mean())}")
print("leave-one-fold-out winners:", loo_winners)
print(f"the winner is {'STABLE' if len(set(loo_winners)) == 1 else 'UNSTABLE'} "
      f"under dropping a single fold")'''

T3_CHECK = r'''# CHECK — do not edit
r10 = results["M1-M0"]
assert abs(r10["factor"] - (1 / n_folds + n_test_mean / n_train_mean)) < 1e-9, \
    "the correction factor must be 1/n + n_test/n_train"
assert r10["factor"] > 1 / n_folds, "the corrected variance must be LARGER than the naive one"
for key, r in results.items():
    assert r["p_corr"] > r["p_naive"], (
        f"{key}: inflating the variance can only make the p-value larger — check the denominator")
assert r10["p_naive"] < 0.05, "M1 vs M0 is the report's 'significant' claim under the naive test"
assert r10["p_corr"] > 0.05, (
    "and it should NOT survive the Nadeau-Bengio correction — that is the finding")
assert len(loo_winners) == n_folds, "one leave-one-fold-out winner per dropped fold"
assert len(set(loo_winners)) > 1, (
    "the shipped winner should flip when a single fold is dropped — that is the instability")
print("CHECK 3 passed — you reproduced both decision findings.")
print(f"  M1 vs M0: p {r10['p_naive']:.4f} (naive) -> {r10['p_corr']:.4f} (corrected) — "
      f"the report's only significance claim does not survive.")
print(f"  the shipped winner changes to {[w for w in loo_winners if w != loo_winners[0]] or '-'} "
      f"when one fold is dropped: selection was not stable.")'''

EXIT_MD = r'''## EXIT TICKET

Prints the audit's three findings, each with what it actually changes — the reported number, the shipped
artifact, or the decision — plus your before/after measurement. Paste it to your teacher, or say *"lab done"*.'''

EXIT_CODE = r'''# TODO — EXIT TICKET: fill in your one-line triage for each finding.
# Classify each by CONSEQUENCE: "inflates the reported number", "degrades the shipped
# artifact", "changes the decision", or "can only be declared".
finding_1_class = "____"   # ungrouped inner calibration split (Tasks 1-2)
finding_2_class = "____"   # winner chosen on a non-significant, unstable margin (Task 3)
takeaway = "____"          # one sentence: what you will change in your next pipeline

print("=" * 78)
print("LAB 036 EXIT TICKET — audit of my own pipeline")
print("=" * 78)
print(f"pipeline under audit : {SOURCE}")
print(f"labelled rows        : {mask.sum():,} across {pd.Series(groups).nunique():,} persons")
print()
print("FINDING 1 — the inner calibration split ignores person_id")
print(f"  outer CV straddling persons      : {sum(outer_report['per_split'])} (clean)")
print(f"  inner cv=5 straddling persons    : {len(inner_report['offenders'])} "
      f"({inner_report['n_straddling_rows']} rows) in outer fold 0")
print(f"  before : log-loss {before['log_loss'][0]:.4f}  ECE {before['ece_top'][0]:.4f}")
print(f"  after  : log-loss {after['log_loss'][0]:.4f}  ECE {after['ece_top'][0]:.4f}")
print(f"  delta  : log-loss {after['log_loss'][0] - before['log_loss'][0]:+.4f}  "
      f"ECE {after['ece_top'][0] - before['ece_top'][0]:+.4f}")
print(f"  consequence class  : {finding_1_class}")
print()
print("FINDING 2 — the shipped winner was selected on noise")
print(f"  M2a vs M1 mean delta : {results['M2a-M1']['mean_d']:+.4f} nats "
      f"({results['M2a-M1']['folds_favouring_a']}/{n_folds} folds favour M2a)")
print(f"  p naive {results['M2a-M1']['p_naive']:.3f} -> p corrected {results['M2a-M1']['p_corr']:.3f}")
print(f"  M1 vs M0 p naive {results['M1-M0']['p_naive']:.4f} -> "
      f"corrected {results['M1-M0']['p_corr']:.4f}")
print(f"  leave-one-fold-out winners : {loo_winners}")
print(f"  consequence class  : {finding_2_class}")
print()
print(f"TAKEAWAY: {takeaway}")
print("=" * 78)'''

EXIT_SOL = r'''# SOLUTION — EXIT TICKET
finding_1_class = ("degrades the shipped artifact — the outer test folds were never touched, "
                   "so the reported ECE stayed honest; the mis-fit calibrator is what shipped")
finding_2_class = ("changes the decision — every metric is correctly measured, but the argmin over "
                   "5 correlated folds picked a model on a margin smaller than fold noise")
takeaway = ("Audit the splits I did not write: library defaults like CalibratedClassifierCV(cv=5) "
            "do not inherit my grouping, and a winner chosen inside the noise band is not a winner.")

print("=" * 78)
print("LAB 036 EXIT TICKET — audit of my own pipeline")
print("=" * 78)
print(f"pipeline under audit : {SOURCE}")
print(f"labelled rows        : {mask.sum():,} across {pd.Series(groups).nunique():,} persons")
print()
print("FINDING 1 — the inner calibration split ignores person_id")
print(f"  outer CV straddling persons      : {sum(outer_report['per_split'])} (clean)")
print(f"  inner cv=5 straddling persons    : {len(inner_report['offenders'])} "
      f"({inner_report['n_straddling_rows']} rows) in outer fold 0")
print(f"  before : log-loss {before['log_loss'][0]:.4f}  ECE {before['ece_top'][0]:.4f}")
print(f"  after  : log-loss {after['log_loss'][0]:.4f}  ECE {after['ece_top'][0]:.4f}")
print(f"  delta  : log-loss {after['log_loss'][0] - before['log_loss'][0]:+.4f}  "
      f"ECE {after['ece_top'][0] - before['ece_top'][0]:+.4f}")
print(f"  consequence class  : {finding_1_class}")
print()
print("FINDING 2 — the shipped winner was selected on noise")
print(f"  M2a vs M1 mean delta : {results['M2a-M1']['mean_d']:+.4f} nats "
      f"({results['M2a-M1']['folds_favouring_a']}/{n_folds} folds favour M2a)")
print(f"  p naive {results['M2a-M1']['p_naive']:.3f} -> p corrected {results['M2a-M1']['p_corr']:.3f}")
print(f"  M1 vs M0 p naive {results['M1-M0']['p_naive']:.4f} -> "
      f"corrected {results['M1-M0']['p_corr']:.4f}")
print(f"  leave-one-fold-out winners : {loo_winners}")
print(f"  consequence class  : {finding_2_class}")
print()
print(f"TAKEAWAY: {takeaway}")
print("=" * 78)'''

STRETCH = r'''# STRETCH (optional, ungraded) — the finding whose class ALLOWS an inflated number.
# The encoder above was fit once on ALL rows, before any CV (as in the submission). Its
# medians, scales and one-hot vocabulary therefore saw every outer test row. That is
# transductive, not leak-free: it cannot be reproduced at inference time on a row that
# does not exist yet. This is the one finding that COULD make the reported number
# optimistic — so predict the direction and size, then refit the encoder inside each
# outer fold (on training-fold PERSONS only) and find out.
from sklearn.base import clone

cls_idx = {c: i for i, c in enumerate(CLASSES)}
y_idx_all = y.map(cls_idx).to_numpy()
all_persons = df["person_id"].to_numpy()
lls = []
for tr, te in outer_splits:
    test_persons = set(groups[te])
    fit_rows = ~np.isin(all_persons, list(test_persons))     # every row of a training person
    enc = clone(encoder).fit(df.loc[fit_rows])
    Xf = enc.transform(df).astype(np.float32)[mask]
    inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                 .split(Xf[tr], y.iloc[tr], groups=groups[tr]))
    m = CalibratedClassifierCV(estimator=base_model(), method="isotonic", cv=inner)
    m.fit(Xf[tr], y.iloc[tr])
    order = [list(m.classes_).index(c) for c in CLASSES]
    lls.append(log_loss(y_idx_all[te], m.predict_proba(Xf[te])[:, order],
                        labels=list(range(len(CLASSES)))))

print(f"encoder fit on ALL rows (as shipped) : {after['log_loss'][0]:.4f}")
print(f"encoder refit inside each fold       : {np.mean(lls):.4f} ± {np.std(lls, ddof=1):.4f}")
print(f"optimism attributable to the transductive encoder: "
      f"{np.mean(lls) - after['log_loss'][0]:+.4f} nats  "
      f"(positive = the honest number is worse)")
print("\nDirection AND size are empirical. On 119k rows a median, a scale and a one-hot")
print("vocabulary barely move when a fifth of the persons are withheld, so expect a delta")
print("inside fold noise: the full 400-tree audit measured -0.0011 nats, i.e. the honest")
print("arrangement came out very slightly BETTER — what a true effect of zero looks like.")
print("That is a MEASUREMENT, not a principle. The same shortcut on a small or drifting")
print("dataset, or on a supervised transform, is how optimistic CV actually happens — and")
print("this remains worth fixing for reproducibility even when it buys no accuracy.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 036 — Revisit your homework pipeline: audit, fix one real defect, re-measure

**Lesson:** [`lessons/0036-revisit-your-homework-pipeline.html`](../lessons/0036-revisit-your-homework-pipeline.html) · **Phase / Year:** Year 1 · Q4

**Primary reading:** your own `homework/report.md` + `src/`, read as a reviewer. Method backing: Nadeau & Bengio, [*Inference for the Generalization Error*](https://doi.org/10.1023/A:1024068626366) (2003) — the corrected resampled t-test you implement in Task 3 (assigned in [L023](../lessons/0023-statistical-comparison.html)).

**Dataset tier:** **A** — *your own data*: the ReAction L&D submission at `~/Projects/homework` (119,498 situations, 5,587 labels, 5 classes, up to 5 labelled situations per person). Set `HOMEWORK_DIR` to override. If the directory is absent (e.g. on Colab), a **seeded Tier-C stand-in with the same structure** is generated so every cell still runs.

**Implementation scope (standard #18):** this lesson assigns no new architecture — the artifact under study *is* your pipeline. The crucial fragments are therefore the **audit instruments**: a reusable group-straddle detector, a person-grouped nested calibration CV, and the Nadeau–Bengio corrected resampled t-test. All three are tools you keep.

**Skill you are practising:** audit a pipeline against the Q1–Q3 leakage spine, then **triage each finding by what it actually changes** — the reported number, the shipped artifact, or the decision — and re-measure before you believe any of it.

**Exit criteria:** EXIT TICKET prints both findings with your consequence classification and your measured before/after numbers.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (loading, encoding, the outer CV); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `scipy` (all in `requirements-labs.txt`). One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Tasks 1 and 3 run in seconds; Task 2 fits two 5-fold CVs and takes about a minute.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — an audit is a triage, not a bug hunt

You have spent Q1–Q3 learning to *build* a leak-free pipeline. This lab is the other half of the skill:
reading a finished pipeline — your own — the way a reviewer would, and being precise about what each defect
actually costs.

The trap is to treat "found a leak" as a verdict. It isn't; it is a hypothesis with a consequence that has to
be named and then measured. Three consequences are worth separating:

- **It inflates the number you reported.** A statistic computed on rows that include the test fold (an imputer
  median, a scaler's mean, a one-hot vocabulary, a cluster centroid) makes the CV estimate optimistic. Fix and
  the number gets *worse* — which is the point.
- **It degrades the artifact you shipped, without touching the number.** This is the one people miss. If a
  mis-split happens *inside* the training fold, the outer test fold stays clean, so your reported metric was
  honest all along — but the model you saved was fitted under a regime it will never see again. Nothing in the
  report looks wrong; the deployed thing is just worse than it should be.
- **It changes the decision, not the measurement.** Every number can be correct and the conclusion still
  wrong: pick the argmin over several models scored on the same few folds and you will often ship a model
  whose "win" is inside the fold-to-fold noise.

**Grouped CV, restated.** When rows are not independent — several situations per person — a random split puts
the same person in train *and* test, and the model gets credit for memorising person-level attributes
(personality, engagement, wearable coverage) instead of generalising. `StratifiedGroupKFold(groups=person_id)`
prevents that. The subtlety this lab turns on: **your grouping obligation does not stop at the split you
wrote.** Any nested split — calibration, early stopping, feature selection, a second model that manufactures a
feature — is also a split, and library defaults do not inherit your groups.

**Worked micro-example (not this lab's data).** Six rows from three persons, `groups = [A, A, B, B, C, C]`.
`KFold(3, shuffle=False)` gives test folds `{0,1}`, `{2,3}`, `{4,5}` — each is exactly one person, clean by
luck. Shuffle it and fold 1 might be `{1,2}`: person A's second row is in test while A's first row trains.
One row of leakage, no error message. The detector in Task 1 is just the set intersection that catches this.

**Calibration, restated.** `CalibratedClassifierCV(estimator, method="isotonic", cv=5)` splits the data it is
given into 5 parts; for each, it fits the base model on 4 and fits an isotonic (monotone step) function on the
5th, mapping raw scores to probabilities. That held-out part is what makes calibration honest — so if it
contains a person the base model already trained on, the scores it calibrates against are sharper than
reality, and the monotone map you learn is the wrong shape for genuinely-unseen people.

Full write-up, the nested-fold visualisation, and the finding table:
[Lesson 036](../lessons/0036-revisit-your-homework-pipeline.html).'''),
        md("## Setup — PROVIDED (load the pipeline under audit)"),
        code(SETUP),
        md("### PROVIDED — the encoder and the outer person-grouped CV, as the submission built them"),
        code(ENCODE),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — the finding whose class *allows* an inflated number

Findings 1 and 2 left the reported metric intact by construction. This one is the only candidate that could
have inflated it: the encoder was fit once on all 119,498 rows before any CV, so its medians, scales and
one-hot vocabulary saw every test row. The submission defends this as safe because the transformer is
unsupervised, and that defence is half right — no *label* leaks. But it is still **transductive**: it uses
information that will not exist when a genuinely new row arrives, so the CV estimate is not a deployment
estimate.

**Predict first, then run it.** How much of the reported log-loss is borrowed — a lot, a little, or nothing
you can measure with five folds? The answer here is worth more than the fix.'''),
        code(STRETCH),
    ]

    nb_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for c in nb_obj["cells"]:
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(keepends=True)
    return nb_obj


def main():
    with open(os.path.join(HERE, "0036-revisit-your-homework-pipeline.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0036-revisit-your-homework-pipeline.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0036-revisit-your-homework-pipeline.ipynb + solution")


if __name__ == "__main__":
    main()
