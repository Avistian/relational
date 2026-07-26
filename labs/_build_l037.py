"""Build Lab 037 (Document a baseline package — build the reproduction gate) — student + solution.

Tier **A** — the learner's OWN prior work again: the ReAction L&D submission at `~/Projects/homework`.
A seeded **Tier-C synthetic stand-in** of the same shape is generated when that directory is absent, so
the notebook still runs on Colab.

Implementation scope (standard #18): the lesson's primary reading is Pineau et al. 2021 (JMLR 22(164)) and
its ML Reproducibility Checklist, which is a *protocol* rather than an architecture. The faithful
implementation of a protocol is the machinery that enforces it, so the crucial fragments are:
  * Task 1 — `content_hash()` + `build_manifest()`: the checklist's "computing infrastructure",
    "exact number of evaluation runs" and "all hyper-parameters used" items, emitted by the run itself.
  * Task 2 — `fingerprint()` + `probe()`: one perturbation at a time against a bitwise reference,
    reproducing the lesson's three findings (threads inert, model seed inert, dtype moves 258 predictions)
    on the learner's own data.
  * Task 3 — `ece()`, a named-estimator registry, `noise_floor()` and `assert_reproduces()`: the checklist's
    "clear definition of the specific measure or statistics used" item, made unambiguous by construction.

The lab uses a CHEAPER model than the submission (120 trees, a 40-column feature sample) so a probe is a few
seconds; every comparison is internally valid because all arms use the same reduced config. The lesson quotes
the full 400-tree measurement.

Run: .venv/bin/python labs/_build_l037.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0037-document-a-baseline-package.ipynb \
    labs/solutions/0037-document-a-baseline-package.ipynb
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


SETUP = r'''# PROVIDED — load the pipeline you are packaging. Just run.
import warnings, os, sys, json, hashlib, time, platform
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

# relkit ships the boilerplate half of a manifest (git state, installed versions, host).
# The interesting half is what you write below.
for _c in [Path.cwd(), *Path.cwd().parents]:
    for _cand in (_c, _c / "labs"):
        if (_cand / "relkit" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break
from relkit.repro import git_state, env_versions, host_info

HOMEWORK = Path(os.environ.get("HOMEWORK_DIR", Path.home() / "Projects" / "homework"))
CLASSES = ("completed_effective", "completed_ineffective", "declined", "dropped_out", "partial")
SEED = 0


def load_real(root):
    d = root / "data"
    persons = pd.read_parquet(d / "persons.parquet")
    situations = pd.read_parquet(d / "situations.parquet")
    responses = pd.read_parquet(d / "responses.parquet")
    df = situations.merge(persons, on="person_id", how="left", validate="m:1")
    return df.merge(responses[["situation_id", "response_category"]], on="situation_id",
                    how="left", validate="1:1")


def load_synthetic(n_persons=4000, seed=0):
    """Tier-C stand-in: same shape (repeated measures, ~5% labelled, 5 classes)."""
    rng = np.random.default_rng(seed)
    n_sit = rng.integers(1, 8, size=n_persons)
    person_id = np.repeat(np.arange(n_persons), n_sit)
    n = len(person_id)
    p_age, p_burn, p_eng = (rng.normal(42, 11, n_persons), rng.uniform(0, 1, n_persons),
                            rng.normal(0, 1, n_persons))
    df = pd.DataFrame({
        "situation_id": np.arange(n), "person_id": person_id,
        "age": p_age[person_id], "burnout_composite": p_burn[person_id],
        "engagement_mean": p_eng[person_id], "workload_index": rng.uniform(0, 1, n),
        "manager_support_score": rng.normal(0, 1, n),
        "context_domain": rng.choice(["formal_training", "on_the_job", "external_event"], n,
                                     p=[0.75, 0.18, 0.07]),
        "region": rng.choice(["DE", "FR", "PL", "ES", "NL"], n),
    })
    lin = (0.9 * df["burnout_composite"] + 0.5 * df["engagement_mean"]
           + 0.4 * df["workload_index"] + rng.normal(0, 1.6, n))
    df["response_category"] = pd.qcut(lin, 5, labels=list(CLASSES)).astype(object)
    df.loc[rng.random(n) > 0.047, "response_category"] = np.nan
    df.loc[rng.random(n) < 0.30, "burnout_composite"] = np.nan
    df.loc[rng.random(n) < 0.07, "age"] = np.nan
    return df


if (HOMEWORK / "data" / "situations.parquet").exists():
    df = load_real(HOMEWORK)
    SOURCE, DATA_ROOT = "REAL (your own submission)", HOMEWORK / "data"
else:
    df = load_synthetic()
    SOURCE, DATA_ROOT = "SYNTHETIC stand-in (homework/ not found — structure is the same)", None

ID_COLS = {"situation_id", "person_id"}


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


NUMERIC = [c for c in df.columns if _is_num(df[c]) and c not in ID_COLS][:40]
CATEGORICAL = [c for c in df.columns
               if not _is_num(df[c]) and c not in ID_COLS | {"response_category"}]

mask = df["response_category"].notna().to_numpy()
y = df.loc[mask, "response_category"].astype(str)
groups = df.loc[mask, "person_id"].to_numpy()
y_idx = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

print(f"source          : {SOURCE}")
print(f"rows            : {len(df):,} ({mask.sum():,} labelled = {100*mask.mean():.2f}%)")
print(f"persons         : {pd.Series(groups).nunique():,} in the labelled set")
print(f"features        : {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical")'''

ENCODE = r'''# PROVIDED — the encoder, the outer person-grouped CV, and the BASELINE CONFIG.
# The config dict is the point: every value below is a knob that could, in principle,
# move the reported number. Task 2 finds out which ones actually do.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

CONFIG = {
    "data":   {"dtype": "float32"},
    "split":  {"kind": "stratified_group_kfold", "n_splits": 5, "group": "person_id", "seed": SEED},
    "model":  {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 31,
               "min_child_samples": 50, "reg_lambda": 1.0, "seed": SEED, "n_jobs": 4},
    "report": {"metric": "log_loss", "estimator": "mean_over_outer_folds", "tol": 0.0},
}

encoder = ColumnTransformer(
    [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
     ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
      CATEGORICAL)],
    remainder="drop",
)
X64 = np.asarray(encoder.fit_transform(df), dtype=np.float64)[mask]


def matrix(dtype="float32"):
    """The design matrix at the requested precision — one of the knobs under test."""
    return X64.astype(np.float32) if dtype == "float32" else X64


def make_model(cfg=None, **over):
    c = dict((cfg or CONFIG)["model"], **over)
    return LGBMClassifier(
        objective="multiclass", n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
        num_leaves=c["num_leaves"], min_child_samples=c["min_child_samples"],
        reg_lambda=c["reg_lambda"], random_state=c["seed"], n_jobs=c["n_jobs"],
        verbose=-1, **{k: v for k, v in over.items() if k in ("deterministic", "force_row_wise")},
    )


def splitter(cfg=None):
    s = (cfg or CONFIG)["split"]
    return StratifiedGroupKFold(n_splits=s["n_splits"], shuffle=True, random_state=s["seed"])


def run_cv(cfg=None, *, dtype=None, model_over=None, permute=False):
    """One full person-grouped CV. Returns (oof_probs, per_fold_log_loss)."""
    cfg = cfg or CONFIG
    X = matrix(dtype or cfg["data"]["dtype"])
    oof = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float64)
    per_fold, rng = [], np.random.default_rng(12345)
    for tr, te in splitter(cfg).split(X, y_idx, groups=groups):
        tr_used = rng.permutation(tr) if permute else tr
        m = make_model(cfg, **(model_over or {})).fit(X[tr_used], y_idx[tr_used])
        p = m.predict_proba(X[te])
        oof[te] = p
        per_fold.append(float(log_loss(y_idx[te], p, labels=list(range(len(CLASSES))))))
    return oof, per_fold


t0 = time.time()
OOF_REF, FOLDS_REF = run_cv()
print(f"reference run: mean log-loss {np.mean(FOLDS_REF):.6f} "
      f"(fold sd {np.std(FOLDS_REF, ddof=1):.4f}) in {time.time()-t0:.1f}s")'''

T1_MD = r'''## Task 1 — the run manifest

**Goal.** Write `content_hash()` and `build_manifest()`, so that every run emits a machine-written record of
what it was: the code, the config, the data, the environment, the machine, the seeds and the results.

**Why it matters.** The Pineau checklist asks for "a description of the computing infrastructure used", "the
exact number of evaluation runs", and "specification of all hyper-parameters used to generate results". Those
are answerable from memory today and from nothing at all in six months. A manifest makes the question *was
this the same experiment?* mechanical — and it is the file `make verify` diffs.

Two hashes matter and they are not interchangeable:

- **`data_sha256`** — over the *bytes of the input files*. Catches a silent data swap, which nothing else will.
- **`config_sha256`** — over the config *serialised canonically*, i.e. with sorted keys, so that two configs
  that differ only in dict insertion order hash the same. Catches "someone changed a knob".

**Hint boundary.** For the data hash, read each file in sorted-name order and feed the bytes to one hash
object, so the digest covers the whole set rather than one file. For the config hash, serialise to JSON with
sorted keys before encoding. Both must be *stable*: hashing the same input twice must give the same digest,
today and next year.'''

T1_CODE = r'''# TODO — Task 1: the run manifest.

def content_hash(paths_or_obj):
    """SHA-256 of the *content* of the inputs, stable across runs.

    Accepts either an iterable of file paths (hash the raw bytes of each, in a
    deterministic order) or a plain Python object (serialise canonically first).
    Return the first 16 hex characters.
    """
    h = hashlib.sha256()
    if isinstance(paths_or_obj, (list, tuple)) and all(isinstance(p, Path) for p in paths_or_obj):
        for p in ____:                       # deterministic order — do NOT rely on the filesystem
            h.update(____)                   # the file's bytes
    else:
        h.update(____)                       # canonical JSON: sorted keys, then encode to bytes
    return h.hexdigest()[:16]


def build_manifest(cfg, oof, per_fold, *, seconds):
    """Everything needed to interpret this run's output, six months from now."""
    data_files = sorted(DATA_ROOT.glob("*.parquet")) if DATA_ROOT else []
    return {
        "git":     ____,                     # from relkit.repro — commit AND dirty flag
        "env":     ____,                     # from relkit.repro — resolved installed versions
        "host":    ____,                     # from relkit.repro
        "data_sha256":   content_hash(data_files) if data_files else "synthetic-stand-in",
        "config_sha256": ____,               # hash the config object itself
        "config":  cfg,
        "seeds":   {"split": ____, "model": ____},   # name them separately — they are not the same knob
        "results": {"log_loss_mean_over_folds": float(np.mean(per_fold)),
                    "log_loss_fold_sd": float(np.std(per_fold, ddof=1))},
        "n_runs":  len(per_fold),
        "wall_seconds": round(seconds, 1),
    }


t0 = time.time()
manifest = build_manifest(CONFIG, OOF_REF, FOLDS_REF, seconds=time.time() - t0)
print(json.dumps(manifest, indent=2)[:1200])'''

T1_SOL = r'''# SOLUTION — Task 1

def content_hash(paths_or_obj):
    h = hashlib.sha256()
    if isinstance(paths_or_obj, (list, tuple)) and all(isinstance(p, Path) for p in paths_or_obj):
        for p in sorted(paths_or_obj):
            h.update(p.read_bytes())
    else:
        h.update(json.dumps(paths_or_obj, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def build_manifest(cfg, oof, per_fold, *, seconds):
    data_files = sorted(DATA_ROOT.glob("*.parquet")) if DATA_ROOT else []
    return {
        "git":     git_state(Path.cwd()),
        "env":     env_versions(),
        "host":    host_info(),
        "data_sha256":   content_hash(data_files) if data_files else "synthetic-stand-in",
        "config_sha256": content_hash(cfg),
        "config":  cfg,
        "seeds":   {"split": cfg["split"]["seed"], "model": cfg["model"]["seed"]},
        "results": {"log_loss_mean_over_folds": float(np.mean(per_fold)),
                    "log_loss_fold_sd": float(np.std(per_fold, ddof=1))},
        "n_runs":  len(per_fold),
        "wall_seconds": round(seconds, 1),
    }


t0 = time.time()
manifest = build_manifest(CONFIG, OOF_REF, FOLDS_REF, seconds=time.time() - t0)
print(json.dumps(manifest, indent=2)[:1200])'''

T1_CHECK = r'''# CHECK 1 — do not edit.
import copy

required = {"git", "env", "host", "data_sha256", "config_sha256", "config", "seeds",
            "results", "n_runs", "wall_seconds"}
assert required <= set(manifest), f"manifest is missing {required - set(manifest)}"
assert manifest["git"] is not None and "dirty" in manifest["git"], (
    "record whether the working tree was dirty — a commit alone looks trustworthy and isn't")
assert "python" in manifest["env"] and "lightgbm" in manifest["env"], (
    "the env block must record RESOLVED installed versions, not a requirements file")
assert manifest["seeds"]["split"] is not None and manifest["seeds"]["model"] is not None, (
    "name the two seeds separately: L037 measured one as inert and one as load-bearing")

# stability: the same input must hash the same way twice
assert content_hash(CONFIG) == content_hash(CONFIG), "content_hash must be deterministic"
# ... and be blind to key order, so a reformat is not a false alarm
shuffled = {k: CONFIG[k] for k in reversed(list(CONFIG))}
assert content_hash(shuffled) == content_hash(CONFIG), (
    "serialise canonically (sorted keys) — dict order is not a configuration change")
# ... but sensitive to an actual change
bumped = copy.deepcopy(CONFIG); bumped["model"]["n_estimators"] += 1
assert content_hash(bumped) != content_hash(CONFIG), "a changed knob MUST change the config hash"
# ... and the data hash must not move when only the config does
m2 = build_manifest(bumped, OOF_REF, FOLDS_REF, seconds=0.0)
assert m2["data_sha256"] == manifest["data_sha256"], "the config hash and the data hash are separate axes"
assert m2["config_sha256"] != manifest["config_sha256"]
print("CHECK 1 passed — the run now describes itself.")
print(f"  config {manifest['config_sha256']}  data {manifest['data_sha256']}")
print(f"  env    lightgbm {manifest['env'].get('lightgbm')} / "
      f"scikit-learn {manifest['env'].get('scikit-learn')} / python {manifest['env']['python']}")'''

T2_MD = r'''## Task 2 — fingerprint the output, then probe one knob at a time

**Goal.** Write `fingerprint()` (a stable hash of the out-of-fold probability matrix) and `probe()` (compare
a perturbed run against the reference and report *how* it differs), then run four perturbations.

**Why it matters.** This is the experiment the lesson is built on, and the reason its conclusions are worth
anything: instead of repeating generic advice about seeds and threads, you *measure* which knobs move your
number. On the full 400-tree pipeline, eight of nine perturbations were byte-identical — including thread
count and the model seed — and the one that moved changed the predicted class for **258 of 5,587 people**
while shifting mean log-loss by only +0.00133.

A fingerprint must be sensitive to any change and insensitive to nothing else. Two traps: a non-contiguous
array's `.tobytes()` is fine but its memory layout is not what you think, and a float32 array and a float64
array holding "the same" numbers have different bytes — which here is a feature, since that is exactly the
difference you are hunting.

**Hint boundary.** Cast to a fixed dtype and force C-contiguity before hashing, so the digest depends on the
values and not on how they were produced. For `probe()`, report at minimum: whether the matrices are exactly
equal, the largest absolute probability change, how many rows changed their argmax, and the change in mean
log-loss. The hash is the alarm; those three are the diagnosis.'''

T2_CODE = r'''# TODO — Task 2: fingerprint + probe.

def fingerprint(arr):
    """Stable 16-hex-char SHA-256 of a numeric array's contents."""
    a = ____                       # fix dtype AND memory layout before hashing
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def probe(label, oof_new, folds_new, *, oof_ref=None, folds_ref=None):
    """Compare a perturbed run to the reference. Returns a dict; also prints a row."""
    oof_ref = OOF_REF if oof_ref is None else oof_ref
    folds_ref = FOLDS_REF if folds_ref is None else folds_ref
    d = np.abs(oof_new - oof_ref)
    rec = {
        "label": label,
        "sha": fingerprint(oof_new),
        "identical": bool(____),                       # exact equality of the two matrices
        "max_abs_dp": float(____),                     # largest single-probability change
        "argmax_flips": int(____),                     # rows whose predicted CLASS changed
        "d_mean_log_loss": float(____),                # change in the reported number
    }
    flag = "same" if rec["identical"] else "DIFFERS"
    print(f"  {label:<28s} {rec['sha']}  {flag:<8s} "
          f"flips={rec['argmax_flips']:<5d} max|dp|={rec['max_abs_dp']:.3e} "
          f"dLL={rec['d_mean_log_loss']:+.6f}")
    return rec


print(f"  {'perturbation':<28s} {'fingerprint':<16s}  {'verdict':<8s}")
probes = [probe("reference (rerun)", *run_cv())]
probes.append(probe("n_jobs 4 -> 1", *run_cv(model_over={"n_jobs": 1})))
probes.append(probe("model seed 0 -> 7", *run_cv(model_over={"seed": 7})))
probes.append(probe("float32 -> float64", *run_cv(dtype="float64")))'''

T2_SOL = r'''# SOLUTION — Task 2

def fingerprint(arr):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def probe(label, oof_new, folds_new, *, oof_ref=None, folds_ref=None):
    oof_ref = OOF_REF if oof_ref is None else oof_ref
    folds_ref = FOLDS_REF if folds_ref is None else folds_ref
    d = np.abs(oof_new - oof_ref)
    rec = {
        "label": label,
        "sha": fingerprint(oof_new),
        "identical": bool(np.array_equal(oof_new, oof_ref)),
        "max_abs_dp": float(d.max()),
        "argmax_flips": int((oof_new.argmax(axis=1) != oof_ref.argmax(axis=1)).sum()),
        "d_mean_log_loss": float(np.mean(folds_new) - np.mean(folds_ref)),
    }
    flag = "same" if rec["identical"] else "DIFFERS"
    print(f"  {label:<28s} {rec['sha']}  {flag:<8s} "
          f"flips={rec['argmax_flips']:<5d} max|dp|={rec['max_abs_dp']:.3e} "
          f"dLL={rec['d_mean_log_loss']:+.6f}")
    return rec


print(f"  {'perturbation':<28s} {'fingerprint':<16s}  {'verdict':<8s}")
probes = [probe("reference (rerun)", *run_cv())]
probes.append(probe("n_jobs 4 -> 1", *run_cv(model_over={"n_jobs": 1})))
probes.append(probe("model seed 0 -> 7", *run_cv(model_over={"seed": 7})))
probes.append(probe("float32 -> float64", *run_cv(dtype="float64")))'''

T2_CHECK = r'''# CHECK 2 — do not edit.
by = {p["label"]: p for p in probes}

assert fingerprint(OOF_REF) == fingerprint(OOF_REF.copy()), (
    "the fingerprint must depend on VALUES, not on the object — copy and original must agree")
assert fingerprint(OOF_REF) != fingerprint(OOF_REF + 1e-12), (
    "the fingerprint must be sensitive: a 1e-12 nudge is still a different matrix")

assert by["reference (rerun)"]["identical"], (
    "rerunning the same configuration must be bitwise identical — if not, something is unpinned")
assert by["n_jobs 4 -> 1"]["identical"], (
    "thread count changed the output. On the reference pipeline it does not; if it does here, "
    "record it in the manifest rather than assuming it away")
assert by["model seed 0 -> 7"]["identical"], (
    "the model seed should be INERT: this configuration sets no bagging_fraction, feature_fraction "
    "or extra_trees, so LightGBM never consults its RNG")

dt = by["float32 -> float64"]
assert not dt["identical"], "dropping the float32 cast should change the matrix"
assert dt["argmax_flips"] > 0, "and it should flip at least one predicted class"
assert abs(dt["d_mean_log_loss"]) < 0.02, (
    "...while barely moving the aggregate: that gap between artifact damage and number damage "
    "is the finding")
print("CHECK 2 passed — you measured which knobs matter instead of guessing.")
print(f"  inert here : thread count, model seed  (bitwise identical)")
print(f"  moves it   : the dtype cast — {dt['argmax_flips']} of {len(y_idx)} predicted classes "
      f"({100*dt['argmax_flips']/len(y_idx):.1f}%) change, "
      f"mean log-loss only {dt['d_mean_log_loss']:+.5f}")
print(f"  the cast lives in a notebook cell, not in the config. After this lab, it lives in CONFIG.")'''

T3_MD = r'''## Task 3 — name the number, find its noise floor, and gate on it

**Goal.** Implement `ece()`, register **two named estimators** of it, measure the **noise floor** of each, and
write the `assert_reproduces()` gate — then make the gate fail on purpose.

**Why it matters.** The submission reports top-label ECE as **0.0332** in the model-selection table and
**0.018** in the README and the ship-gate table, for the same model on the same predictions. Neither is
wrong; they are different estimators of the same metric, and "ECE" does not distinguish them. A package that
emits `{"metric": "ece_top", "estimator": "pooled_oof", "bins": 15, "n": 5587}` cannot make that mistake.

**Predict before you run it.** The model in *this* notebook is deliberately uncalibrated — no isotonic step —
so it is much worse calibrated than the shipped one. Will the gap between the two estimators be *wider* here
than the submission's 1.87×, or *narrower*? Commit to an answer; the check cell will tell you, and the
reasoning matters more than the number.

The noise floor is the second half. ECE is a weighted sum of **absolute** gaps over bins, so sampling noise
cannot cancel — it accumulates, and the estimate is biased upward as bins get sparser. You measure the bias
directly by building a control that is perfectly calibrated *by construction*: resample each row's label from
that row's own predicted probability vector. Its true ECE is exactly zero, so whatever your estimator reports
for it is pure bias.

**Formula.** `ECE = Σ_b (|B_b| / N) · |accuracy(B_b) − confidence(B_b)|`, where rows are placed in `B_b` by
their top-class probability, over 15 equal-width bins on [0, 1].

**Hint boundary.** For `ece()`: take the max probability per row as the confidence and whether the argmax was
right as the outcome, bin the confidences, and weight each bin's absolute gap by its share of rows — skipping
empty bins. For `noise_floor()`: draw one label per row from that row's probability vector, score it with the
same estimator, and repeat; return the mean and standard deviation. For `assert_reproduces()`: raise
`AssertionError` with a message naming the estimator, the expected value, what was observed and the tolerance
— a gate that fails without saying what moved costs you the afternoon it was supposed to save.'''

T3_CODE = r'''# TODO — Task 3: named estimators, the noise floor, and the gate.

def ece(probs, y_true_idx, n_bins=15):
    """Expected calibration error on the top-label probability, equal-width bins."""
    conf = ____                              # per-row confidence
    correct = ____                           # per-row 0/1 outcome
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += ____                        # weight x |accuracy - confidence| for this bin
    return float(total)


# A registry, so a number can never travel without the recipe that produced it.
ESTIMATORS = {
    "pooled_oof":        lambda probs, yi, folds: ____,   # one value over ALL rows at once
    "mean_over_folds":   lambda probs, yi, folds: ____,   # per-fold value, then averaged
}

fold_test_idx = [te for _, te in splitter().split(matrix(), y_idx, groups=groups)]


def noise_floor(probs, estimator, folds, reps=60, seed=0):
    """What this estimator reports for a PERFECTLY calibrated model (true ECE = 0)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        y_syn = np.array([rng.choice(len(CLASSES), p=r / r.sum()) for r in probs])
        vals.append(____)                    # score the synthetic labels with the SAME estimator
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def assert_reproduces(name, observed, expected, tol):
    """The gate. Bitwise when tol == 0; otherwise |observed - expected| <= tol."""
    delta = abs(observed - expected)
    if ____:                                 # the failing condition
        raise AssertionError(____)           # must name: estimator, expected, observed, delta, tol
    return delta


reported = {}
for name, fn in ESTIMATORS.items():
    val = fn(OOF_REF, y_idx, fold_test_idx)
    floor_mu, floor_sd = noise_floor(OOF_REF, fn, fold_test_idx)
    reported[name] = {"value": val, "floor": floor_mu, "floor_sd": floor_sd,
                      "signal": val - floor_mu}
    print(f"  ece_top / {name:<17s} = {val:.4f}   "
          f"noise floor {floor_mu:.4f} +- {floor_sd:.4f}   signal {val - floor_mu:+.4f}")

# The gate, exercised both ways.
tol = CONFIG["report"]["tol"]
expected_ll = float(np.mean(FOLDS_REF))
assert_reproduces("log_loss/mean_over_outer_folds", float(np.mean(run_cv()[1])), expected_ll, tol)
print(f"\n  gate PASSED on a faithful rerun (tol = {tol})")

try:
    assert_reproduces("log_loss/mean_over_outer_folds",
                      float(np.mean(run_cv(dtype="float64")[1])), expected_ll, tol)
    gate_caught = False
except AssertionError as e:
    gate_caught, gate_message = True, str(e)
    print(f"  gate FAILED on the float64 run, as it should:\n    {gate_message}")'''

T3_SOL = r'''# SOLUTION — Task 3

def ece(probs, y_true_idx, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


ESTIMATORS = {
    "pooled_oof":      lambda probs, yi, folds: ece(probs, yi),
    "mean_over_folds": lambda probs, yi, folds: float(
        np.mean([ece(probs[te], yi[te]) for te in folds])),
}

fold_test_idx = [te for _, te in splitter().split(matrix(), y_idx, groups=groups)]


def noise_floor(probs, estimator, folds, reps=60, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        y_syn = np.array([rng.choice(len(CLASSES), p=r / r.sum()) for r in probs])
        vals.append(estimator(probs, y_syn, folds))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def assert_reproduces(name, observed, expected, tol):
    delta = abs(observed - expected)
    if delta > tol:
        raise AssertionError(
            f"{name} did not reproduce: expected {expected:.6f}, observed {observed:.6f}, "
            f"|delta| = {delta:.6f} > tol = {tol:g}")
    return delta


reported = {}
for name, fn in ESTIMATORS.items():
    val = fn(OOF_REF, y_idx, fold_test_idx)
    floor_mu, floor_sd = noise_floor(OOF_REF, fn, fold_test_idx)
    reported[name] = {"value": val, "floor": floor_mu, "floor_sd": floor_sd,
                      "signal": val - floor_mu}
    print(f"  ece_top / {name:<17s} = {val:.4f}   "
          f"noise floor {floor_mu:.4f} +- {floor_sd:.4f}   signal {val - floor_mu:+.4f}")

tol = CONFIG["report"]["tol"]
expected_ll = float(np.mean(FOLDS_REF))
assert_reproduces("log_loss/mean_over_outer_folds", float(np.mean(run_cv()[1])), expected_ll, tol)
print(f"\n  gate PASSED on a faithful rerun (tol = {tol})")

try:
    assert_reproduces("log_loss/mean_over_outer_folds",
                      float(np.mean(run_cv(dtype="float64")[1])), expected_ll, tol)
    gate_caught = False
except AssertionError as e:
    gate_caught, gate_message = True, str(e)
    print(f"  gate FAILED on the float64 run, as it should:\n    {gate_message}")'''

T3_CHECK = r'''# CHECK 3 — do not edit.
# ece() sanity: a model that is always right and always maximally confident is perfectly calibrated.
n_toy = 200
perfect = np.zeros((n_toy, len(CLASSES))); perfect[:, 0] = 1.0
assert ece(perfect, np.zeros(n_toy, dtype=int)) == 0.0, "ece must be 0 for a perfect confident model"
assert abs(ece(perfect, np.ones(n_toy, dtype=int)) - 1.0) < 1e-9, (
    "...and 1 for a maximally confident model that is always wrong")

assert set(ESTIMATORS) == {"pooled_oof", "mean_over_folds"}, "register both estimators by name"
p, f = reported["pooled_oof"], reported["mean_over_folds"]
assert p["value"] != f["value"], (
    "the two estimators of the SAME metric on the SAME predictions must not be identical")
assert f["value"] > p["value"], (
    "per-fold ECE should be LARGER: sparser bins add absolute-valued noise that cannot cancel")
assert f["floor"] > p["floor"], (
    "and the noise floor must grow the same way, since it has the same cause")
assert f["floor"] / p["floor"] > 1.5, (
    "the floors should separate substantially — a fifth of the rows per bin is a much noisier ruler")
assert p["signal"] > f["signal"], (
    "so the pooled estimator resolves more real signal above its own floor — it is the one to gate on")

assert gate_caught, "assert_reproduces must FAIL on the float64 run; a gate that cannot fail is a comment"
for token in ("expected", "observed", "tol"):
    assert token in gate_message, (
        f"the failure message must name '{token}' — a gate that will not say what moved is a riddle")
print("CHECK 3 passed — the number now travels with its recipe, its floor and its gate.")
print(f"  pooled_oof      {p['value']:.4f}  (floor {p['floor']:.4f}) -> signal {p['signal']:+.4f}")
print(f"  mean_over_folds {f['value']:.4f}  (floor {f['floor']:.4f}) -> signal {f['signal']:+.4f}")
print(f"  values differ by {100*abs(f['value']/p['value'] - 1):.0f}%, "
      f"but the FLOORS differ by {100*(f['floor']/p['floor'] - 1):.0f}%.")
print()
print("  Read that carefully, because it is the opposite of what you were set up to expect.")
print(f"  This model is UNCALIBRATED (no isotonic step) and genuinely bad: ECE {p['value']:.3f} is")
print(f"  {p['value']/p['floor']:.0f}x its own noise floor. When the miscalibration is that large it")
print("  swamps the estimator's bias, so both recipes agree and the choice is harmless.")
print("  The submission's SHIPPED model is calibrated: ECE 0.018 pooled against a floor of")
print("  0.0149, and 0.0332 per-fold against a floor of 0.0335. There the same choice is")
print("  worth 1.87x — nearly the entire reported difference.")
print("  So: the estimator you pick matters LEAST when the number is bad, and MOST when it is")
print("  good. Naming it is cheap insurance you buy before you know which case you are in.")'''

EXIT_MD = r'''## EXIT TICKET

Prints the manifest summary, the probe table, both named estimators with their noise floors, and your
tolerance decision. Paste it to your teacher, or say *"lab done"*.'''

EXIT_TAIL = r'''print("=" * 78)
print("LAB 037 EXIT TICKET — the reproduction gate")
print("=" * 78)
print(f"pipeline           : {SOURCE}")
print(f"config / data hash : {manifest['config_sha256']} / {manifest['data_sha256']}")
print(f"env                : lightgbm {manifest['env'].get('lightgbm')}, "
      f"scikit-learn {manifest['env'].get('scikit-learn')}, python {manifest['env']['python']}")
print(f"headline           : log-loss {manifest['results']['log_loss_mean_over_folds']:.6f} "
      f"(fold sd {manifest['results']['log_loss_fold_sd']:.4f}), "
      f"OOF {fingerprint(OOF_REF)}")
print()
print("PROBES — which knobs move the number")
for pr in probes:
    print(f"  {pr['label']:<28s} {'same' if pr['identical'] else 'DIFFERS':<8s} "
          f"flips={pr['argmax_flips']:<5d} dLL={pr['d_mean_log_loss']:+.6f}")
print(f"  knobs that moved it : {knobs_that_moved}")
print()
print("ESTIMATOR OF RECORD — the same metric, two recipes")
for nm, r in reported.items():
    print(f"  ece_top/{nm:<17s} {r['value']:.4f}  floor {r['floor']:.4f}  "
          f"signal {r['signal']:+.4f}")
print(f"  estimator I would gate on : {estimator_of_record}")
print()
print(f"TOLERANCE : {tolerance_choice}")
print(f"  because : {tolerance_reason}")
print()
print(f"TAKEAWAY  : {takeaway}")
print("=" * 78)'''

EXIT_CODE = r'''# TODO — EXIT TICKET: your four decisions.
knobs_that_moved = "____"     # of the four you probed, which changed the out-of-fold matrix?
estimator_of_record = "____"  # "pooled_oof" or "mean_over_folds" — and it is not a coin toss
tolerance_choice = "____"     # the value you would put in CONFIG["report"]["tol"]
tolerance_reason = "____"     # one sentence, referring to a number you measured above
takeaway = "____"             # one sentence: what you will build into your next pipeline

''' + EXIT_TAIL

EXIT_SOL = r'''# SOLUTION — EXIT TICKET
knobs_that_moved = ("only the dtype cast — thread count and the model seed were bitwise identical, "
                    "so the model seed is inert here (no bagging or feature sampling is configured)")
estimator_of_record = ("pooled_oof — at fold size the estimator's noise floor is almost as large as "
                       "the value itself, so mean_over_folds resolves far less real signal")
tolerance_choice = "0.0 (bitwise: compare the OOF fingerprint, not the number)"
tolerance_reason = ("a faithful rerun was byte-for-byte identical, so a hash gate produces no false "
                    "alarms and needs no threshold that could be widened after a failure; the moment "
                    "something genuinely stochastic enters I would derive a tolerance from measured "
                    "drift and keep it well under the smallest difference I would act on")
takeaway = ("Emit a manifest from the run itself and gate on the output fingerprint, so a changed "
            "environment or a changed knob is loud rather than silent.")

''' + EXIT_TAIL

STRETCH = r'''# STRETCH (optional, ungraded) — write the lockfile this workspace does not have.
# requirements-labs.txt pins nothing: `lightgbm>=4.0`, `scikit-learn>=1.5`, `numpy>=1.26`.
# Those constraints admit combinations that do not work together — L037 measured one:
# lightgbm 4.5.0 with scikit-learn 1.9.0 raises TypeError, because scikit-learn deleted
# `force_all_finite` in 1.8 and LightGBM 4.5.0 still calls it. Every verified number in
# this workspace came from one particular resolution of those constraints, recorded nowhere.
import subprocess, sys
from pathlib import Path

root = Path.cwd()
while root != root.parent and not (root / "requirements-labs.txt").exists():
    root = root.parent
lock = root / "requirements-labs.lock.txt"

frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                        capture_output=True, text=True, check=False).stdout
if frozen.strip():
    header = ("# Generated by Lab 037. The EXACT environment that produced the numbers in\n"
              "# this workspace's learning records. Rebuild with:\n"
              "#     pip install -r requirements-labs.lock.txt\n"
              "# requirements-labs.txt states what we TOLERATE; this states what we RAN.\n")
    lock.write_text(header + frozen)
    print(f"wrote {lock} ({len(frozen.splitlines())} pinned packages)")
    print("\nthe four that decide the numbers:")
    for line in frozen.splitlines():
        if line.split("==")[0].lower() in {"lightgbm", "scikit-learn", "numpy", "pandas"}:
            print("  " + line)
else:
    print("pip freeze produced nothing — skipping (are you in a venv?)")

print("\nA lockfile is only half the job. The other half is `make verify`: re-run the entry")
print("point, rebuild the manifest, and diff it field by field against the committed one.")
print("A changed env with an unchanged OOF fingerprint is an upgrade you have just proven")
print("safe. An unchanged env with a changed fingerprint is a bug you introduced. And a")
print("changed data hash nobody announced is the worst news the file can carry — which is")
print("exactly why it is the one nothing else would ever tell you.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 037 — Document a baseline package: build the reproduction gate

**Lesson:** [`lessons/0037-document-a-baseline-package.html`](../lessons/0037-document-a-baseline-package.html) · **Phase / Year:** Year 1 · Q4

**Primary reading:** Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox & Larochelle, [*Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*](https://jmlr.org/papers/v22/20-303.html), JMLR 22(164), 2021 — §5 and the ML Reproducibility Checklist in Appendix Fig. 8.

**Dataset tier:** **A** — *your own data*: the ReAction L&D submission at `~/Projects/homework`. Set `HOMEWORK_DIR` to override. If the directory is absent (e.g. on Colab), a **seeded Tier-C stand-in with the same structure** is generated so every cell still runs.

**Implementation scope (standard #18):** the primary reading is a *protocol*, not an architecture, so the faithful implementation of it is the machinery that enforces it. You build three checklist items into code: "a description of the computing infrastructure used" and "all hyper-parameters used to generate results" (Task 1, the manifest), "the exact number of evaluation runs" and "a description of how experiments were run" (Task 2, the probe), and "a clear definition of the specific measure or statistics used to report results" (Task 3, the named-estimator registry). Boilerplate — git state, installed versions, host — is PROVIDED in [`relkit/repro.py`](relkit/repro.py); the load-bearing half is yours, and belongs in that module when you are done.

**Skill you are practising:** turn an audited pipeline into a package that regenerates its own headline number on command and **fails loudly when it cannot** — pinning what measurably moves the number, naming the number precisely enough to compare, and pre-registering a tolerance.

**Compute note:** this lab runs a cheaper model than the submission — **120 trees** and a 40-column feature sample, not 400 trees over 132 features — so each of the five probe runs finishes in a few seconds. Every comparison here is internally valid because all arms use the identical reduced config; the lesson quotes the full 400-tree measurements. A downscaled run is a different experiment, and saying so is part of the skill.

**Exit criteria:** EXIT TICKET prints your manifest, your probe table, both estimators with their noise floors, and your justified tolerance.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (loading, encoding, the CV harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
`pandas`, `numpy`, `scikit-learn`, `lightgbm` (all in `requirements-labs.txt`). One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Total runtime: about two minutes, most of it in Task 2's five CV runs.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — a package is a machine, not a description

[L036](../lessons/0036-revisit-your-homework-pipeline.html) asked whether your reported number was **right**.
This lab asks whether it is **stable** — whether running the pipeline again gives it back. The two are
independent: a leaky pipeline reproduces its leak perfectly, and an impeccable pipeline whose number cannot be
regenerated is not evidence of anything.

**Three rungs, and they are not the same claim.** *Repeatability* is you re-running your own setup and getting
your own answer. *Reproducibility* is someone else running **your** artifacts and getting your answer.
*Replicability* is someone else, from artifacts they wrote themselves. (ACM
[swapped the last two definitions in 2020](https://www.acm.org/publications/badging-terms) to match the wider
sciences, so when precision matters, say "same artifacts" or "independent artifacts" and skip the noun.) A
package targets the middle rung.

**Constraint ≠ pin ≠ lock.** `lightgbm>=4.0` says which versions you would *tolerate*. `lightgbm==4.6.0` says
which one you want. A lockfile records the version **and content hash of every package actually installed**,
transitive dependencies included. Only the third describes what you ran — and the stretch task writes the one
this workspace is missing.

**Bitwise beats a tolerance when you can get it.** If two runs' outputs hash identically there is nothing to
argue about, no threshold, and nothing to widen after a failure. On the reference pipeline eight of nine
perturbations *were* bitwise identical, which is why the config below sets `tol: 0.0`. Fall back to a numeric
tolerance only when something genuinely stochastic enters — and then derive it from measured drift.

**Worked micro-example (not this lab's data).** Score four rows at confidences `[0.9, 0.9, 0.6, 0.6]`, with
the first of each pair correct. Pool them into one bin per confidence level: bin `0.9` has accuracy ½ against
confidence 0.9 (gap 0.40) and bin `0.6` has accuracy ½ against 0.6 (gap 0.10), so
`ECE = ½·0.40 + ½·0.10 = 0.25`. Now split the four rows into two folds of two and average: each fold has one
bin, one right and one wrong is impossible in a fold of two drawn this way — you get gaps of 0.10 and 0.90
depending on the draw, averaging *higher* than 0.25 because the absolute value stops them cancelling. That is
the whole mechanism behind 0.0332 vs 0.018, at a scale you can check by hand.

Full write-up, the perturbation ledger and the noise-floor tables:
[Lesson 037](../lessons/0037-document-a-baseline-package.html).'''),
        md("## Setup — PROVIDED (load the pipeline you are packaging)"),
        code(SETUP),
        md("### PROVIDED — the encoder, the CV harness, and the baseline config"),
        code(ENCODE),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — write this workspace's missing lockfile

Everything above packaged *the submission*. This task turns the same instrument on the workspace you are
learning in, which currently has the weaker half of the setup: `requirements-labs.txt` states lower bounds
(`lightgbm>=4.0`, `scikit-learn>=1.5`, `numpy>=1.26`) and there is no lockfile at all. Every verified number
in every learning record here was produced by one particular resolution of those constraints, and that
resolution is written down nowhere.

**Predict first.** How many packages do you think are actually installed — how many transitive dependencies
sit behind the eight lines of `requirements-labs.txt`? Then run it.'''),
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
    with open(os.path.join(HERE, "0037-document-a-baseline-package.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0037-document-a-baseline-package.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0037-document-a-baseline-package.ipynb + solution")


if __name__ == "__main__":
    main()
