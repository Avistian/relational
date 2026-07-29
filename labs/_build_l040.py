"""Build Lab 040 (Year 1 exit exam) — student + solution notebooks.

Capstone: regenerate a disclosed XGBoost baseline on OpenML adult under the L020
fair protocol, attempt one honest LightGBM challenger, classify BEAT/TIE/FAIL
against a ±0.002 ROC-AUC noise band, and write Grinsztajn's three inductive
biases + a STAND/REVISE stance on the L039 synthesis claim.

Tier A: adult via relkit (public, regenerable). Homework packaging is a stretch
when ~/Projects/homework is present — Q4 audit findings still discipline the
noise-band rule.

Run:  python labs/_build_l040.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from _colab import bootstrap_cells  # noqa: E402

LABS = os.path.dirname(__file__)
SOL = os.path.join(LABS, "solutions")
os.makedirs(SOL, exist_ok=True)

CELLS: list = []


def md(src: str) -> None:
    CELLS.append(("md", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


def todo(sol: str, stu: str) -> None:
    CELLS.append(("todo", sol, stu))


md("""# Lab 040 — Year 1 Exit Exam: beat XGBoost or explain why not

**Lesson:** [`lessons/0040-year-1-exit-exam.html`](../lessons/0040-year-1-exit-exam.html) · **Phase / Year:** Year 1 · Q4 (exit)

**Primary reading:** Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022 ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — exit re-read of abstract + §5. Keep your L039 synthesis essay beside it.

**Dataset tier:** A — real, open. OpenML `adult` (income > 50K, ~24% positive) via `relkit`. Same L020 protocol so the reference is regenerable by anyone. Homework (`~/Projects/homework`) remains the Q4 audit artifact — stretch task below if present.

**Skill you are practising:** regenerate a disclosed XGBoost baseline under a fair fixed protocol, attempt one honest Year-1 challenger, classify BEAT / TIE / FAIL against a disclosed noise band, and write Grinsztajn's three inductive biases with evidence of record — ending in an explicit STAND or REVISE on the L039 claim.

**Exit criteria:** EXIT TICKET prints (1) regenerable reference + challenger ROC-AUC, (2) the fork, (3) three bias sentences, (4) STAND/REVISE. A TIE plus a solid explanation is a full pass.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn**, **xgboost**, **lightgbm**. Dataset cached by `labs/data/fetch_datasets.py`. No local install? Open from [`notebooks.html`](../notebooks.html) — **Open in Colab**.
""")

md(r"""## Concept recap — the exit contract

Year 1's curriculum exit has **two** deliverables:

1. **Regenerable tree baseline** — a disclosed XGBoost number anyone can reproduce (here: fixed-default XGB on `adult`, L020 protocol).
2. **Written three-bias understanding** — irregular targets, privileged orientation, junk-feature robustness — each with a verified number and a flip condition (Grinsztajn 2022 §5).

The experimental fork is **BEAT / TIE / EXPLAIN**, not "you must win":

| Fork | Rule on this lab | Pass? |
|------|------------------|-------|
| **BEAT** | challenger − reference > `NOISE` (0.002 ROC-AUC) under the fair protocol | yes, if leak audit + regenerable |
| **TIE** | \|delta\| ≤ `NOISE` | yes — modal Year-1 outcome |
| **FAIL** then **EXPLAIN** | challenger below reference − `NOISE` | yes, with the bias write-up |

L020 verified pattern to expect: ref ≈ 0.9282, tuned LGBM ≈ 0.9296 (Δ ≈ +0.0014 → **TIE**). Calling that a beat would repeat the homework's 0.0032-nat winner's curse (L036).

**Toy micro-example.** Reference 0.900, challenger 0.901, noise 0.002 → TIE. Challenger 0.910 → BEAT candidate. Challenger 0.895 → FAIL, then explain via biases.

Full write-up: [Lesson 040](../lessons/0040-year-1-exit-exam.html).
""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED — data, fixed split, preprocessing, scorer, noise band. Just run.
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
_here = Path(".").resolve()
for _p in (_here, _here.parent, _here / "labs"):
    sys.path.insert(0, str(_p))

import numpy as np
from relkit.data import load_tier_a
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RS = 0
NOISE = 0.002   # disclosed exit noise band (matches L020 CHECK tolerance)

X, y = load_tier_a("adult")
cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
num_cols = [c for c in X.columns if c not in cat_cols]
prev = float(y.mean())
print(f"adult: {X.shape[0]} rows, {X.shape[1]} cols, prevalence {prev:.3f} "
      f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")

def pre():
    return ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

def report(name, model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p >= 0.5).astype(int))
    ap = average_precision_score(yte, p)
    print(f"  {name:28s} ROC-AUC {auc:.4f}  acc {acc:.4f}  PR-AUC {ap:.4f}")
    return {"auc": auc, "acc": acc, "ap": ap}

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print(f"split: train {Xtr.shape[0]}  test {Xte.shape[0]}  "
      f"train prev {ytr.mean():.3f}  test prev {yte.mean():.3f}")
print(f"NOISE band = ±{NOISE} ROC-AUC  |  seed = {RS}")
print("setup ready")
""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — regenerate the XGBoost reference — TODO (crucial fragment)

**Goal:** build the pre-registered reference — fixed-default XGBoost *inside* a pipeline — and land in the published ROC-AUC band ≈ 0.92–0.93.

**Why it matters:** the exit's first curriculum deliverable is a **regenerable** baseline. Wrapping the booster in `pre()` keeps preprocessing per-fold-safe. Do not tune here — the hyper-parameters are disclosed so anyone can match you.

**You implement:** a `Pipeline` with `pre()` then the provided `XGBClassifier`.

**Hint boundary:** `Pipeline([("pre", pre()), ("clf", ref_clf)])`.
""")

todo(
    """# TODO — assemble the reference pipeline (preprocessing INSIDE the pipeline)
ref_clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=RS, n_jobs=4,
)
ref = Pipeline([("pre", pre()), ("clf", ref_clf)])
ref.fit(Xtr, ytr)
ref_m = report("XGB reference (fixed)", ref, Xte, yte)
""",
    """# TODO — assemble the reference pipeline (preprocessing INSIDE the pipeline)
ref_clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=RS, n_jobs=4,
)
ref = ____                    # Pipeline: ("pre", pre()) then ("clf", ref_clf)
ref.fit(Xtr, ytr)
ref_m = report("XGB reference (fixed)", ref, Xte, yte)
""",
)

code("""# CHECK — do not edit
assert list(ref.named_steps) == ["pre", "clf"], "Reference must be a Pipeline: pre -> clf."
assert 0.92 <= ref_m["auc"] <= 0.935, "Reference ROC-AUC should land in the published band ~0.92-0.93."
print(f"Task 1 ok — regenerable reference ROC-AUC {ref_m['auc']:.4f} "
      f"(L020 verified 0.9282). This is the bar.")
""")

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — one honest challenger (disclosed LightGBM) — TODO (crucial fragment)

**Goal:** fit a disclosed LightGBM configuration under the **identical** protocol — same split, same `pre()`, same metric — and score it once on the held-out test set.

**Why it matters:** an explain-why-not without an attempt is unearned; a beat without a same-protocol challenger is undefined. We use a *disclosed* config (not a secret search-until-win) so the challenger is as regenerable as the reference. L020's tuned LGBM landed at ≈ 0.9296; expect a match, not a crush.

**You implement:** a `Pipeline` with `pre()` then the provided `LGBMClassifier`.

**Hint boundary:** same `Pipeline([("pre", pre()), ("clf", chall_clf)])` pattern as Task 1.
""")

todo(
    """# TODO — assemble the challenger pipeline (same protocol as the reference)
chall_clf = LGBMClassifier(
    n_estimators=500, num_leaves=63, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=RS, n_jobs=4, verbose=-1,
)
chall = Pipeline([("pre", pre()), ("clf", chall_clf)])
chall.fit(Xtr, ytr)
chall_m = report("LGBM challenger (disclosed)", chall, Xte, yte)
delta = chall_m["auc"] - ref_m["auc"]
print(f"  delta vs reference: {delta:+.4f}   (NOISE band ±{NOISE})")
""",
    """# TODO — assemble the challenger pipeline (same protocol as the reference)
chall_clf = LGBMClassifier(
    n_estimators=500, num_leaves=63, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=RS, n_jobs=4, verbose=-1,
)
chall = ____                  # Pipeline: ("pre", pre()) then ("clf", chall_clf)
chall.fit(Xtr, ytr)
chall_m = report("LGBM challenger (disclosed)", chall, Xte, yte)
delta = chall_m["auc"] - ref_m["auc"]
print(f"  delta vs reference: {delta:+.4f}   (NOISE band ±{NOISE})")
""",
)

code("""# CHECK — do not edit
assert list(chall.named_steps) == ["pre", "clf"], "Challenger must be a Pipeline: pre -> clf."
assert chall_m["auc"] >= ref_m["auc"] - 0.01, "Challenger should be in the same ballpark as the reference."
print(f"Task 2 ok — challenger ROC-AUC {chall_m['auc']:.4f} (delta {delta:+.4f}). "
      f"Do NOT call a tiny positive a beat yet — classify in Task 3.")
""")

# ---------------------------------------------------------------- Task 3
md(r"""## Task 3 — classify the fork — TODO (crucial fragment)

**Goal:** implement `classify_verdict(ref_auc, chall_auc, noise=NOISE) → "BEAT" | "TIE" | "FAIL"`.

**Why it matters:** this is the load-bearing exit skill. Soft-selling Δ=+0.001 as a "beat" is the same defect as shipping M2a over M1 on 0.0032 nats (L036). Ties are full passes.

**Rules:**
- `chall − ref > noise` → `"BEAT"`
- `ref − chall > noise` → `"FAIL"`
- otherwise → `"TIE"`

**Hint boundary:** three comparisons against `noise`; return one of the three strings exactly.
""")

todo(
    '''# TODO — the exit fork classifier
def classify_verdict(ref_auc, chall_auc, noise=NOISE):
    d = chall_auc - ref_auc
    if d > noise:
        return "BEAT"
    if d < -noise:
        return "FAIL"
    return "TIE"

fork = classify_verdict(ref_m["auc"], chall_m["auc"], NOISE)
print(f"fork = {fork}   (delta {chall_m['auc'] - ref_m['auc']:+.4f}, noise ±{NOISE})")
''',
    '''# TODO — the exit fork classifier
def classify_verdict(ref_auc, chall_auc, noise=NOISE):
    d = chall_auc - ref_auc
    if d > noise:
        return ____     # "BEAT"
    if d < -noise:
        return ____     # "FAIL"
    return ____         # "TIE"

fork = classify_verdict(ref_m["auc"], chall_m["auc"], NOISE)
print(f"fork = {fork}   (delta {chall_m['auc'] - ref_m['auc']:+.4f}, noise ±{NOISE})")
''',
)

code("""# CHECK — do not edit
assert classify_verdict(0.90, 0.910, 0.002) == "BEAT"
assert classify_verdict(0.90, 0.901, 0.002) == "TIE"
assert classify_verdict(0.90, 0.895, 0.002) == "FAIL"
assert classify_verdict(0.9282, 0.9296, 0.002) == "TIE", "L020 pattern must classify as TIE."
assert fork in ("BEAT", "TIE", "FAIL")
print(f"Task 3 ok — classifier correct; this run's fork is {fork}.")
""")

# ---------------------------------------------------------------- Task 4
md(r"""## Task 4 — three biases + essay stance — TODO (crucial fragment)

**Goal:** fill three one-sentence bias explanations (mechanism + one evidence-of-record cue + flip) and choose `STAND` or `REVISE` for the L039 claim.

**Why it matters:** the curriculum's second exit deliverable is *written* understanding. Recognition quizzes are not enough. Use inductive-bias language — not "trees are more powerful" (M47).

**You implement:** the three strings and the stance. The CHECK only verifies non-empty structure and a legal stance word; your teacher grades the prose.

**Hint boundary:** each bias sentence should mention the mechanism and a flip. Stance is exactly `STAND` or `REVISE`.
""")

todo(
    '''# TODO — written exit criterion (three biases + essay stance)
bias_smooth = (
    "Irregular targets: trees place hard steps while nets have a smoothness prior; "
    "Gaussian-smoothing the target collapses the GBT-MLP gap (L025) — flip = smooth target."
)
bias_rotate = (
    "Privileged orientation: axis-aligned splits attend to meaningful columns; "
    "a lossless rotation collapses trees 0.987→0.747 while MLPs stay flat (L026) — flip = rotate basis."
)
bias_junk = (
    "Junk-feature robustness: gain-gated splits ignore noise columns; "
    "adding 100 pure-noise features reverses an MLP win over GBT (L027) — flip = remove junk / smooth+clean."
)

# STAND if this run agrees with the L039 working claim (trees match/beat honest single-table
# upgrades on typical flat tables; claim has boundaries; open burden stays open).
# REVISE only if the regenerable number truly changes that claim — and say what changed.
essay_stance = "STAND"

assert essay_stance in ("STAND", "REVISE")
print("bias_smooth:", bias_smooth)
print("bias_rotate:", bias_rotate)
print("bias_junk:", bias_junk)
print("essay_stance:", essay_stance)
''',
    '''# TODO — written exit criterion (three biases + essay stance)
bias_smooth = "____"   # irregular / smoothness bias + number/cue + flip
bias_rotate = "____"   # orientation / rotation bias + number/cue + flip
bias_junk = "____"     # junk-feature bias + number/cue + flip

# STAND if this run agrees with the L039 working claim; REVISE only if the number changes it.
essay_stance = "____"  # "STAND" or "REVISE"

assert essay_stance in ("STAND", "REVISE")
print("bias_smooth:", bias_smooth)
print("bias_rotate:", bias_rotate)
print("bias_junk:", bias_junk)
print("essay_stance:", essay_stance)
''',
)

code("""# CHECK — do not edit
for name, s in [("smooth", bias_smooth), ("rotate", bias_rotate), ("junk", bias_junk)]:
    assert isinstance(s, str) and len(s.strip()) >= 40, f"{name} bias sentence too short / empty."
    assert "____" not in s, f"{name} still has a blank."
assert essay_stance in ("STAND", "REVISE")
print(f"Task 4 ok — three bias sentences present; stance = {essay_stance}. "
      f"Teacher grades the prose on cold explain-back.")
""")

# ---------------------------------------------------------------- EXIT
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste to your teacher. Fill the explain line: if fork is TIE or FAIL, say *why not* in one sentence pointing at biases/exhaustion; if BEAT, say what cleared the bar and that a leak audit came first.
""")

todo(
    '''# TODO — complete the explain_line string
explain_line = (
    "Matched the regenerable XGB bar within the ±0.002 noise band; further single-table "
    "cleverness (another GBDT) is near-redundant on this typical flat table because tree "
    "inductive biases already fit irregular targets, privileged columns, and junk — "
    "consistent with the L039 claim and the L028-L033 exhaustion cascade."
)

print("=== EXIT TICKET — Lesson 040 (Year 1 exit exam) ===")
print(f"dataset: adult  prevalence {prev:.3f}  |  metric: ROC-AUC  |  split: 80/20 stratified, seed {RS}")
print(f"NOISE band: ±{NOISE}")
print(f"{'model':28s} {'ROC-AUC':>8} {'acc':>7}   delta-vs-ref")
print(f"{'XGB reference (fixed)':28s} {ref_m['auc']:>8.4f} {ref_m['acc']:>7.4f}   --")
print(f"{'LGBM challenger':28s} {chall_m['auc']:>8.4f} {chall_m['acc']:>7.4f}   {chall_m['auc']-ref_m['auc']:+.4f}")
print(f"fork: {fork}")
print()
print("bias_smooth:", bias_smooth)
print("bias_rotate:", bias_rotate)
print("bias_junk:", bias_junk)
print()
print("explain:", explain_line)
print(f"essay_stance: {essay_stance}")
print()
print("Year 1 exit deliverables: regenerable baseline = YES; three biases written = YES.")
''',
    '''# TODO — complete the explain_line string
explain_line = "____"  # one sentence: why BEAT, or why not (biases / exhaustion / noise)

print("=== EXIT TICKET — Lesson 040 (Year 1 exit exam) ===")
print(f"dataset: adult  prevalence {prev:.3f}  |  metric: ROC-AUC  |  split: 80/20 stratified, seed {RS}")
print(f"NOISE band: ±{NOISE}")
print(f"{'model':28s} {'ROC-AUC':>8} {'acc':>7}   delta-vs-ref")
print(f"{'XGB reference (fixed)':28s} {ref_m['auc']:>8.4f} {ref_m['acc']:>7.4f}   --")
print(f"{'LGBM challenger':28s} {chall_m['auc']:>8.4f} {chall_m['acc']:>7.4f}   {chall_m['auc']-ref_m['auc']:+.4f}")
print(f"fork: {fork}")
print()
print("bias_smooth:", bias_smooth)
print("bias_rotate:", bias_rotate)
print("bias_junk:", bias_junk)
print()
print("explain:", explain_line)
print(f"essay_stance: {essay_stance}")
print()
print("Year 1 exit deliverables: regenerable baseline = YES; three biases written = YES.")
''',
)

code("""# CHECK — do not edit
assert isinstance(explain_line, str) and len(explain_line.strip()) >= 40 and "____" not in explain_line
print("EXIT ok — paste the ticket above to your teacher (or say 'lab done').")
""")

md(r"""## Stretch (optional, ungraded)

1. **Homework verify.** If `~/Projects/homework` has the L037 package (`Makefile` + `baseline.yaml`), run `make verify` and paste the regenerable homework headline next to the adult exit number. State one way the L036 noise-band / ECE-estimator findings discipline how you read the adult delta.
2. **Stack probe.** Build OOF predictions for ref XGB + challenger LGBM (`cross_val_predict` on train, 3-fold). Print their correlation. If corr > 0.95, does a logistic stack move test AUC outside the noise band? (L020: corr ≈ 0.997, stack adds nothing.)
3. **Flip thought experiment.** In three sentences, describe a *different* flat dataset where you would *expect* a neural challenger to BEAT the XGB bar — naming which bias flips.
""")

code("""# STRETCH — ungraded.
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.linear_model import LogisticRegression
# cv = StratifiedKFold(3, shuffle=True, random_state=RS)
# oof_x = cross_val_predict(ref, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
# oof_l = cross_val_predict(chall, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
# print("OOF corr:", float(np.corrcoef(oof_x, oof_l)[0, 1]))
""")


def build(cells, *, solution: bool) -> dict:
    nb_cells = list(bootstrap_cells())
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l040-{i:02d}"
        if kind == "md":
            nb_cells.append({
                "cell_type": "markdown", "id": cid, "metadata": {},
                "source": entry[1].splitlines(keepends=True),
            })
        elif kind == "todo":
            src = entry[1] if solution else entry[2]
            nb_cells.append({
                "cell_type": "code", "id": cid, "metadata": {},
                "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True),
            })
        else:
            nb_cells.append({
                "cell_type": "code", "id": cid, "metadata": {},
                "execution_count": None, "outputs": [],
                "source": entry[1].splitlines(keepends=True),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


student = build(CELLS, solution=False)
solution = build(CELLS, solution=True)
with open(os.path.join(LABS, "0040-year-1-exit-exam.ipynb"), "w") as f:
    json.dump(student, f, indent=1)
    f.write("\n")
with open(os.path.join(SOL, "0040-year-1-exit-exam.ipynb"), "w") as f:
    json.dump(solution, f, indent=1)
    f.write("\n")
print("wrote labs/0040-year-1-exit-exam.ipynb and solutions/0040-year-1-exit-exam.ipynb")
