"""Build Lab 025 (the smoothness inductive bias) — emits the blank student notebook
and a filled solution notebook. Tier-C mechanism-isolation lab (synthetic irregular
regression); mirrors the L021/L023 concept-lab structure.

Crucial fragment: implement the Gaussian TARGET smoother (Grinsztajn §5.2), then show the
GBT-vs-MLP gap closes as the target's high-frequency structure is smoothed away.

Run: .venv/bin/python labs/_build_l025.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0025-inductive-bias-smoothness.ipynb labs/solutions/0025-inductive-bias-smoothness.ipynb
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


# ---- shared PROVIDED setup ----
SETUP = r'''# PROVIDED — synthetic irregular data, models, and the fit/score harness. Just run.
import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SEED = 0
SCALES = [0.0, 0.5, 1.0, 2.0]     # target-smoothing length-scales to sweep

def make_irregular_regression(n=2000, d=8, seed=SEED):
    """An IRREGULAR (non-smooth) target: a sign/XOR flip, a hard step, a high-frequency
    sinusoid, and a smooth linear part. This is the regime where trees beat MLPs."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, d))
    y = (np.sign(X[:, 0] * X[:, 1])          # sharp sign boundary (XOR-like)
         + (X[:, 2] > 0.3).astype(float) * 1.5   # hard step
         + np.sin(4.0 * X[:, 3])                 # high-frequency wiggle
         + 0.5 * X[:, 4])                         # a smooth linear part
    return X, y

X, y = make_irregular_regression()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED)
print(f"synthetic irregular regression: n={len(X)}, d={X.shape[1]}, "
      f"train {len(Xtr)} / test {len(Xte)}")

def fit_score(ytr_target, yte_target):
    """Fit a GBT and an MLP on the (possibly smoothed) TRAIN target; return their test R^2
    against the (same-smoothing) TEST target. Trees and MLPs, head to head."""
    gbt = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr_target)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4,
                     learning_rate_init=3e-3, max_iter=1500, random_state=SEED),
    ).fit(Xtr, ytr_target)
    return r2_score(yte_target, gbt.predict(Xte)), r2_score(yte_target, mlp.predict(Xte))

print("ready — harness fit_score(ytr_target, yte_target) -> (gbt_r2, mlp_r2)")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — the Gaussian target smoother — TODO (crucial fragment)

**Goal:** implement `gaussian_smooth_target(X, y, h)` — Grinsztajn's §5.2 operation. The smoothed
target at point `i` is a **Gaussian-weighted average of every point's target**, with weights that
decay with distance in *standardized* feature space:

$$\tilde{y}_i = \frac{\sum_j w_{ij}\, y_j}{\sum_j w_{ij}}, \qquad w_{ij} = \exp\!\left(-\frac{\lVert x_i - x_j\rVert^2}{2h^2}\right)$$

As `h → 0` every point only sees itself, so `ỹ = y` (raw); as `h` grows, each point averages over
an ever-wider neighbourhood, erasing high-frequency wiggles and, in the limit, returning the constant
global mean.

**Why it matters:** this is the scalpel of the whole experiment. Smoothing the *target* (not the
model) lets us delete exactly the irregular, high-frequency structure and ask: *was that what the tree
was using?* If the tree's advantage lives there, smoothing will make it vanish.

**You implement:** inside the loop, (1) the squared Euclidean distance `d2` from point `i` to all
points (in standardized space `Xs`), and (2) the Gaussian weights `w` from `d2` and `h`.

**Hint boundary:** `d2` is a sum of squared coordinate differences along `axis=1`; `w` is
`exp(-d2 / (2*h*h))`. The weighted average `w @ y / w.sum()` is already written for you.

**Toy check (not this lab's answer):** with two points at distance 0 and a huge `h`, both weights → 1,
so each smoothed value → the mean of the two targets.'''

T1_CODE = r'''# TODO — implement the Gaussian target smoother (the crucial fragment)
def gaussian_smooth_target(X, y, h):
    if h <= 0:
        return y.copy()                        # h=0 is the raw target (no smoothing)
    Xs = StandardScaler().fit_transform(X)     # distances in standardized feature space
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        d2 = ____                              # squared distance from point i to ALL points (axis=1)
        w = ____                               # Gaussian weights: exp(-d2 / (2 h^2))
        out[i] = w @ y / w.sum()               # weighted average of targets (provided)
    return out

# sanity: h=0 is identity; a large h collapses toward the global mean
_raw = gaussian_smooth_target(Xtr, ytr, 0.0)
_big = gaussian_smooth_target(Xtr, ytr, 50.0)
print(f"h=0   -> identity? {np.allclose(_raw, ytr)}")
print(f"h=50  -> near the mean? std {_big.std():.4f} (raw std {ytr.std():.3f}), "
      f"mean {_big.mean():+.3f} (raw mean {ytr.mean():+.3f})")'''

T1_SOL = (T1_CODE
    .replace('        d2 = ____                              # squared distance from point i to ALL points (axis=1)',
             '        d2 = np.sum((Xs - Xs[i]) ** 2, axis=1) # squared distance from point i to ALL points')
    .replace('        w = ____                               # Gaussian weights: exp(-d2 / (2 h^2))',
             '        w = np.exp(-d2 / (2.0 * h * h))         # Gaussian weights'))

T1_CHECK = r'''# CHECK — do not edit
assert np.allclose(gaussian_smooth_target(Xtr, ytr, 0.0), ytr), "h=0 must be the identity (raw target)."
_big = gaussian_smooth_target(Xtr, ytr, 50.0)
assert _big.std() < 0.1 * ytr.std(), "A huge h must collapse the target toward its (near-constant) mean."
_mid = gaussian_smooth_target(Xtr, ytr, 1.0)
assert _mid.std() < ytr.std(), "Smoothing must reduce the target's variance."
print("Task 1 ok — smoother is identity at h=0, shrinks variance as h grows, "
      "and collapses to the mean at large h. That is Grinsztajn's target scalpel.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — sweep the smoothing and measure the gap — TODO

**Goal:** for each length-scale `h` in `SCALES`, smooth **both** the train and the test target the
same way, fit a GBT and an MLP (via the provided `fit_score`), and record their test R² and the gap.

**Why it matters:** this is Grinsztajn's Finding 1 in one loop. If the tree's edge is the irregular
structure, then deleting that structure (larger `h`) should shrink the GBT − MLP gap toward zero — and
you will see it happen live.

**You implement:** call your smoother on `ytr`/`yte` at the current `h`, and append the gap
`gbt_r2 − mlp_r2`.

**Hint boundary:** smooth `ytr` and `yte` with `gaussian_smooth_target(Xtr, ytr, h)` and
`gaussian_smooth_target(Xte, yte, h)`; the gap is `g - m`. `var_kept` (provided) is the fraction of the
target's variance that survives — watch the gap track it.'''

T2_CODE = r'''# TODO — smooth train + test targets at each h, fit both models, record the gap (~15s)
gbt_r2, mlp_r2, gap, var_kept = [], [], [], []
for h in SCALES:
    ytr_h = ____                               # smooth the TRAIN target at this h
    yte_h = ____                               # smooth the TEST target at this h
    g, m = fit_score(ytr_h, yte_h)
    gbt_r2.append(g); mlp_r2.append(m)
    gap.append(____)                           # the GBT - MLP gap at this h
    var_kept.append(np.var(yte_h) / np.var(yte))

print(f"{'h':>5} | {'GBT R2':>7} | {'MLP R2':>7} | {'gap':>7} | var kept")
for i, h in enumerate(SCALES):
    print(f"{h:>5.1f} | {gbt_r2[i]:>7.3f} | {mlp_r2[i]:>7.3f} | {gap[i]:>+7.3f} | {var_kept[i]:>6.2f}")'''

T2_SOL = (T2_CODE
    .replace('    ytr_h = ____                               # smooth the TRAIN target at this h',
             '    ytr_h = gaussian_smooth_target(Xtr, ytr, h) # smooth the TRAIN target at this h')
    .replace('    yte_h = ____                               # smooth the TEST target at this h',
             '    yte_h = gaussian_smooth_target(Xte, yte, h) # smooth the TEST target at this h')
    .replace('    gap.append(____)                           # the GBT - MLP gap at this h',
             '    gap.append(g - m)                          # the GBT - MLP gap at this h'))

T2_CHECK = r'''# CHECK — do not edit
gap = np.array(gap)
assert gap[0] > 0.15, "On the RAW irregular target the GBT should beat the MLP clearly (gap > 0.15)."
assert gap.min() < 0.06, "Once the target is smoothed, the gap should collapse toward zero."
assert (gap[0] - gap.min()) > 0.2, "Smoothing should erase most of the GBT's advantage."
print(f"Task 2 ok — raw gap {gap[0]:+.3f} collapses to {gap.min():+.3f} once smoothed. "
      f"The tree's whole edge was the high-frequency structure the MLP is biased against.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — read the mechanism — TODO

**Goal:** show that the gap does not fade gradually — it tracks the target's **variance**, collapsing
exactly when the high-frequency structure is destroyed.

**Why it matters:** a gap that shrinks *in lockstep with the variance you deleted* is the signature of
a mechanism, not a coincidence. That is what licenses the claim "smoothness is *the* reason", not just
"trees happened to win".

**You implement:** the index of the first `h` where less than 30% of the variance survives, and the gap
there.

**Hint boundary:** `var_kept` is a list; find the first index where `var_kept[i] < 0.30`
(`np.argmax(np.array(var_kept) < 0.30)`), then read `gap[that_index]`.'''

T3_CODE = r'''# TODO — locate the collapse and confirm it lines up with the variance loss
var_kept = np.array(var_kept)
collapse_i = ____                              # first index where var_kept < 0.30 (np.argmax(...))
gap_when_smoothed = ____                       # gap[collapse_i]

print(f"raw gap (h={SCALES[0]}):            {gap[0]:+.3f}   (variance kept {var_kept[0]:.2f})")
print(f"gap once >70% smoothed (h={SCALES[collapse_i]}): {gap_when_smoothed:+.3f}   "
      f"(variance kept {var_kept[collapse_i]:.2f})")
print(f"the advantage tracked the variance, not h itself: {'yes' if gap[0] - gap_when_smoothed > 0.2 else 'no'}")'''

T3_SOL = (T3_CODE
    .replace('collapse_i = ____                              # first index where var_kept < 0.30 (np.argmax(...))',
             'collapse_i = int(np.argmax(var_kept < 0.30))   # first index where var_kept < 0.30')
    .replace('gap_when_smoothed = ____                       # gap[collapse_i]',
             'gap_when_smoothed = gap[collapse_i]            # the gap once the jags are gone'))

T3_CHECK = r'''# CHECK — do not edit
assert var_kept[collapse_i] < 0.30, "collapse_i should point at a heavily-smoothed target."
assert abs(gap_when_smoothed) < 0.08, "Once the high-frequency variance is gone, the gap should be ~0."
print(f"Task 3 ok — the GBT led by {gap[0]:+.3f} on the raw target and by {gap_when_smoothed:+.3f} once "
      f">70% of the variance was smoothed away. The edge WAS the irregularity — that is the smoothness bias.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why does *smoothing the target function* close the GBT-vs-MLP gap,
and what does that prove about *why* trees win on typical tabular data?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 025 ===")
print(f"experiment : Gaussian-smooth an irregular target at growing length-scale; refit GBT vs MLP")
print(f"raw target : GBT {gbt_r2[0]:.3f} vs MLP {mlp_r2[0]:.3f}  (gap {gap[0]:+.3f}, variance kept {var_kept[0]:.2f})")
print(f"smoothed   : GBT {gbt_r2[collapse_i]:.3f} vs MLP {mlp_r2[collapse_i]:.3f}  "
      f"(gap {gap[collapse_i]:+.3f}, variance kept {var_kept[collapse_i]:.2f})")
print(f"verdict    : the gap collapsed as the high-frequency variance was erased")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"An MLP has a smoothness (low-frequency / spectral) inductive bias, so it over-smooths irregular '
    'targets while a piecewise-constant tree follows the jags; smoothing the target deletes exactly that '
    'high-frequency structure, so the tree-vs-MLP gap collapses in lockstep with the variance removed — '
    'proving the tree wins on typical tabular data precisely because those targets are irregular, not '
    'because trees are simply stronger models."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 025 — The smoothness inductive bias: smoothing the target closes the gap

**Lesson:** [`lessons/0025-inductive-bias-smoothness.html`](../lessons/0025-inductive-bias-smoothness.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 D&B ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — §5.2 (Finding 1: neural networks are biased toward overly smooth solutions). Theory backing: Rahaman et al. 2019, *On the Spectral Bias of Neural Networks* ([arXiv:1806.08734](https://arxiv.org/abs/1806.08734)).

**Dataset tier:** C — synthetic irregular regression, to *isolate the mechanism* (per `lab-authoring`: Tier C is for mechanism isolation). No real dataset can turn its own irregularity on and off; a synthetic target can.

**Skill you are practising:** implement Grinsztajn's **Gaussian target smoother**, then show the GBT-vs-MLP gap **collapses as the target's high-frequency structure is smoothed away** — the experiment that proves smoothness is the mechanism.

**Exit criteria:** EXIT TICKET prints the raw gap (≈ +0.33 R², GBT ahead) collapsing to ≈ 0 once ~70%+ of the target's variance is smoothed away.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, models, the fit/score harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** only (no network — the data is synthetic). The sweep fits 8 models and takes ~15 seconds.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — inductive bias, and the smoothness of neural nets

**Inductive bias.** Any learner that generalizes must *prefer* some functions over others (otherwise it could never choose among the infinitely many curves through the training points). That preference is its inductive bias — not a bug, but the thing that makes learning possible. The only question is whether the bias *matches the data*.

**The MLP's bias is smoothness.** Trained by gradient descent, a neural net fits the **low-frequency** (smooth) part of a target first and the **high-frequency** (jagged) part slowly or not at all — the *spectral bias* (Rahaman 2019). It can represent sharp functions in principle; it just doesn't reach them easily on finite data.

**The tree's disposition is the opposite.** A decision tree splits the space into axis-aligned boxes and predicts a constant per box, so a single split makes an arbitrarily sharp step. Its natural output is **piecewise-constant and jagged** — a perfect match for an irregular target.

**The experiment (§5.2).** To prove the smoothness bias is *the* mechanism, Grinsztajn smooths the **target function** (not the model) with a Gaussian kernel of growing length-scale `h`, then refits both models. Smoothing deletes exactly the high-frequency structure a tree exploits — so if that structure is the tree's edge, the gap should vanish as `h` grows. You will implement the smoother and watch it happen.

**Toy micro-example (not this lab's answer).** Two points with targets `1.0` and `3.0`. At `h → 0` each keeps its own value; at `h → ∞` both weights → 1, so each smoothed value → the mean `2.0`. Bigger `h` = wider averaging = smoother target.

Full write-up + the 1-D fit widget and the gap-vs-smoothing curve: [Lesson 025](../lessons/0025-inductive-bias-smoothness.html).'''),
        md("## Setup — PROVIDED (synthetic data, models, fit/score harness)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — probe the bias

1. **A smooth target from the start.** Replace the target with a purely smooth function (e.g. `y = X[:,0] + 0.5*X[:,1]**2 + X[:,2]`). Re-run the sweep at `h=0`: the MLP should already tie or beat the GBT — no jags to exploit, no tree advantage.
2. **Frequency knob.** In `make_irregular_regression`, raise the wiggle frequency `sin(4.0*X[:,3])` to `sin(10.0*X[:,3])`. Does the raw gap grow? Higher frequency = more of exactly what the MLP cannot fit.
3. **Model smoothness, not target smoothness.** Instead of smoothing the target, widen the MLP (more layers/units) and train longer. Does it ever match the tree on the raw target? (Spectral bias says: slowly, and not cheaply.)
4. **Where it breaks.** Push `h` very large (e.g. 8.0). Both R² should fall — you have smoothed away *all* the signal, not just the jags. Smoothing is a diagnostic, never a fix.'''),
        code(r'''# STRETCH — ungraded.
# def smooth_target_only(y_like):  # idea 1: swap in a smooth target and re-run the sweep
#     pass
# # idea 4: gaps at very large h — both models lose signal, not just the tree
# for h in [4.0, 8.0]:
#     yt = gaussian_smooth_target(Xtr, ytr, h); ye = gaussian_smooth_target(Xte, yte, h)
#     g, m = fit_score(yt, ye)
#     print(f"h={h}: GBT {g:.3f}  MLP {m:.3f}  var kept {np.var(ye)/np.var(yte):.3f}")'''),
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
    student = build(solution=False)
    sol = build(solution=True)
    with open(os.path.join(HERE, "0025-inductive-bias-smoothness.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0025-inductive-bias-smoothness.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0025-inductive-bias-smoothness.ipynb + labs/solutions/0025-inductive-bias-smoothness.ipynb")


if __name__ == "__main__":
    main()
