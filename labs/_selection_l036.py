"""L036: re-analyse the homework's own saved per-fold CV artifacts.

Answers the audit question "was the shipped winner chosen by signal or by noise?"
using the user's own artifacts/cv_folds_*.csv — no refitting involved.

Applies the L023 toolkit: paired per-fold deltas, the naive paired t-test the
report used, the Nadeau & Bengio (2003) corrected resampled t-test, and a
Holm correction for the 4 comparisons the report ran.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/home/avist/Projects/homework/artifacts")
MODELS = {
    "M0": "cv_folds_M0_logreg.csv",
    "M1": "cv_folds_M1_lgbm_iso.csv",
    "M2a": "cv_folds_M2a_lgbm_km.csv",
    "M2b": "cv_folds_M2b_lgbm_prop.csv",
    "M2c": "cv_folds_M2c_lgbm_km_prop.csv",
}

folds = {}
for name, fn in MODELS.items():
    df = pd.read_csv(ART / fn)
    folds[name] = df
print("columns:", list(folds["M1"].columns))
print()

ll = pd.DataFrame({k: v["log_loss"] for k, v in folds.items()})
ece = pd.DataFrame({k: v["ece_top"] for k, v in folds.items()})
ntest = folds["M1"]["n_test"].to_numpy()
print("per-fold log-loss (the user's own 5 folds)")
print(ll.round(4).to_string())
print()
print("means:", ll.mean().round(4).to_dict())
print("stds :", ll.std(ddof=1).round(4).to_dict())
print("ece means:", ece.mean().round(4).to_dict())
print()

n_folds = len(ll)
# Nadeau & Bengio correction factor: folds share training data, so the naive
# variance sigma^2/n understates it by (1/n + n_test/n_train).
n_test = float(ntest.mean())
n_train = float(len(folds["M1"]) and (ll.shape[0] and 5587 - n_test))
corr_factor = 1.0 / n_folds + n_test / n_train
print(f"n_folds={n_folds} mean n_test={n_test:.0f} n_train={n_train:.0f} "
      f"corrected-variance factor={corr_factor:.4f} (naive is 1/n={1/n_folds:.4f})")
print()

rows = []
for a, b in [("M1", "M0"), ("M2a", "M1"), ("M2b", "M1"), ("M2c", "M1"),
             ("M2a", "M2c"), ("M2a", "M2b")]:
    d = (ll[a] - ll[b]).to_numpy()
    mean_d, sd = d.mean(), d.std(ddof=1)
    t_naive = mean_d / (sd / np.sqrt(n_folds))
    p_naive = 2 * stats.t.sf(abs(t_naive), df=n_folds - 1)
    t_corr = mean_d / np.sqrt(corr_factor * sd**2)
    p_corr = 2 * stats.t.sf(abs(t_corr), df=n_folds - 1)
    half_naive = stats.t.ppf(0.975, n_folds - 1) * sd / np.sqrt(n_folds)
    half_corr = stats.t.ppf(0.975, n_folds - 1) * np.sqrt(corr_factor * sd**2)
    rows.append({
        "cmp": f"{a} - {b}", "mean_d": mean_d, "sd_d": sd,
        "t_naive": t_naive, "p_naive": p_naive,
        "t_corr": t_corr, "p_corr": p_corr,
        "half_naive": half_naive, "half_corr": half_corr,
        "n_folds_favouring_a": int((d < 0).sum()),
    })
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(5).to_string(index=False))
print()

# Holm-Bonferroni over the 4 comparisons the report actually reported.
reported = res[res["cmp"].isin(["M1 - M0", "M2a - M1", "M2b - M1", "M2c - M1"])].copy()
reported = reported.sort_values("p_naive").reset_index(drop=True)
m = len(reported)
holm = []
for i, r in reported.iterrows():
    holm.append(min(1.0, r["p_naive"] * (m - i)))
reported["p_holm"] = np.maximum.accumulate(holm)
print("Holm-corrected (on the naive p-values the report quoted):")
print(reported[["cmp", "mean_d", "p_naive", "p_holm", "p_corr"]].round(5).to_string(index=False))
print()

# How big is the winner's margin relative to fold noise?
d = (ll["M2a"] - ll["M1"]).to_numpy()
print(f"M2a - M1 per-fold deltas: {np.round(d, 4).tolist()}")
print(f"  mean {d.mean():+.4f} | fold-to-fold sd of M1 log-loss {ll['M1'].std(ddof=1):.4f}")
print(f"  ratio |mean delta| / sd(M1) = {abs(d.mean()) / ll['M1'].std(ddof=1):.3f}")
print(f"  folds where M2a beats M1: {(d < 0).sum()}/5")

# Would a different seed have picked a different winner? Rank stability proxy:
# leave-one-fold-out recomputation of the argmin.
print("\nleave-one-fold-out winner (argmin mean log-loss):")
for drop in range(n_folds):
    sub = ll.drop(index=drop)
    print(f"  drop fold {drop}: {sub.mean().idxmin()}  "
          f"({', '.join(f'{k}={v:.4f}' for k, v in sub.mean().round(4).items())})")

out = {
    "per_fold_log_loss": {k: [round(float(x), 6) for x in v] for k, v in ll.items()},
    "per_fold_ece_top": {k: [round(float(x), 6) for x in v] for k, v in ece.items()},
    "means_log_loss": {k: round(float(v), 6) for k, v in ll.mean().items()},
    "stds_log_loss": {k: round(float(v), 6) for k, v in ll.std(ddof=1).items()},
    "means_ece_top": {k: round(float(v), 6) for k, v in ece.mean().items()},
    "n_test_mean": n_test,
    "corr_factor": corr_factor,
    "tests": res.round(6).to_dict(orient="records"),
    "holm": reported[["cmp", "mean_d", "p_naive", "p_holm", "p_corr"]].round(6)
              .to_dict(orient="records"),
}
Path(__file__).with_name("_selection_l036_results.json").write_text(json.dumps(out, indent=2))
print("\nwrote _selection_l036_results.json")
