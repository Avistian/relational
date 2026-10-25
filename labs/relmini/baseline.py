"""Point-in-time baseline for RelBench rel-f1 `driver-dnf` (local cache, read-only).

For each task row (driverId, date) features use only `results` rows with date strictly
before the task date. `feature_cutoff` records the latest result date used.
Metrics are LOCAL reproducibility evidence only; not RelBench-comparable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path.home() / ".cache" / "relbench" / "rel-f1"
SEED = 0
FEATURES = ["f_n_prior", "f_dnf_rate", "f_mean_grid", "f_mean_points", "f_age_years", "f_has_history"]

CONSTRUCTOR_FEATURES = ["f_constructor_n_prior", "f_constructor_mean_points", "f_constructor_has_history"]


def load_real(split, cache=CACHE):
    task = pd.read_parquet(cache / "tasks" / "driver-dnf" / f"{split}.parquet")
    res = pd.read_parquet(cache / "db" / "results.parquet")
    drivers = pd.read_parquet(cache / "db" / "drivers.parquet")
    return task, res, drivers


def build_features(task, results, drivers, constructor_results=None):
    """Row-aligned with `task`. Prior stats come from results with date < task date."""
    r = results[["driverId", "date", "grid", "position", "points"]].copy()
    r["driverId"] = r["driverId"].astype("int64")
    r["dnf"] = r["position"].isna().astype(float)  # proxy: no classified position
    # per-(driver, day) sums, then cumulative per driver
    day = r.groupby(["driverId", "date"], as_index=False).agg(
        n=("dnf", "size"), dnf=("dnf", "sum"), grid=("grid", "sum"), points=("points", "sum"))
    day = day.sort_values(["driverId", "date"])
    for c in ["n", "dnf", "grid", "points"]:
        day[c] = day.groupby("driverId")[c].cumsum()
    day = day.rename(columns={"date": "feature_cutoff"})

    t = task.reset_index(drop=True).copy()
    t["driverId"] = t["driverId"].astype("int64")
    t["_row"] = np.arange(len(t))
    t = t.sort_values("date")
    day = day.sort_values("feature_cutoff")
    m = pd.merge_asof(t, day, left_on="date", right_on="feature_cutoff", by="driverId",
                      direction="backward", allow_exact_matches=False)
    m = m.sort_values("_row").reset_index(drop=True)

    n = m["n"].fillna(0)
    safe = n.where(n > 0, np.nan)
    m["f_n_prior"] = n
    m["f_dnf_rate"] = (m["dnf"] / safe).fillna(0.5)
    m["f_mean_grid"] = (m["grid"] / safe).fillna(10.0)  # constant prior; no global stats
    m["f_mean_points"] = (m["points"] / safe).fillna(0.0)
    dob = drivers.assign(driverId=drivers["driverId"].astype("int64")).set_index("driverId")["dob"]
    m["f_age_years"] = ((m["date"] - m["driverId"].map(dob)).dt.days / 365.25).fillna(30.0)
    m["f_has_history"] = (n > 0).astype(float)
    if constructor_results is not None:
        # Resolve the driver's team from strictly prior results, never the label-day team.
        teams = results[["driverId", "constructorId", "date"]].copy()
        teams["driverId"] = teams["driverId"].astype("int64")
        # Historic shared drives can give several teams on one date. Do not pick
        # one by file order or fall back to an older known team: mark it unknown.
        teams = teams.groupby(["driverId", "date"], as_index=False).agg(
            constructorId=("constructorId", "min"), team_count=("constructorId", "nunique"))
        teams.loc[teams["team_count"] != 1, "constructorId"] = -1
        teams = teams.drop(columns="team_count")
        teams = teams.rename(columns={"date": "constructor_assignment_cutoff"})
        m = pd.merge_asof(m.sort_values("date"), teams.sort_values("constructor_assignment_cutoff"),
                          left_on="date", right_on="constructor_assignment_cutoff", by="driverId",
                          direction="backward", allow_exact_matches=False)
        c = constructor_results.groupby(["constructorId", "date"], as_index=False).agg(
            cn=("points", "size"), cp=("points", "sum")).sort_values(["constructorId", "date"])
        c[["cn", "cp"]] = c.groupby("constructorId")[["cn", "cp"]].cumsum()
        c = c.rename(columns={"date": "constructor_cutoff"})
        # Nullable join keys use a sentinel outside the cached nonnegative ID domain.
        m["constructorId"] = m["constructorId"].fillna(-1).astype("int64")
        c["constructorId"] = c["constructorId"].astype("int64")
        m = pd.merge_asof(m.sort_values("date"), c.sort_values("constructor_cutoff"),
                          left_on="date", right_on="constructor_cutoff", by="constructorId",
                          direction="backward", allow_exact_matches=False)
        m["f_constructor_n_prior"] = m["cn"].fillna(0.)
        m["f_constructor_mean_points"] = (m["cp"] / m["cn"]).fillna(0.)
        m["f_constructor_has_history"] = m["cn"].notna().astype(float)
        m = m.drop(columns=["cn", "cp"]).sort_values("_row").reset_index(drop=True)
    return m.drop(columns=["_row", "n", "dnf", "grid", "points"])


def assert_no_leakage(feats):
    if "feature_cutoff" not in feats:
        raise ValueError("feature_cutoff is required to audit leakage")
    for column in ("feature_cutoff", "constructor_assignment_cutoff", "constructor_cutoff"):
        if column not in feats:
            continue
        has = feats[column].notna()
        bad = int((feats.loc[has, column] >= feats.loc[has, "date"]).sum())
        assert bad == 0, f"leakage: {bad} rows use data at/after the task timestamp ({column})"


def auroc(y, p):
    return float(roc_auc_score(y, p))


def fit_predict(train, eval_, results, drivers, seed=SEED, *, constructor_results=None, use_constructor=False):
    if use_constructor and constructor_results is None:
        raise ValueError("constructor results required when constructor features are enabled")
    source = constructor_results if use_constructor else None
    ftr = build_features(train, results, drivers, source)
    fev = build_features(eval_, results, drivers, source)
    columns = FEATURES + (CONSTRUCTOR_FEATURES if use_constructor else [])
    assert_no_leakage(ftr)
    assert_no_leakage(fev)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(ftr[columns], train["did_not_finish"].to_numpy())
    return model.predict_proba(fev[columns])[:, 1]


def run(seed=SEED, cache=CACHE):
    train, res, drivers = load_real("train", cache)
    out = {"seed": seed, "model": "StandardScaler+LogisticRegression", "data": "REAL rel-f1 driver-dnf (local cache)"}
    for split in ("val", "test"):
        ev, _, _ = load_real(split, cache)
        p = fit_predict(train, ev, res, drivers, seed)
        out[f"{split}_auroc"] = auroc(ev["did_not_finish"].to_numpy(), p)
        out[f"{split}_n"] = int(len(ev))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
