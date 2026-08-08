"""Why did the from-scratch TabNet beat `pytorch_tabnet` in `_verify_l043.py`? (honesty follow-up, #20/#22)

The main run reported from-scratch 0.748 vs reference 0.694 (credit_g) and 0.824 vs 0.766 (diabetes),
both OUTSIDE the 0.04 tolerance — and in the *unexpected* direction (a from-scratch model that beats the
reference usually means the comparison, not the model, is broken).

Hypothesis: the reference runs its OWN training loop, so "identical protocol" does not actually transfer.
Its trials in the main run stopped very early (several with `best_epoch = 0`), which suggests the shared
patience/epoch budget starves it rather than that the architectures differ.

Test: give the reference a *longer* leash (patience 25, up to 200 epochs) with everything else the same,
and see whether it closes the gap. If it does, the gap was training length, not architecture.

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l043_refcheck.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _verify_l043 import load_dense, frame, DATASETS, BUDGET, SEEDS  # noqa: E402

DATA = DATASETS[:2]


def search_ref(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed, max_epochs, patience):
    from pytorch_tabnet.tab_model import TabNetClassifier
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None, "best_epoch": None}
    for t in range(budget):
        n_units = int(rng.choice([8, 16, 24]))
        clf = TabNetClassifier(
            n_d=n_units, n_a=n_units, n_steps=int(rng.choice([3, 4, 5])),
            gamma=float(rng.choice([1.0, 1.2, 1.5, 2.0])),
            lambda_sparse=float(rng.choice([0.0, 1e-6, 1e-4, 1e-3, 1e-2])),
            optimizer_params=dict(lr=float(rng.choice([0.005, 0.01, 0.02, 0.025]))),
            seed=seed + t, verbose=0, device_name="cpu")
        clf.fit(Xtr, ytr.astype(int), eval_set=[(Xva, yva.astype(int))], eval_metric=["auc"],
                max_epochs=max_epochs, patience=patience, batch_size=1024,
                virtual_batch_size=128, drop_last=False)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]),
                    "best_epoch": int(clf.best_epoch)}
    return best


def main():
    t0 = time.time()
    scratch = json.load(open(os.path.join(HERE, "_verify_l043_results.json")))
    out = {}
    for name in DATA:
        X, y = load_dense(name)
        ours = scratch["bakeoff"]["per_dataset"][name]["tabnet"]["mean"]
        row = {"scratch": round(ours, 3)}
        for tag, (ep, pat) in {"shared_budget": (120, 12), "long_leash": (200, 25)}.items():
            res = [search_ref(*frame(X, y, s), budget=BUDGET, seed=s, max_epochs=ep, patience=pat)
                   for s in SEEDS]
            m = float(np.mean([r["test"] for r in res]))
            row[tag] = {"mean": round(m, 3), "delta_vs_scratch": round(ours - m, 3),
                        "within_tol": abs(ours - m) < 0.04,
                        "best_epochs": [r["best_epoch"] for r in res]}
            print(f"{name:>10} [{tag:>13}]: reference {m:.3f} vs scratch {ours:.3f} "
                  f"(|Δ| {abs(ours-m):.3f}) best_epochs={row[tag]['best_epochs']}", flush=True)
        out[name] = row

    with open(os.path.join(HERE, "_verify_l043_refcheck_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote labs/_verify_l043_refcheck_results.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
