"""Decisive test: is the credit_g from-scratch-vs-reference gap the LEARNING-RATE SCHEDULE? (#20/#22)

`_verify_l043_refcheck.py` ruled out training length on credit_g (a longer leash made the reference
slightly worse). The remaining asymmetry is that our `relkit.tabnet.train_tabnet` implements the paper's
prescription — "Initially large learning rate is important, which should be gradually decayed until
convergence" (Arik & Pfister 2019, Appendix F) — as a StepLR decay, while `pytorch_tabnet`'s default
`fit()` uses Adam with NO scheduler.

Test: give the reference the same geometric decay (gamma 0.9 every 20 epochs) on credit_g and compare.

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l043_refsched.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _verify_l043 import load_dense, frame, BUDGET, SEEDS  # noqa: E402

NAME = "credit_g"


def search_ref_scheduled(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        n_units = int(rng.choice([8, 16, 24]))
        clf = TabNetClassifier(
            n_d=n_units, n_a=n_units, n_steps=int(rng.choice([3, 4, 5])),
            gamma=float(rng.choice([1.0, 1.2, 1.5, 2.0])),
            lambda_sparse=float(rng.choice([0.0, 1e-6, 1e-4, 1e-3, 1e-2])),
            optimizer_params=dict(lr=float(rng.choice([0.005, 0.01, 0.02, 0.025]))),
            # the paper's prescription, matched to relkit.tabnet.train_tabnet
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=20, gamma=0.9),
            seed=seed + t, verbose=0, device_name="cpu")
        clf.fit(Xtr, ytr.astype(int), eval_set=[(Xva, yva.astype(int))], eval_metric=["auc"],
                max_epochs=120, patience=12, batch_size=1024, virtual_batch_size=128, drop_last=False)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]


def main():
    t0 = time.time()
    X, y = load_dense(NAME)
    ours = json.load(open(os.path.join(HERE, "_verify_l043_results.json")))
    ours = ours["bakeoff"]["per_dataset"][NAME]["tabnet"]["mean"]
    prior = json.load(open(os.path.join(HERE, "_verify_l043_refcheck_results.json")))[NAME]

    scores = [search_ref_scheduled(*frame(X, y, s), budget=BUDGET, seed=s) for s in SEEDS]
    m = float(np.mean(scores))
    out = {"dataset": NAME, "scratch": round(ours, 3),
           "reference_no_scheduler": prior["shared_budget"]["mean"],
           "reference_with_paper_lr_decay": round(m, 3),
           "seeds": [round(float(s), 3) for s in scores],
           "delta_vs_scratch": round(ours - m, 3),
           "within_tol": abs(ours - m) < 0.04}
    print(f"{NAME}: scratch {ours:.3f} | reference no-scheduler "
          f"{prior['shared_budget']['mean']:.3f} | reference + paper LR decay {m:.3f} "
          f"(|Δ| {abs(ours-m):.3f}) -> {'VALIDATED' if out['within_tol'] else 'STILL OUT OF TOLERANCE'}")
    with open(os.path.join(HERE, "_verify_l043_refsched_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote labs/_verify_l043_refsched_results.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
