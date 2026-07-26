"""L037 — same code, same data, same seed, same machine, ONE dependency version apart.

Runs exactly the reference configuration of `_repro_l037.py` (person-grouped
5-fold LightGBM, n_jobs=6, seed 0, float32 input) and prints the environment
fingerprint next to the OOF hash, so the same script can be executed under two
interpreters and the outputs compared.

    .venv/bin/python           labs/_repro_env_l037.py    # lightgbm 4.6.0
    /tmp/l037-old/bin/python   labs/_repro_env_l037.py    # lightgbm 4.5.0

Appends to labs/_repro_env_l037_results.json keyed by the environment tuple.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOMEWORK = Path("/home/avist/Projects/homework")
sys.path.insert(0, str(HOMEWORK))

import lightgbm  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402
from src.modeling import CLASSES, RANDOM_STATE, cv_splitter  # noqa: E402

OUT = Path(__file__).with_name("_repro_env_l037_results.json")


def main() -> None:
    d = HOMEWORK / "data"
    df = build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )
    df_feat, y, mask = split_xy(df)
    X = make_feature_pipeline().fit_transform(df_feat).astype(np.float32)[mask]
    groups = df.loc[mask, "person_id"].to_numpy()
    y_idx = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    oof = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float64)
    per_fold = []
    t0 = time.time()
    for i, (tr, te) in enumerate(cv_splitter().split(X, y_idx, groups=groups)):
        m = LGBMClassifier(
            objective="multiclass", n_estimators=400, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=6, verbose=-1,
        ).fit(X[tr], y_idx[tr])
        p = m.predict_proba(X[te])
        oof[te] = p
        per_fold.append(float(log_loss(y_idx[te], p, labels=list(range(len(CLASSES))))))
        print(f"  fold {i}: ll={per_fold[-1]:.6f}", flush=True)

    key = f"lightgbm-{lightgbm.__version__}_sklearn-{sklearn.__version__}_numpy-{np.__version__}"
    rec = {
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
        "per_fold_log_loss": per_fold,
        "mean_log_loss": float(np.mean(per_fold)),
        "sha256": hashlib.sha256(oof.tobytes()).hexdigest()[:16],
        "seconds": round(time.time() - t0, 1),
    }
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[key] = rec
    np.save(Path(__file__).with_name(f"_repro_env_oof_lgb{lightgbm.__version__}.npy"), oof)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
