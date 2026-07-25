import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/avist/Projects/homework")
from lightgbm import LGBMClassifier  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402

d = "/home/avist/Projects/homework/data/"
t = time.time()
df = build_dataset(
    pd.read_parquet(d + "persons.parquet"),
    pd.read_parquet(d + "situations.parquet"),
    pd.read_parquet(d + "responses.parquet"),
)
Xdf, y, mask = split_xy(df)
X = make_feature_pipeline().fit_transform(Xdf).astype(np.float32)[mask]
print(f"prep {time.time() - t:.1f}s  X {X.shape} {X.dtype}", flush=True)

for nj in (12, 6, 2):
    m = LGBMClassifier(
        objective="multiclass", n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, reg_lambda=1.0, random_state=0, n_jobs=nj, verbose=-1,
    )
    t = time.time()
    m.fit(X[:4470], y.iloc[:4470])
    dt = time.time() - t
    print(f"n_jobs={nj}: one multiclass fit {dt:.1f}s -> 25 fits = {25 * dt / 60:.1f} min",
          flush=True)
