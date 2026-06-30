"""Verify the target-encoding leakage demo for Lesson 009 lab."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

RS = 0
rng = np.random.RandomState(RS)
n = 4000
# High-cardinality category that is PURE NOISE (no real link to y).
cat = rng.randint(0, n // 2, size=n)      # ~2 rows per category
y = rng.randint(0, 2, size=n)             # 50/50, independent of cat
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

# LEAKY: mean-encode using the whole dataset (each row sees its own label via its category).
means = pd.Series(y).groupby(cat).transform("mean").to_numpy().reshape(-1, 1)
leaky = cross_val_score(LogisticRegression(), means, y, cv=cv, scoring="accuracy").mean()

# SAFE: TargetEncoder inside a pipeline, fit on train folds only (cross-fitted).
pipe = Pipeline([("enc", TargetEncoder()), ("clf", LogisticRegression())])
safe = cross_val_score(pipe, pd.DataFrame({"cat": cat}), y, cv=cv, scoring="accuracy").mean()

print(f"leaky (encode on full data) CV acc : {leaky:.3f}")
print(f"safe  (encode inside pipeline) acc : {safe:.3f}")
print(f"true signal: none (y independent of cat) -> honest acc ~0.5")
