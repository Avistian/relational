"""Scratch verification for Lesson 009 numbers. Deleted after use."""
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

RS = 0
rng = np.random.RandomState(RS)
n = 4000
# Four raw features; target depends on a RATIO OF DIFFERENCES (Heaton 2016: no model learns it)
a = rng.uniform(1, 9, n); b = rng.uniform(1, 9, n)
c = rng.uniform(1, 9, n); d = rng.uniform(9.5, 11, n)  # c-d in ~[-10,-0.5]: bounded off 0 but small, so ratio is sharply nonlinear
y = (a - b) / (c - d) + rng.normal(0, 0.05, n)

X_raw = np.column_stack([a, b, c, d])
ratio = ((a - b) / (c - d)).reshape(-1, 1)
X_eng = np.column_stack([a, b, c, d, ratio])

cv = KFold(5, shuffle=True, random_state=RS)
for name, model in [("Ridge", Ridge()),
                    ("HistGBDT", HistGradientBoostingRegressor(random_state=RS))]:
    r2_raw = cross_val_score(model, X_raw, y, cv=cv, scoring="r2").mean()
    r2_eng = cross_val_score(model, X_eng, y, cv=cv, scoring="r2").mean()
    print(f"{name:9s}  raw R2 {r2_raw:+.3f}   +engineered ratio R2 {r2_eng:+.3f}")

# datetime decomposition demo: signal is hour-of-day cyclical; raw unix-ish int can't see it
t = rng.uniform(0, 1000, n)           # "timestamp"
hour = (t % 24)
y2 = np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.1, n)
X_t_raw = t.reshape(-1, 1)
X_t_eng = np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])
r2_traw = cross_val_score(HistGradientBoostingRegressor(random_state=RS), X_t_raw, y2, cv=cv, scoring="r2").mean()
r2_teng = cross_val_score(HistGradientBoostingRegressor(random_state=RS), X_t_eng, y2, cv=cv, scoring="r2").mean()
print(f"\nDatetime  raw-timestamp R2 {r2_traw:+.3f}   +cyclical hour R2 {r2_teng:+.3f}")
