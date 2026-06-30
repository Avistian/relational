"""Builds labs/0009-feature-engineering.ipynb from cell specs. Deleted after use."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Lab 009 — Engineer the features models can't learn

**Lesson:** [`lessons/0009-feature-engineering.html`](../lessons/0009-feature-engineering.html) · **Phase / Year:** Phase 1 / Year 1

**Skill you are practising:** engineer features that encode structure a model can't synthesize itself (ratios, cyclical datetime), and keep fit-bearing features (target encoding) inside the pipeline so they never leak.

**Exit criteria:** the EXIT TICKET prints raw-vs-engineered R² for the ratio and datetime demos, plus the leaky-vs-safe target-encoding gap, with your one-sentence takeaway.

---

### How this notebook works
- **PROVIDED** cells are complete — just run them.
- **TODO** cells have blanks (`____`). Fill them in.
- **CHECK** cells auto-run and give you immediate feedback — don't edit them.
- Run top to bottom. When the **EXIT TICKET** prints cleanly, paste it back to your teacher (or say *"lab done"*).

### Environment
One-time setup from the repo root: `bash labs/setup-env.sh`.  
Then select kernel **Relational Labs (.venv)** or interpreter `.venv/bin/python`.""")

md("## Setup — PROVIDED (run me)")
code("""# PROVIDED
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

RS = 0
rng = np.random.RandomState(RS)
print("setup ok")""")

# ---- Task 1: ratio of differences ----
md("""## Task 1 — The ratio of differences — TODO

Heaton (2016) showed **no** common model learns a *ratio of differences* on its own. Here the
target is `y = (a - b) / (c - d)`. Engineer that exact feature and watch a linear model jump.

Fill the blank so `X_eng` is `X_raw` with the engineered ratio column appended.
Hint: the ratio is `(a - b) / (c - d)`; stack with `np.column_stack([X_raw, ratio])`.""")
code("""n = 4000
a = rng.uniform(1, 9, n); b = rng.uniform(1, 9, n)
c = rng.uniform(1, 9, n); d = rng.uniform(9.5, 11, n)   # c-d bounded off zero but small
y = (a - b) / (c - d) + rng.normal(0, 0.05, n)
X_raw = np.column_stack([a, b, c, d])

# TODO: engineer the ratio-of-differences feature and append it.
ratio = ____
X_eng = np.column_stack([X_raw, ratio.reshape(-1, 1)])

cv = KFold(5, shuffle=True, random_state=RS)
r2_raw = cross_val_score(Ridge(), X_raw, y, cv=cv, scoring="r2").mean()
r2_eng = cross_val_score(Ridge(), X_eng, y, cv=cv, scoring="r2").mean()
print(f"Ridge  raw R2 {r2_raw:+.3f}   +engineered ratio R2 {r2_eng:+.3f}")""")
code("""# CHECK — do not edit.
assert X_eng.shape[1] == X_raw.shape[1] + 1, "X_eng should have exactly one extra column."
assert r2_raw < 0.8, "Raw R2 should be clearly limited (the model can't see the ratio)."
assert r2_eng > 0.95, "With the ratio engineered in, Ridge should be near-perfect."
print(f"Task 1 ok — engineering the ratio lifted Ridge from {r2_raw:.3f} to {r2_eng:.3f}.")""")

# ---- Task 2: cyclical datetime ----
md("""## Task 2 — Cyclical datetime encoding — TODO

A raw timestamp hides an hour-of-day signal inside one big number. Decompose it into a
**sin/cos** pair so the cycle (and the midnight wrap) becomes visible.

Fill the blanks: `hour_sin = sin(2π·hour/24)` and `hour_cos = cos(2π·hour/24)`.""")
code("""t = rng.uniform(0, 1000, n)          # a "timestamp"
hour = t % 24
y2 = np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.1, n)
X_t_raw = t.reshape(-1, 1)

# TODO: build the cyclical encoding.
hour_sin = ____
hour_cos = ____
X_t_eng = np.column_stack([hour_sin, hour_cos])

r2_traw = cross_val_score(HistGradientBoostingRegressor(random_state=RS), X_t_raw, y2, cv=cv, scoring="r2").mean()
r2_teng = cross_val_score(HistGradientBoostingRegressor(random_state=RS), X_t_eng, y2, cv=cv, scoring="r2").mean()
print(f"Datetime  raw-timestamp R2 {r2_traw:+.3f}   +cyclical hour R2 {r2_teng:+.3f}")""")
code("""# CHECK — do not edit.
assert np.allclose(hour_sin**2 + hour_cos**2, 1.0, atol=1e-6), "sin^2 + cos^2 must equal 1 — check your encoding."
assert r2_teng > r2_traw + 0.05, "The cyclical encoding should clearly beat the raw timestamp."
print(f"Task 2 ok — sin/cos lifted R2 from {r2_traw:.3f} to {r2_teng:.3f}.")""")

# ---- Task 3: target-encoding leakage ----
md("""## Task 3 — Target encoding: leaky vs leak-free — TODO

Fit-bearing features can **leak**. Here a high-cardinality category is **pure noise** (independent
of `y`). Mean-encoding it on the *whole* dataset lets each row peek at its own label → fake skill.
Doing it inside a pipeline (fit on train folds only) tells the truth (~0.5).

Fill the blank so `safe` evaluates a `Pipeline` of `TargetEncoder` → `LogisticRegression`.""")
code("""cat = rng.randint(0, n // 2, size=n)      # ~2 rows per category
yc = rng.randint(0, 2, size=n)            # 50/50, INDEPENDENT of cat
scv = StratifiedKFold(5, shuffle=True, random_state=RS)

# LEAKY (provided): mean-encode using the whole dataset, then cross-validate.
leaky_feat = pd.Series(yc).groupby(cat).transform("mean").to_numpy().reshape(-1, 1)
leaky = cross_val_score(LogisticRegression(), leaky_feat, yc, cv=scv, scoring="accuracy").mean()

# TODO: leak-free — TargetEncoder INSIDE a pipeline so it is fit per fold.
safe_pipe = Pipeline([("enc", ____), ("clf", LogisticRegression())])
safe = cross_val_score(safe_pipe, pd.DataFrame({"cat": cat}), yc, cv=scv, scoring="accuracy").mean()

print(f"leaky (encode on full data) CV acc : {leaky:.3f}")
print(f"safe  (encode inside pipeline) acc : {safe:.3f}   (truth ~0.5 — the category is noise)")""")
code("""# CHECK — do not edit.
assert leaky > 0.6, "Leaky encoding should look deceptively good on this noise category."
assert safe < 0.55, "The leak-free pipeline should report ~chance on a noise category."
print(f"Task 3 ok — leakage inflated accuracy by {leaky - safe:.3f} on a feature with no real signal.")""")

# ---- Exit ticket ----
md("""## Exit ticket — TODO

Fill the one-sentence takeaway, run, and paste the output back to your teacher.""")
code('''# TODO: complete the takeaway.
print("=== EXIT TICKET — Lesson 009 ===")
print(f"ratio of differences : Ridge R2 {r2_raw:+.3f} -> {r2_eng:+.3f}")
print(f"cyclical datetime    : HistGBDT R2 {r2_traw:+.3f} -> {r2_teng:+.3f}")
print(f"target encoding      : leaky {leaky:.3f} vs leak-free {safe:.3f} (noise category)")
print()
print("takeaway:", "____")   # which feature could the model never learn alone, and why must target encoding live in the pipeline?''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "relational-labs"},
    "language_info": {"name": "python"},
}
out = "/home/avist/Projects/relational/labs/0009-feature-engineering.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
