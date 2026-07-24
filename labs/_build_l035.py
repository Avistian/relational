"""Build Lab 035 (What joins destroy — construct an aggregation collision, then break it) — student + solution.

Tier C (small SYNTHETIC tables: customers, orders, order_items). L035's paper is Fey et al. 2024 §1-2 (a
POSITION preview, not an architecture), so the implementation scope (standard #18) is to DEMONSTRATE the
paper's issue (4) — "by forcing data into a single table, information is aggregated into lower-granularity
features, thus losing out on valuable fine-grain signal" — as a concrete, runnable aggregation collision,
then show which structure a graph-keeping model recovers.

Three tasks:
  * Task 1 (crucial fragment) — flatten with the L034 aggregate (count/sum/avg/max over pre-t orders) and
    CONFIRM two customers with different histories collide onto an identical feature row.
  * Task 2 (crucial fragment) — fit a real classifier on the flat features and prove it gives the two
    colliding customers the IDENTICAL predicted probability (identical input -> identical output; no tuning
    can fix an information loss).
  * Task 3 (crucial fragment) — recover the discarded structure with two structure-aware features:
    spend_trend (last - first, needs temporal ORDER) and n_distinct_products (a two-hop join, needs event
    IDENTITY) — and show the rows now differ.

Deterministic synthetic data (no randomness). Cutoff t = 2024-06-01; all orders are pre-t (PIT handled in
L034), so the lesson here is purely the loss, not leakage. The collision pair:
  Ada: orders Jan $10 (milk), Mar $30 (bread), May $50 (eggs)  -> rising, 3 products
  Bo : orders Jan $50 (wine), Mar $30 (wine),  May $10 (wine)  -> falling, 1 product
  both -> n=3, total=90, avg=30, max=50 (IDENTICAL flat row)

Run: .venv/bin/python labs/_build_l035.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0035-what-joins-destroy.ipynb \
    labs/solutions/0035-what-joins-destroy.ipynb
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


SETUP = r'''# PROVIDED — the toy relational database (Tier C, synthetic) + labels + cutoff. Just run.
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# The prediction time. Every order below is BEFORE t (point-in-time correctness was L034's job);
# this lab isolates the LOSS from flattening, not leakage.
t = pd.Timestamp("2024-06-01")

# --- customers dimension (grain: one row per customer) + a churn label to predict ---
customers = pd.DataFrame({
    "customer_id": ["Ada", "Bo", "Cy", "Di", "Ez", "Fi"],
    "segment":     ["retail", "retail", "smb", "retail", "smb", "retail"],
    "churn":       [0,        1,       0,     1,       0,     1],   # ground truth (differs for Ada vs Bo!)
})

# --- orders fact/event table (FK customer_id -> customers). order_ts gives the TIME ORDER. ---
orders = pd.DataFrame({
    "order_id":    ["o1","o2","o3", "o4","o5","o6", "o7","o8", "o9", "o10","o11","o12","o13", "o14","o15","o16"],
    "customer_id": ["Ada","Ada","Ada", "Bo","Bo","Bo", "Cy","Cy", "Di", "Ez","Ez","Ez","Ez", "Fi","Fi","Fi"],
    "order_ts":    pd.to_datetime([
        "2024-01-10","2024-03-05","2024-05-20",   # Ada: 10, 30, 50  -> RISING
        "2024-01-12","2024-03-08","2024-05-22",   # Bo : 50, 30, 10  -> FALLING
        "2024-02-01","2024-04-01",                 # Cy : 20, 20
        "2024-02-15",                              # Di : 100
        "2024-01-05","2024-02-05","2024-03-05","2024-04-05",  # Ez: 5,5,5,5
        "2024-01-20","2024-03-20","2024-05-01",   # Fi : 60, 20, 40
    ]),
    "amount":      [10,30,50,  50,30,10,  20,20,  100,  5,5,5,5,  60,20,40],
})

# --- order_items detail table (FK order_id -> orders): WHICH product each order bought (a 2nd hop) ---
order_items = pd.DataFrame({
    "item_id":    ["i1","i2","i3",  "i4","i5","i6",  "i7","i8",  "i9",  "i10","i11","i12","i13",  "i14","i15","i16"],
    "order_id":   ["o1","o2","o3",  "o4","o5","o6",  "o7","o8",  "o9",  "o10","o11","o12","o13",  "o14","o15","o16"],
    "product_id": ["milk","bread","eggs",   "wine","wine","wine",   "milk","eggs",   "milk",
                   "milk","milk","milk","milk",   "bread","eggs","milk"],
})

print("customers:", customers.shape, "| orders:", orders.shape, "| order_items:", order_items.shape)
print("The collision pair: Ada (rising spend, 3 products) vs Bo (falling spend, 1 product).")
orders[orders.customer_id.isin(["Ada","Bo"])]'''


# ---- Task 1: flatten, and confirm the collision ----
T1_MD = r'''## Task 1 — flatten, and confirm the collision (crucial fragment) — TODO

**Goal:** reuse the L034 flatten — for each customer, aggregate their pre-`t` orders into
`n_orders`, `total_spend`, `avg_basket`, `max_basket` — then look at **Ada** and **Bo**.

**Why it matters:** Ada's spend *rises* ($10→$30→$50) across three different products; Bo's *falls*
($50→$30→$10) on a single product. They are obviously different customers. Fey et al. 2024 §2, issue (4):
forcing data into one table "aggregates into lower-granularity features, thus losing valuable fine-grain
signal." You are about to see that loss become **total** — the two rows come out identical.

**Hint boundary:** filter `orders` to `order_ts < t`, `groupby("customer_id")`, and `.agg(...)` the four
named columns (count on `order_id`; sum/mean/max on `amount`). Same move as L034 Task 1.'''

T1_CODE = r'''# TODO — fill the two blanks (the flatten)
AGG_COLS = ["n_orders", "total_spend", "avg_basket", "max_basket"]

def flatten(orders, t):
    past = orders[____]                          # keep only orders placed STRICTLY BEFORE t
    agg = (past.groupby(____)                     # one group per customer -> the customer grain
                .agg(n_orders=("order_id", "count"),
                     total_spend=("amount", "sum"),
                     avg_basket=("amount", "mean"),
                     max_basket=("amount", "max"))
                .reset_index())
    return agg

flat = flatten(orders, t)
print(flat.to_string(index=False))
print("\nAda vs Bo on the flat features:")
print(flat[flat.customer_id.isin(["Ada","Bo"])][["customer_id"] + AGG_COLS].to_string(index=False))'''

T1_SOL = (T1_CODE
          .replace('    past = orders[____]                          # keep only orders placed STRICTLY BEFORE t',
                   '    past = orders[orders["order_ts"] < t]         # keep only orders placed STRICTLY BEFORE t')
          .replace('    agg = (past.groupby(____)                     # one group per customer -> the customer grain',
                   '    agg = (past.groupby("customer_id")            # one group per customer -> the customer grain'))

T1_CHECK = r'''# CHECK — do not edit
_ada = flat.set_index("customer_id").loc["Ada", AGG_COLS]
_bo  = flat.set_index("customer_id").loc["Bo",  AGG_COLS]
assert (_ada.values == _bo.values).all(), "Ada and Bo should collide onto an identical feature row."
assert int(_ada["n_orders"]) == 3 and int(_ada["total_spend"]) == 90, "collision row should be n=3, total=90."
assert int(_ada["avg_basket"]) == 30 and int(_ada["max_basket"]) == 50, "collision row should be avg=30, max=50."
print("Task 1 ok — Ada and Bo are DIFFERENT customers but flatten to the IDENTICAL row "
      "n=3/total=90/avg=30/max=50. Aggregation is lossy: this is an aggregation collision.")'''


# ---- Task 2: prove a fitted model can't separate them ----
T2_MD = r'''## Task 2 — prove a fitted model gives them the same prediction (crucial fragment) — TODO

**Goal:** fit a real classifier on the flat features (`AGG_COLS`) to predict `churn`, then read off the
predicted probability for **Ada** and **Bo**. Show they are **identical**.

**Why it matters:** this is the payload of the lesson. Ada's true label is `churn=0` and Bo's is `churn=1`
— but their feature rows are identical, so **any** model, however well tuned, feeds on the same input and
must emit the same output. An aggregation collision is an *information* loss, not a capacity or tuning
problem; the model is guaranteed to be wrong about at least one of them.

**Hint boundary:** build `X = flat[AGG_COLS]`, `y = flat["churn"]`; fit any deterministic sklearn classifier
(e.g. `LogisticRegression(max_iter=1000)` or `DecisionTreeClassifier(random_state=0)`); call
`predict_proba` and compare the Ada row to the Bo row. Get each customer's row by position from `flat`.'''

T2_CODE = r'''# TODO — fit a classifier on the FLAT features and compare Ada's vs Bo's predicted probability
from sklearn.linear_model import LogisticRegression

# attach the label (it lives in the customers dimension) to the flat design matrix
dm = flat.merge(customers[["customer_id", "churn"]], on="customer_id")
X = dm[AGG_COLS]
y = dm["churn"]
clf = ____                                        # any deterministic sklearn classifier
clf.fit(X, y)

proba = clf.predict_proba(X)[:, 1]                 # P(churn=1) per customer, in dm's row order
p = dict(zip(dm["customer_id"], proba))
print(f"P(churn) — Ada = {p['Ada']:.6f}   Bo = {p['Bo']:.6f}")
print(f"true churn — Ada = {int(customers.set_index('customer_id').loc['Ada','churn'])}   "
      f"Bo = {int(customers.set_index('customer_id').loc['Bo','churn'])}")'''

T2_SOL = T2_CODE.replace(
    'clf = ____                                        # any deterministic sklearn classifier',
    'clf = LogisticRegression(max_iter=1000)           # any deterministic sklearn classifier')

T2_CHECK = r'''# CHECK — do not edit
assert abs(p["Ada"] - p["Bo"]) < 1e-9, "identical feature rows MUST yield an identical predicted probability."
_true_ada = int(customers.set_index("customer_id").loc["Ada", "churn"])
_true_bo  = int(customers.set_index("customer_id").loc["Bo",  "churn"])
assert _true_ada != _true_bo, "the two customers genuinely differ in the label the flat table cannot express."
print("Task 2 ok — the model gives Ada and Bo the SAME P(churn) though their true labels differ (0 vs 1). "
      "No tuning can fix an identical input; the signal was destroyed by the flatten, not by the model.")'''


# ---- Task 3: recover the lost structure ----
T3_MD = r'''## Task 3 — recover the discarded structure (crucial fragment) — TODO

**Goal:** add two **structure-aware** features that a graph would keep for free, and show Ada and Bo now
differ:
- `spend_trend` = (amount of the customer's **last** pre-`t` order) − (amount of their **first**). Needs the
  **temporal ORDER** the flatten threw away.
- `n_distinct_products` = number of distinct `product_id` a customer bought, via a **two-hop** join
  customer → orders → order_items. Needs **event IDENTITY** and a higher-order path.

**Why it matters:** each recovered feature restores one dimension the aggregation destroyed — order, then
identity. That is exactly the fine-grain signal Fey issue (4) names. But notice: each is a bespoke,
hand-written, per-task feature you had to *think of*; the space of such collisions is unbounded. Breaking
one collision by hand is the treadmill; learning over the PK/FK graph (RDL) addresses the whole class.

**Hint boundary:** for `spend_trend`, sort each customer's pre-`t` orders by `order_ts` and take
`amount.iloc[-1] - amount.iloc[0]` per group. For `n_distinct_products`, inner-join `order_items` to the
pre-`t` orders on `order_id`, then `groupby("customer_id")["product_id"].nunique()`.'''

T3_CODE = r'''# TODO — fill the two blanks (recover ORDER, then IDENTITY)
past = orders[orders["order_ts"] < t].sort_values("order_ts")

# (a) spend_trend: last minus first order amount, per customer (recovers TEMPORAL ORDER)
trend = (past.groupby("customer_id")["amount"]
             .agg(lambda s: ____)                 # last order amount MINUS first order amount
             .rename("spend_trend").reset_index())

# (b) n_distinct_products: two-hop customer -> orders -> order_items (recovers EVENT IDENTITY)
past_items = order_items.merge(past[["order_id", "customer_id"]], on="order_id", how="inner")
prod = (past_items.groupby("customer_id")["product_id"]
                  .agg(____)                        # count of DISTINCT products
                  .rename("n_distinct_products").reset_index())

rich = flat.merge(trend, on="customer_id", how="left").merge(prod, on="customer_id", how="left")
rich[["spend_trend", "n_distinct_products"]] = rich[["spend_trend", "n_distinct_products"]].fillna(0)
print(rich[["customer_id"] + AGG_COLS + ["spend_trend", "n_distinct_products"]].to_string(index=False))'''

T3_SOL = (T3_CODE
          .replace('             .agg(lambda s: ____)                 # last order amount MINUS first order amount',
                   '             .agg(lambda s: s.iloc[-1] - s.iloc[0])  # last order amount MINUS first order amount')
          .replace('                  .agg(____)                        # count of DISTINCT products',
                   '                  .agg("nunique")                   # count of DISTINCT products'))

T3_CHECK = r'''# CHECK — do not edit
_r = rich.set_index("customer_id")
assert _r.loc["Ada", "spend_trend"] == 40 and _r.loc["Bo", "spend_trend"] == -40, \
    "spend_trend should be +40 for rising Ada and -40 for falling Bo (recovers temporal order)."
assert int(_r.loc["Ada", "n_distinct_products"]) == 3 and int(_r.loc["Bo", "n_distinct_products"]) == 1, \
    "n_distinct_products should be 3 for Ada and 1 for Bo (recovers event identity)."
_new = ["spend_trend", "n_distinct_products"]
assert not (_r.loc["Ada", _new].values == _r.loc["Bo", _new].values).all(), \
    "with the structure-aware features, Ada and Bo must no longer collide."
print("Task 3 ok — spend_trend (+40 vs -40) recovers TEMPORAL ORDER; n_distinct_products (3 vs 1) recovers "
      "EVENT IDENTITY. Ada and Bo are separable again — but each feature was hand-written for this one task.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the colliding pair on the flat features (identical rows), the model's
identical prediction for them, the recovered structure-aware features that break the tie, and a one-line
takeaway.

**Takeaway prompt:** in one sentence, name (a) the structure the flatten destroyed to cause the collision,
(b) which recovered feature restores which dimension, and (c) why hand-writing these features is a treadmill
that relational deep learning proposes to end by learning over the PK/FK graph.'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 035 (what joins destroy) ===")
print("\nThe collision on flat features (Ada vs Bo are IDENTICAL):")
print(flat[flat.customer_id.isin(["Ada","Bo"])][["customer_id"] + AGG_COLS].to_string(index=False))
print(f"\nFitted model P(churn): Ada = {p['Ada']:.4f}, Bo = {p['Bo']:.4f}  (identical, yet true labels 0 vs 1)")
print("\nAfter recovering structure (order + identity), they separate:")
print(rich[rich.customer_id.isin(["Ada","Bo"])][["customer_id","spend_trend","n_distinct_products"]].to_string(index=False))
print("\ntakeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    'print("\\ntakeaway:", "____")',
    'print("\\ntakeaway:", "Join+aggregate is a lossy map: count/sum/avg/max are order-blind and '
    'identity-blind, so Ada (rising, 3 products) and Bo (falling, 1 product) collapse to the identical row '
    'n=3/total=90/avg=30/max=50 and a fitted model must give them the same prediction though their true '
    'labels differ; spend_trend (+40 vs -40) restores the discarded TEMPORAL ORDER and n_distinct_products '
    '(3 vs 1) restores EVENT IDENTITY via a two-hop join, but each is a bespoke, per-task, leak-checked '
    'feature and the space of collisions is unbounded -- which is why relational deep learning keeps the '
    'database as its PK/FK graph (row=node, foreign key=edge) and learns the aggregations end-to-end rather '
    'than hand-writing one aggregate per lost dimension forever.")')


STRETCH = r'''# STRETCH (optional, ungraded) — the treadmill made visible.
# You broke the Ada/Bo collision with two features. But count/sum/avg/max collide on MANY histories.
# Below: a THIRD customer whose orders are yet another permutation/composition still colliding on the
# original four aggregates — show it also needs NEW features to separate, and that there is no finite set
# that pre-empts every collision.
extra_orders = pd.DataFrame({
    "order_id": ["z1","z2","z3"], "customer_id": ["Zoe","Zoe","Zoe"],
    "order_ts": pd.to_datetime(["2024-02-01","2024-03-01","2024-04-01"]),
    "amount": [30, 10, 50],   # different order AGAIN, but n=3, total=90, avg=30, max=50 -> same collision
})
combo = pd.concat([orders, extra_orders], ignore_index=True)
flat_z = flatten(combo, t).set_index("customer_id").loc[["Ada","Bo","Zoe"], AGG_COLS]
print(flat_z.to_string())
print("\nAll three collide on the flat four. Each new history is a new collision needing a new bespoke "
      "feature; the flatten cannot be patched into completeness. RDL learns over the graph so the structure "
      "(order, identity, cardinality, multi-hop paths) is never discarded in the first place.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 035 — What joins destroy: build an aggregation collision, then break it

**Lesson:** [`lessons/0035-what-joins-destroy.html`](../lessons/0035-what-joins-destroy.html) · **Phase / Year:** Year 1 · Q4

**Primary reading:** Fey et al. — [*Relational Deep Learning: Graph Representation Learning on Relational Databases*](https://arxiv.org/abs/2312.04615) (2024), **§1–2** (the five issues with manual feature engineering; the relational entity graph). ★ preview.

**Dataset tier:** **C** (small SYNTHETIC tables — the mechanism, not a benchmark). No download; runs in seconds.

**Implementation scope (standard #18):** Fey 2024 is a *position* preview, not an architecture, so instead of building a model you **demonstrate the paper's issue (4)** — "forcing data into a single table aggregates into lower-granularity features, thus losing valuable fine-grain signal" — as a concrete, runnable **aggregation collision**, then recover the discarded structure by hand and see why that doesn't scale.

**Skill you are practising:** enumerate the structure a flatten-then-aggregate pipeline discards (cardinality, identity, temporal order, multi-hop paths) and recognise an aggregation collision.

**Exit criteria:** EXIT TICKET prints the colliding pair (identical flat rows), the model's identical prediction for them, the structure-aware features that break the tie, and your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (the toy database + labels); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
`pandas` + `numpy` + `scikit-learn` (all in `requirements-labs.txt`). One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. No dataset is fetched — the tables are defined inline.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — aggregation is a lossy map

L034 built the flatten: pick a **grain** (one customer as of `t`), **join** the neighbour tables via foreign
keys, and **aggregate** the one-to-many rows into fixed-width columns. This lab asks the next question: *what
did that aggregation throw away?*

An aggregate is a **projection**. It maps a variable-length, structured neighbourhood (a customer's orders,
each with a time, an amount, and a product) down to a few fixed numbers (`count`, `sum`, `avg`, `max`).
Projections are **lossy**: many different neighbourhoods map to the same output. The sharpest proof is an
**aggregation collision** — two genuinely different entities producing the *identical* feature row, so no
model on the flat table can tell them apart.

Fey et al. (2024, §2) list five problems with the manual flatten; the load-bearing one here is issue **(4)**:
"by forcing data into a single table, information is aggregated into lower-granularity features, **thus losing
out on valuable fine-grain signal.**" The structure lost has names: **cardinality** (how many), **identity**
(which), **temporal order** (the sequence), and **higher-order paths** (signal ≥2 hops away).

**Toy micro-example (not this lab's data).** Two customers, orders `[$10, $50]` vs `[$50, $10]`. Same `sum`
(60), same `mean` (30), same `max` (50) — a collision on those three aggregates — yet one is rising and one
is falling. Only a feature that keeps **order** (e.g. last − first = +40 vs −40) can tell them apart.

Full write-up + the interactive collision visualisation: [Lesson 035](../lessons/0035-what-joins-destroy.html).'''),
        md("## Setup — PROVIDED (the toy 3-table database + churn labels + cutoff t)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — the treadmill

You broke *one* collision with two features. But `count`/`sum`/`avg`/`max` collide on unboundedly many
histories. Below, a third customer with yet another order composition still lands on the same four
aggregates — showing you cannot patch the flatten into completeness. Each new history is a new collision
needing a new hand-written feature; learning over the PK/FK graph is the alternative that keeps the
structure instead of chasing losses one at a time.'''),
        code(STRETCH),
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
    with open(os.path.join(HERE, "0035-what-joins-destroy.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0035-what-joins-destroy.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0035-what-joins-destroy.ipynb + solution")


if __name__ == "__main__":
    main()
