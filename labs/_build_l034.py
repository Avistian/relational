"""Build Lab 034 (Relational data without RDL — flatten a 3-table schema, leak-free) — student + solution.

Tier C (three small SYNTHETIC tables: customers, orders, order_items). L034's "paper" is the Kimball
dimensional-modeling summary, not an architecture — so the implementation scope (standard #18) is to
OPERATIONALISE the flatten: take a relational schema and produce ONE leak-free design matrix at the
customer grain, and prove the point-in-time guard is load-bearing.

Three tasks:
  * Task 1 (crucial fragment) — point-in-time filter + entity-grain aggregation: keep only orders with
    order_ts < t, then groupby("customer_id") into n_orders / total_spend / avg_basket / max_basket.
  * Task 2 (crucial fragment) — the leak-free design matrix: LEFT JOIN the aggregates onto customers so
    zero-order customers survive, fillna(0), and confirm the grain is one row per customer.
  * Task 3 (crucial fragment) — prove the PIT guard matters: recompute the aggregate over ALL orders (no
    cutoff) and show the numbers change for a customer with a future order (leakage, live).

Deterministic synthetic data (no randomness, so CHECKs are stable). Cutoff t = 2024-06-01.
  C1: orders 40 (Jan), 55 (Mar), 30 (May) BEFORE t, plus 999 (Jul) AFTER t  -> safe n=3/total=125; leaky n=4/1124
  C2: 120 (Feb)                                                              -> n=1/120
  C3: 20 (Apr), 25 (May)                                                     -> n=2/45
  C4: only 500 (Aug, AFTER t)                                                -> n=0 (LEFT JOIN + fillna test)
  C5: no orders at all                                                       -> n=0
  C6: 80 (Jan), 80 (May)                                                     -> n=2/160

Run: .venv/bin/python labs/_build_l034.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0034-relational-data-without-rdl.ipynb \
    labs/solutions/0034-relational-data-without-rdl.ipynb
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


SETUP = r'''# PROVIDED — the toy relational database (Tier C, synthetic) and the cutoff. Just run.
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# The prediction time: features may use only information from BEFORE this instant.
t = pd.Timestamp("2024-06-01")

# --- customers dimension (one row per customer = the grain we want to predict at) ---
customers = pd.DataFrame({
    "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
    "segment":     ["retail", "retail", "smb", "retail", "smb", "retail"],
})

# --- orders fact/event table (one row per order; FK customer_id -> customers) ---
orders = pd.DataFrame({
    "order_id":    ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "o10"],
    "customer_id": ["C1", "C1", "C1", "C1", "C2", "C3", "C3", "C4", "C6", "C6"],
    "order_ts":    pd.to_datetime([
        "2024-01-10", "2024-03-05", "2024-05-20", "2024-07-02",   # C1: 3 before t, 1 AFTER t ($999)
        "2024-02-14",                                             # C2
        "2024-04-01", "2024-05-09",                              # C3
        "2024-08-01",                                            # C4: only AFTER t
        "2024-01-30", "2024-05-25",                             # C6
    ]),
    "amount":      [40, 55, 30, 999, 120, 20, 25, 500, 80, 80],
})
# C5 has NO orders at all (tests that a LEFT JOIN keeps it).

# --- order_items fact/detail table (one row per line item; FK order_id -> orders) — used in the stretch ---
order_items = pd.DataFrame({
    "item_id":    ["i1", "i2", "i3", "i4", "i5", "i6"],
    "order_id":   ["o1", "o1", "o2", "o5", "o6", "o9"],
    "product_id": ["pA", "pB", "pA", "pC", "pA", "pB"],
    "qty":        [1, 2, 1, 3, 1, 1],
})

print("customers:", customers.shape, "| orders:", orders.shape, "| order_items:", order_items.shape)
print("cutoff t =", t.date(), "— features may use only orders BEFORE this date")
orders'''


# ---- Task 1: PIT filter + entity-grain aggregation ----
T1_MD = r'''## Task 1 — point-in-time filter + entity-grain aggregation (crucial fragment) — TODO

**Goal:** write `aggregate_orders(orders, t)` that returns one row per customer summarising **only** that
customer's orders placed **strictly before** the cutoff `t`. Produce the columns `n_orders_before_t`,
`total_spend_before_t`, `avg_basket_before_t`, `max_basket_before_t`.

**Why it matters:** this is the flatten's core. The `order_ts < t` filter is the **point-in-time guard**
(L002/L021/L022, now at the feature step); the `groupby("customer_id")` fixes the **customer grain**. Get
either wrong and you either leak the future or produce order-level rows instead of one row per customer.

**Hint boundary:** filter `orders` to rows whose `order_ts` is **before** `t`; then `groupby("customer_id")`
and use `.agg(...)` with named outputs (`count` on `order_id`, `sum`/`mean`/`max` on `amount`). Return a
frame with `customer_id` as a column (`reset_index()`).'''

T1_CODE = r'''# TODO — fill the two blanks (the crucial fragment)
def aggregate_orders(orders, t):
    past = orders[____]                                  # keep only orders placed STRICTLY BEFORE t
    agg = (past.groupby(____)                            # one group per customer -> fixes the grain
                .agg(n_orders_before_t=("order_id", "count"),
                     total_spend_before_t=("amount", "sum"),
                     avg_basket_before_t=("amount", "mean"),
                     max_basket_before_t=("amount", "max"))
                .reset_index())
    return agg

agg = aggregate_orders(orders, t)
print(agg.to_string(index=False))'''

T1_SOL = (T1_CODE
          .replace('    past = orders[____]                                  # keep only orders placed STRICTLY BEFORE t',
                   '    past = orders[orders["order_ts"] < t]                 # keep only orders placed STRICTLY BEFORE t')
          .replace('    agg = (past.groupby(____)                            # one group per customer -> fixes the grain',
                   '    agg = (past.groupby("customer_id")                   # one group per customer -> fixes the grain'))

T1_CHECK = r'''# CHECK — do not edit
_c1 = agg.set_index("customer_id").loc["C1"]
assert int(_c1["n_orders_before_t"]) == 3, "C1 has 3 orders BEFORE t (the $999 Jul order is after t and must be excluded)."
assert int(_c1["total_spend_before_t"]) == 125, "C1 total before t should be 40+55+30 = 125 (not 1124)."
assert abs(float(_c1["avg_basket_before_t"]) - 125/3) < 1e-6, "C1 avg basket should be 125/3 ≈ 41.67."
assert int(_c1["max_basket_before_t"]) == 55, "C1 max basket before t should be 55 (not the future 999)."
assert "C4" not in agg["customer_id"].values, "C4's only order is AFTER t, so it has no pre-t rows to aggregate yet."
print("Task 1 ok — PIT filter + customer-grain aggregation correct. C1: n=3, total=125, avg≈41.67, max=55.")'''


# ---- Task 2: leak-free design matrix ----
T2_MD = r'''## Task 2 — the leak-free design matrix (crucial fragment) — TODO

**Goal:** write `build_design_matrix(customers, agg)` that LEFT-joins the aggregates onto the `customers`
table so **every** customer appears — including those with no pre-`t` orders — then fills the missing
aggregates with `0` (a customer with no orders has *count 0*, not *unknown*).

**Why it matters:** the join choice is a correctness decision. A plain (inner) join silently **drops**
customers like C4 (order only after `t`) and C5 (no orders) — a biased training set. `LEFT JOIN` + `fillna(0)`
keeps them and encodes "no history" honestly (contrast the missingness taxonomy, L006). The result is the
design matrix Q1–Q3 always assumed you had.

**Hint boundary:** `customers.merge(agg, on="customer_id", how=...)` with the option that keeps ALL left
rows; then fill the four aggregate columns' NaNs with `0`.'''

T2_CODE = r'''# TODO — fill the two blanks (the crucial fragment)
AGG_COLS = ["n_orders_before_t", "total_spend_before_t", "avg_basket_before_t", "max_basket_before_t"]

def build_design_matrix(customers, agg):
    flat = customers.merge(agg, on="customer_id", how=____)   # keep ALL customers, even zero-order ones
    flat[AGG_COLS] = flat[AGG_COLS].fillna(____)              # "no orders" -> 0, not missing
    return flat

flat = build_design_matrix(customers, agg)
print(flat.to_string(index=False))'''

T2_SOL = (T2_CODE
          .replace('    flat = customers.merge(agg, on="customer_id", how=____)   # keep ALL customers, even zero-order ones',
                   '    flat = customers.merge(agg, on="customer_id", how="left")  # keep ALL customers, even zero-order ones')
          .replace('    flat[AGG_COLS] = flat[AGG_COLS].fillna(____)              # "no orders" -> 0, not missing',
                   '    flat[AGG_COLS] = flat[AGG_COLS].fillna(0)                 # "no orders" -> 0, not missing'))

T2_CHECK = r'''# CHECK — do not edit
assert len(flat) == len(customers) == 6, "grain broken: the design matrix must have exactly one row per customer."
assert flat["customer_id"].nunique() == 6, "customer_id must be unique — one row per customer."
_f = flat.set_index("customer_id")
assert int(_f.loc["C4", "n_orders_before_t"]) == 0, "C4 (only a future order) must survive with n_orders_before_t = 0."
assert int(_f.loc["C5", "n_orders_before_t"]) == 0, "C5 (no orders at all) must survive with n_orders_before_t = 0."
assert int(_f.loc["C2", "total_spend_before_t"]) == 120, "C2 total should be 120."
print("Task 2 ok — one leak-free row per customer; C4 and C5 kept with 0-order aggregates. Grain: one row per customer as of t.")'''


# ---- Task 3: prove the PIT guard is load-bearing ----
T3_MD = r'''## Task 3 — prove the point-in-time guard is load-bearing (crucial fragment) — TODO

**Goal:** build a **leaky** version of the aggregate that ignores the cutoff (aggregates **all** of a
customer's orders, past and future), and compare it to the safe version for C1.

**Why it matters:** the schema gives you no protection — pandas/SQL will happily average future orders. The
only thing standing between you and leakage is the `order_ts < t` filter you wrote in Task 1. Removing it
should visibly change C1's numbers (the future $999 order leaks in), proving the guard is not decoration.

**Hint boundary:** reuse the Task 1 aggregation but on **all** `orders` (no time filter); then compare
`n_orders`/`total_spend` for C1 between the safe and leaky frames.'''

T3_CODE = r'''# TODO — build the LEAKY aggregate (no cutoff) and compare to the safe one for C1
def aggregate_orders_LEAKY(orders):
    agg_all = (____                                        # SAME groupby+agg as Task 1 but over ALL orders (no time filter)
                .agg(n_orders=("order_id", "count"),
                     total_spend=("amount", "sum"))
                .reset_index())
    return agg_all

leaky = aggregate_orders_LEAKY(orders).set_index("customer_id")
safe  = agg.set_index("customer_id")

print("C1 SAFE (order_ts < t):  n =", int(safe.loc["C1", "n_orders_before_t"]),
      " total =", int(safe.loc["C1", "total_spend_before_t"]))
print("C1 LEAKY (all orders):   n =", int(leaky.loc["C1", "n_orders"]),
      " total =", int(leaky.loc["C1", "total_spend"]))'''

T3_SOL = T3_CODE.replace(
    '    agg_all = (____                                        # SAME groupby+agg as Task 1 but over ALL orders (no time filter)',
    '    agg_all = (orders.groupby("customer_id")               # SAME groupby+agg as Task 1 but over ALL orders (no time filter)')

T3_CHECK = r'''# CHECK — do not edit
assert int(leaky.loc["C1", "n_orders"]) == 4, "leaky C1 should count all 4 orders (incl. the future $999)."
assert int(safe.loc["C1", "n_orders_before_t"]) == 3, "safe C1 should count only the 3 pre-t orders."
assert leaky.loc["C1", "total_spend"] != safe.loc["C1", "total_spend_before_t"], \
    "the leaky and safe totals MUST differ — that difference is the leaked future."
print(f"Task 3 ok — dropping the PIT filter changed C1 from n=3/total=125 to n=4/total="
      f"{int(leaky.loc['C1','total_spend'])}. The order_ts < t guard is load-bearing.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the flat design matrix, a one-sentence statement of its **grain**, the
safe-vs-leaky comparison that proves your point-in-time guard works, and a one-line takeaway.

**Takeaway prompt:** in one sentence, name the grain of your design matrix, the three moves that build it
(join, aggregate, point-in-time filter), and why this hand-built flatten is exactly what the relational
thesis proposes to replace.'''

EXIT_CODE = r'''# TODO: complete the grain sentence and the takeaway string
print("=== EXIT TICKET — Lesson 034 (relational data without RDL) ===")
print("\nFlat design matrix (one row per customer, as of t):")
print(flat.to_string(index=False))
print("\nGrain of this table:", "____")   # one sentence: what does ONE row mean?
print(f"\nPIT check — C1 safe total={int(agg.set_index('customer_id').loc['C1','total_spend_before_t'])} "
      f"vs leaky total={int(aggregate_orders_LEAKY(orders).set_index('customer_id').loc['C1','total_spend'])} "
      f"(the difference is leaked future spend).")
print("\ntakeaway:", "____")'''

EXIT_SOL = (EXIT_CODE
    .replace('print("\\nGrain of this table:", "____")   # one sentence: what does ONE row mean?',
             'print("\\nGrain of this table:", "one row = one customer summarised as of the cutoff t; the label would be read from the window AFTER t.")')
    .replace('print("\\ntakeaway:", "____")',
             'print("\\ntakeaway:", "The design matrix is built by choosing a grain (one customer as of t), '
             'LEFT-JOINing the orders fact table onto the customers dimension by the customer_id foreign key, '
             'AGGREGATING the one-to-many orders into fixed-width columns (count/sum/mean/max), and guarding '
             'every aggregate with order_ts < t so no feature uses the future -- a hand-written, per-task '
             'pipeline of DFS-style primitives that deliberately discards the relational structure (cardinality, '
             'order, identity, multi-hop paths), which is exactly what relational deep learning proposes to '
             'learn end-to-end over the PK/FK graph instead: the returns moved across the join.")'))


STRETCH = r'''# STRETCH (optional, ungraded) — a TWO-HOP feature: customer -> orders -> order_items.
# Reaching product-level signal is a multi-hop path (each hop a join). Count distinct products per customer,
# using ONLY items that belong to that customer's PRE-t orders (the PIT guard must survive the extra hop).
past_orders = orders[orders["order_ts"] < t][["order_id", "customer_id"]]
items_pit = order_items.merge(past_orders, on="order_id", how="inner")   # items on pre-t orders only
prod = (items_pit.groupby("customer_id")
                 .agg(n_distinct_products=("product_id", "nunique"))
                 .reset_index())
flat2 = flat.merge(prod, on="customer_id", how="left")
flat2["n_distinct_products"] = flat2["n_distinct_products"].fillna(0).astype(int)
print(flat2[["customer_id", "n_orders_before_t", "n_distinct_products"]].to_string(index=False))
print("\nEach hop (customer->orders->items) is another join to write AND keep leak-free — the manual flatten's "
      "cost grows with path length. L035 measures what even a rich set of these aggregates still throws away.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 034 — Relational data without RDL: flatten a 3-table schema, leak-free

**Lesson:** [`lessons/0034-relational-data-without-rdl.html`](../lessons/0034-relational-data-without-rdl.html) · **Phase / Year:** Year 1 · Q4

**Primary reading:** Kimball & Ross — [*Fact Tables and Dimension Tables*](https://www.kimballgroup.com/2003/01/fact-tables-and-dimension-tables/) (Kimball Group), and the [Dimensional Modeling Techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/) summary (from *The Data Warehouse Toolkit*, 3rd ed., Ch. 2).

**Dataset tier:** **C** (three small SYNTHETIC tables — the mechanism, not a benchmark). No download; runs in seconds.

**Implementation scope (standard #18):** the "paper" here is the Kimball dimensional-modeling summary, not an architecture — so instead of building a model you **operationalise the flatten**: turn a relational schema into ONE leak-free design matrix at the customer grain, and prove the point-in-time guard is load-bearing.

**Skill you are practising:** read a 3-table schema (fact + dimensions, PK/FK) and write the join + point-in-time aggregation that flattens it into one design matrix.

**Exit criteria:** EXIT TICKET prints the flat design matrix, a one-sentence statement of its grain, the safe-vs-leaky comparison proving your PIT guard works, and your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (the toy database, the cutoff); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
Only `pandas` + `numpy` (both in `requirements-labs.txt`). One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. No dataset is fetched — the tables are defined inline.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — schema, grain, join, aggregate, point-in-time

Real data lives in a **relational database**: several **tables** linked by **keys**. A **primary key (PK)**
uniquely names each row of a table (`customers.customer_id`); a **foreign key (FK)** in another table stores
that value to link back (`orders.customer_id`). In the Kimball picture, **fact/event** tables record things
that happened (an order, with a timestamp), and **dimension** tables describe the entities they point to (the
customer). Drawn out, one fact table ringed by its dimensions is a **star schema**.

The **grain** is what *one row* means. `orders` has grain "one order"; we want a training table at grain
**"one customer, as of a cutoff `t`"**. Because one customer has **many** orders (a *one-to-many* link), we
cannot just paste the orders in — we must **aggregate** them: collapse the many orders into a fixed set of
columns (`COUNT`, `SUM`, `AVG`, `MAX`). Aggregation is lossy by design: `AVG(amount)` keeps the mean and
forgets the individual baskets.

**Point-in-time (PIT) correctness** is the discipline that keeps the flatten honest: every feature for the
row at time `t` must use **only** data from **before** `t`. In code that is a single filter —
`orders[orders.order_ts < t]` — applied *before* the aggregation. Drop it and each feature silently averages
in the customer's *future* orders: label leakage (L002/L021/L022) committed at the feature step.

**Toy micro-example (not this lab's tables).** A customer with orders `[$10 (past), $90 (future)]` and cutoff
`t` between them: the *safe* `avg_basket_before_t` is `$10`; the *leaky* all-orders average is `$50`. Same
`AVG`, wildly different feature — the only difference is the PIT filter.

Full write-up + the interactive schema/flatten visualisations: [Lesson 034](../lessons/0034-relational-data-without-rdl.html).'''),
        md("## Setup — PROVIDED (the toy 3-table database + cutoff t)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — a two-hop feature (customer → orders → order_items)

The features that "still pay" (L033) are often deeper in the schema. Reaching product-level signal is a
**multi-hop** path: customer → orders → order_items → product. Each hop is another join that must *also* stay
point-in-time correct. Below, count the distinct products a customer bought, using only items on their pre-`t`
orders — and notice how the manual flatten's cost grows with every hop.'''),
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
    with open(os.path.join(HERE, "0034-relational-data-without-rdl.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0034-relational-data-without-rdl.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0034-relational-data-without-rdl.ipynb + solution")


if __name__ == "__main__":
    main()
