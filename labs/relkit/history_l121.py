# %% Imports and scope
"""Original course computations, not implementations of full ILP or DFS search."""
import numpy as np
import torch

# %% Task 1: evaluate an existential relational rule
def exists_late(rows, roots):
    late_owners={customer for order,customer,late in rows if late==1}
    return [customer in late_owners for customer in roots]

# %% Task 2: construct query-time aggregate features
def aggregate_at(rows, roots, cutoff):
    result=[]
    for customer in roots:
        values=[amount for _,owner,amount,event,available in rows
                if owner==customer and event<=cutoff and available<=cutoff]
        result.append([len(values),sum(values),max(values,default=0)])
    return np.asarray(result,dtype=float).reshape(-1,3)

# %% Task 3: preserve the intermediate grouping in a two-hop computation
def path_signal(values, leaf_order, order_root, n_orders, n_roots):
    order_values=values.new_zeros((n_orders,values.shape[1]))
    order_values.index_add_(0,leaf_order,values)
    root_values=values.new_zeros((n_roots,values.shape[1]))
    root_values.index_add_(0,order_root,order_values.square())
    return root_values
