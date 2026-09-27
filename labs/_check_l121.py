"""Live task checks using hand-derived counterexamples and SQLite oracle."""
import json,sqlite3
from pathlib import Path
import numpy as np
import torch

def check_rule(fn):
 rows=[(11,7,0),(12,7,1),(13,8,1)] # order, customer, late
 assert fn(rows,[7,8,9])==[True,True,False], 'An existential rule needs one matching child; empty is false.'
 assert fn([(11,7,0)],[7,8])==[False,False], 'Do not confuse existence of an order with existence of a late order.'
 assert fn(rows[::-1],[8,7])==[True,True], 'Query order matters, row order does not.'

def check_aggregate(fn):
 # id, customer, amount, event, available; the late-arriving row is excluded at day 5.
 rows=[(11,7,2.,2,2),(12,7,8.,4,7),(13,8,3.,3,3),(14,7,4.,4,4)]
 got=fn(rows,[8,7,9],5)
 assert np.allclose(got,[[1,3,3],[2,6,4],[0,0,0]]), 'Use both clocks, preserve root order, define empty aggregates.'
 assert np.allclose(fn(rows,[7],7),[[3,14,8]]), 'Late arrival becomes usable once both clocks permit it.'
 con=sqlite3.connect(':memory:');con.execute('CREATE TABLE events(id,c,amount,event,available)');con.executemany('INSERT INTO events VALUES(?,?,?,?,?)',rows)
 expected=[con.execute('SELECT COUNT(*),COALESCE(SUM(amount),0),COALESCE(MAX(amount),0) FROM events WHERE c=? AND event<=? AND available<=?',(c,5,5)).fetchone() for c in [8,7,9]]
 assert np.allclose(got,expected),'Independent SQL aggregate mismatch';con.close()

def check_path(fn):
 # leaf values [2,8] belong to separate orders, [5,5] to one order.
 x=torch.tensor([[2.],[8.],[5.],[5.]],requires_grad=True)
 leaf_order=torch.tensor([0,1,2,2]);order_root=torch.tensor([0,0,1]);out=fn(x,leaf_order,order_root,3,2)
 assert torch.allclose(out,torch.tensor([[68.],[100.]])), 'Sum within each order, square, then sum into roots.'
 out[0].sum().backward();assert torch.allclose(x.grad,torch.tensor([[4.],[16.],[0.],[0.]])), 'Supervision must follow the two-hop ownership path.'
 perm=torch.tensor([3,0,2,1]);assert torch.allclose(fn(x.detach()[perm],leaf_order[perm],order_root,3,2),out.detach()),'Permutation invariance failed.'

if __name__=='__main__':
 from relkit.history_l121 import exists_late,aggregate_at,path_signal
 for check,fn in [(check_rule,exists_late),(check_aggregate,aggregate_at),(check_path,path_signal)]:check(fn)
 print('PASS: rule, both-clock aggregates, differentiable two-hop path')
