"""Behavioral tests: interventions, adjustment, and relational identity."""
import unittest
import numpy as np
import pandas as pd
from relkit.causal_l185 import intervene,adjusted_effect,assemble

class CausalContracts(unittest.TestCase):
 def test_do_replaces_action_and_reuses_noise(self):
  f=pd.DataFrame({'U':[0,0,1,1],'B':[0,1,0,1],'A':[0,1,0,1],'E':[.075,.075,.925,.925]})
  before=f.copy(deep=True)
  np.testing.assert_array_equal(intervene(f,action=0),[0,0,0,0])
  np.testing.assert_array_equal(intervene(f,action=1),[1,1,1,1])
  np.testing.assert_array_equal(intervene(f,badge=0),intervene(f,badge=1))
  pd.testing.assert_frame_equal(f,before)
 def test_adjustment_removes_composition_effect(self):
  rows=[]
  # Exact risk differences .2 in both strata; biased treatment allocation.
  for u,a,n,positive in [(0,0,80,8),(0,1,20,6),(1,0,20,12),(1,1,80,64)]:
   rows += [{'U':u,'A':a,'Y':int(i<positive)} for i in range(n)]
  f=pd.DataFrame(rows)
  self.assertAlmostEqual(adjusted_effect(f),.2)
  with self.assertRaises(ValueError):adjusted_effect(f[~((f.U==0)&(f.A==1))])
 def test_duplicate_and_missing_relationships_rejected(self):
  t={'companies':pd.DataFrame({'company_id':[0],'U':[1],'B':[1],'available_at':[0]}),
     'customers':pd.DataFrame({'customer_id':[3],'company_id':[0],'cutoff':[1],'split':['test']}),
     'actions':pd.DataFrame({'customer_id':[3],'A':[0],'action_at':[1]}),
     'outcomes':pd.DataFrame({'customer_id':[3],'Y':[1],'outcome_at':[2],'E':[.3]})}
  self.assertEqual(len(assemble(t)),1)
  for key in t:
   bad={k:v.copy() for k,v in t.items()};bad[key]=pd.concat([bad[key],bad[key]])
   with self.assertRaises(ValueError):assemble(bad)
  bad={k:v.copy() for k,v in t.items()};bad['actions']=bad['actions'].iloc[:0]
  with self.assertRaises(ValueError):assemble(bad)
  bad={k:v.copy() for k,v in t.items()};bad['companies']['available_at']=2
  with self.assertRaises(ValueError):assemble(bad)

if __name__=='__main__':unittest.main()
