"""Behavioral contracts: test the information boundary, not implementation text."""
import copy,unittest
import numpy as np
from relkit.flatten_b14 import flatten,standardize,predict,keyed_mse
class Contracts(unittest.TestCase):
 def test_time_visibility_and_late_arrival(self):
  entities={1:4.,2:-1.}
  # (row ID, entity, event time, available time, value).
  events=[(0,1,2,3,2.),(1,1,6,7,6.),(2,1,7,12,100.),(3,1,10,10,20.),(4,2,1,2,999.)]
  q=[(1,10),(1,14),(2,0)]
  np.testing.assert_equal(flatten(entities,events,q,10),[[4,2,4,6],[4,2,4,6],[-1,0,0,0]])
  # Adding a post-snapshot event must never change the fixed-snapshot features.
  np.testing.assert_equal(flatten(entities,events+[(5,1,11,12,-999.)],q,10),flatten(entities,events,q,10))
  # A repeated physical event is an input error, not an extra observation.
  with self.assertRaises(ValueError):flatten(entities,events+[events[0]],q,10)
 def test_query_and_entity_identity(self):
  with self.assertRaises(ValueError):flatten({1:0},[],[(1,2),(1,2)],3)
  with self.assertRaises(ValueError):flatten({1:0},[],[(9,2)],3)
 def test_support_only_normalization(self):
  s=np.array([[1.,5],[3.,5]]);q=np.array([[100.,5.]])
  a,b=standardize(s,q);np.testing.assert_allclose(a,[[-1,0],[1,0]]);np.testing.assert_allclose(b,[[98,0]])
  np.testing.assert_equal(standardize(s,np.array([[-999.,5.]]))[0],a)
 def test_predictors_hand_computed(self):
  # Centered labels, unit ridge penalty. Slope=2/(2+1); kernel symmetric.
  x=np.array([[-1.],[1.]]);y=np.array([-1.,1.]);q=np.array([[0.],[1.]])
  np.testing.assert_allclose(predict(x,y,q,'ridge'),[0,2/3],atol=1e-12)
  r=np.exp(-4);np.testing.assert_allclose(predict(x,y,q,'rbf'),[0,(1-r)/(2-r)],atol=1e-12)
  with self.assertRaises(ValueError):predict(x,y,q,'invented')
 def test_complete_keys_and_independent_order(self):
  k=[(1,10),(1,20)];rows=[{'key':[1,20],'prediction':5.},{'key':[1,10],'prediction':1.}]
  self.assertEqual(keyed_mse(k,[2,3],rows),2.5)
  for bad in [rows[:1],rows+[rows[0]],[dict(key=[9,20],prediction=1),rows[1]],[dict(key=[1,20],prediction=float('nan')),rows[1]]]:
   with self.assertRaises(ValueError):keyed_mse(k,[2,3],bad)
if __name__=='__main__':unittest.main()
