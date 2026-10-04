"""Load-bearing contract tests, first executed against intentional stubs."""
import unittest
import numpy as np
from relkit.refinement_b22 import attention_read,replace_support,paired_effect,parameters,episodes,forward,experiment

class Contracts(unittest.TestCase):
 def test_attention_known_mean_and_stability(self):
  q=np.zeros((2,2));k=np.array([[1.,0.],[0.,1.]]);v=np.array([[2.,4.],[6.,8.]])
  np.testing.assert_allclose(attention_read(q,k,v),[[4,6],[4,6]],rtol=0,atol=1e-14)
  a=attention_read(np.array([[1000.,0.]]),k,v);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(a,[[2,4]])
  np.testing.assert_allclose(attention_read(q,k[::-1],v[::-1]),attention_read(q,k,v))
  with self.assertRaises(ValueError):attention_read(q,k,np.ones((3,2)))
 def test_support_only_replacement(self):
  before=np.arange(12.).reshape(4,3);after=before+np.arange(1,5)[:,None];b=before.copy();a=after.copy()
  np.testing.assert_array_equal(replace_support(before,after,2,'normal'),after)
  np.testing.assert_array_equal(replace_support(before,after,2,'identity'),after)
  skip=replace_support(before,after,2,'skip');np.testing.assert_array_equal(skip[:2],before[:2]);np.testing.assert_array_equal(skip[2:],after[2:])
  perm=replace_support(before,after,2,'permute');np.testing.assert_array_equal(perm[:2],before[:2]+[[2],[1]]);np.testing.assert_array_equal(perm[2:],after[2:])
  np.testing.assert_array_equal(before,b);np.testing.assert_array_equal(after,a)
  for n,m in [(0,'skip'),(4,'skip'),(2,'invalid')]:
   with self.assertRaises(ValueError):replace_support(before,after,n,m)
 def test_paired_metric_sign_and_units(self):
  y=np.array([0,1]);base=np.array([[2.,0.],[0.,2.]]);change=base[:,::-1]
  got=paired_effect(base,change,y);self.assertAlmostEqual(got['delta_ce'],2.);self.assertEqual(got['delta_accuracy_pp'],-100.)
  self.assertEqual(paired_effect(base,base,y)['delta_ce'],0.)
  self.assertAlmostEqual(paired_effect(base+10000,change+10000,y)['delta_ce'],2.)
  with self.assertRaises(ValueError):paired_effect(base,change,np.array([0,2]))

class Integration(unittest.TestCase):
 def test_complete_paired_state_contract(self):
  data=episodes();self.assertEqual(len(data),24);self.assertEqual(len({d['id'] for d in data}),24)
  for d in data:
   self.assertEqual(np.bincount(d['support_y']).tolist(),[6,6]);self.assertEqual(np.bincount(d['query_y']).tolist(),[8,8])
  d=data[0];w=parameters(0);a=forward(w,d['support_x'],d['support_y'],d['query_x'])
  identity=forward(w,d['support_x'],d['support_y'],d['query_x'],'identity')
  np.testing.assert_array_equal(a['logits'],identity['logits'])
  for mode in ['skip','permute']:
   b=forward(w,d['support_x'],d['support_y'],d['query_x'],mode)
   np.testing.assert_array_equal(a['states'][2][12:],b['states'][2][12:])
  changed=forward(w,d['support_x'],1-np.asarray(d['support_y']),d['query_x'])
  self.assertGreater(np.max(np.abs(np.array(a['logits'])-changed['logits'])),1e-8)
  r=experiment();self.assertEqual(len(r['conditions']),288);self.assertEqual(len(r['paired']),216)
  self.assertEqual(sum(len(x['logits']) for x in r['conditions']),4608)

if __name__=='__main__':unittest.main()
