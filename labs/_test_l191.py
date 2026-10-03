"""Behavioral contracts: direction, missing coverage, oracle selection and ties."""
import unittest, math
from relkit.tracking_l191 import signed_gap, eligible, compare_pool, average_ranks
class Contracts(unittest.TestCase):
 def test_direction(self):
  self.assertAlmostEqual(signed_gap(84.59,70.87,'AUROC'),13.72)
  self.assertAlmostEqual(signed_gap(7.298,7.374,'MAE'),.076)
  for args in [(math.nan,1,'MAE'),(1,2,'accuracy')]:
   with self.assertRaises(ValueError):signed_gap(*args)
 def test_gate(self):
  ok=dict(access='open_code',family='foundation',protocol='reported_same_table')
  self.assertTrue(eligible(ok,'foundation'))
  for k,v in [('access','unverified'),('protocol','different'),('family','supervised')]:
   self.assertFalse(eligible(dict(ok,**{k:v}),'foundation'))
  self.assertFalse(eligible({},'foundation'))
 def test_single_and_oracle(self):
  rows={'a':[9,1],'b':[4,4],'partial':[10,None]}
  x=compare_pool([8,6],rows,'AUROC')
  self.assertEqual(x['single_methods'],['a']);self.assertEqual(x['oracle_values'],[9,4])
  self.assertEqual(x['excluded'],['partial']);self.assertEqual(x['single_gap'],2)
  self.assertEqual(x['oracle_gap'],.5)
 def test_ties_and_regression(self):
  x=compare_pool([2,4],{'a':[4,4],'b':[2,8]},'MAE',baseline=[2,4])
  self.assertEqual(x['single_methods'],['a','b']);self.assertEqual(x['oracle_gap'],0)
  self.assertEqual(average_ranks([1,1,3],False),[1.5,1.5,3.0])
 def test_missing_and_shapes(self):
  self.assertEqual(compare_pool([1,2],{},'AUROC')['status'],'NO_ELIGIBLE_COMPARATOR')
  for rows in [{'a':[1]},{'a':[1,float('inf')]}]:
   with self.assertRaises(ValueError):compare_pool([1,2],rows,'AUROC')
  with self.assertRaises(ValueError):compare_pool([1,2],{'a':[1,2]},'MAE',baseline=[0,1])
if __name__=='__main__':unittest.main()
