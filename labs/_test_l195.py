"""Behavioral contracts: keyed losses, practical margins, and evidence scope."""
import unittest
from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope
class Contracts(unittest.TestCase):
 def test_pair(self):
  t=[(1,10,2.),(1,20,4.)];a=[(1,20,5.),(1,10,2.)];b=[(1,10,4.),(1,20,4.)]
  r=paired_mae(t,a,b);self.assertEqual(r['advantage'],[2.,-1.]);self.assertEqual(r['candidate_mae'],.5);self.assertEqual(r['baseline_mae'],1.)
  for bad in [a[:-1],a+[a[0]],[(1,20,float('nan')),(1,10,2.)],[(1,30,5.),(1,10,2.)]]:
   with self.assertRaises(ValueError):paired_mae(t,bad,b)
 def test_margin(self):
  for lo,hi,m,out in [(-.45,.09,0,'UNRESOLVED'),(.03,.08,.02,'ABOVE_MARGIN'),(-.08,-.03,.02,'BELOW_NEGATIVE_MARGIN'),(-.01,.01,.02,'WITHIN_MARGIN'),(0,.02,0,'UNRESOLVED')]:self.assertEqual(interval_verdict(lo,hi,m),out)
  self.assertEqual(interval_verdict(None,None,.1,False),'INCOMPLETE')
  for args in [(1,0,0),(.1,.2,-1),(float('nan'),1,0),(None,None,0),(.1,.2,True)]:
   with self.assertRaises(ValueError):interval_verdict(*args)
 def test_scope(self):
  e=dict(authenticated=True,complete=True,measured=True,comparable=True)
  self.assertEqual(claim_scope('pipeline',e),'SCOPED_COMPARISON')
  for missing in e:
   x=dict(e);x[missing]=False;self.assertEqual(claim_scope('pipeline',x),'INSUFFICIENT_EVIDENCE')
  for c in ['relational_signal','architecture_cause','general_superiority','undervaluation']:
   self.assertEqual(claim_scope(c,e),'NOT_ESTABLISHED')
  self.assertEqual(claim_scope('fresh_training',e),'NOT_RUN')
  with self.assertRaises(ValueError):claim_scope('all models fail',e)
if __name__=='__main__':unittest.main()
