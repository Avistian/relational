import unittest
import numpy as np
from relkit.cost_b09 import aligned_rmse,pareto_front,support_attention
class Contracts(unittest.TestCase):
 def test_alignment(self):
  self.assertAlmostEqual(aligned_rmse([4,7],[2,6],[7,4],[8,2]),2**.5)
  for ids,p in [([4,4],[2,6]),([4,9],[2,6]),([4,7],[2,float('nan')])]:
   with self.assertRaises(ValueError):aligned_rmse([4,7],[2,6],ids,p)
 def test_dominance(self):
  np.testing.assert_array_equal(pareto_front([[1,2,4],[2,3,5],[2,1,4],[1,2,4]]),[True,False,True,True])
 def test_visibility(self):
  q=np.array([[1.,0.]])
  k=np.array([[0.,1.],[0.,-1.],[100.,0.]])
  v=np.array([[2.],[6.],[999.]])
  np.testing.assert_allclose(support_attention(q,k,v,2),[[4.]])
  v[-1]= -1000
  np.testing.assert_allclose(support_attention(q,k,v,2),[[4.]])
if __name__=='__main__':unittest.main()
