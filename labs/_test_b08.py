"""Behavioral contracts: independent enumerated access and numeric loss."""
import unittest, torch
from relkit.limix_b08 import sample_visibility,feature_visibility,masked_mse
class Contracts(unittest.TestCase):
 def test_sample_access(self):
  self.assertEqual(sample_visibility(4,2).tolist(),[[1,1,0,0]]*4)
  with self.assertRaises(ValueError):sample_visibility(4,0)
 def test_feature_access(self):
  self.assertEqual(feature_visibility(2,2).tolist(),[[1,1,1,1],[1,1,1,1],[1,1,0,0],[1,1,0,0]])
 def test_loss_and_gradient(self):
  p=torch.tensor([1.,100.,5.],requires_grad=True); t=torch.tensor([3.,-100.,4.]);m=torch.tensor([1,0,1],dtype=torch.bool)
  loss=masked_mse(p,t,m);self.assertAlmostEqual(loss.item(),2.5);loss.backward()
  self.assertEqual(p.grad.tolist(),[-2.,0.,1.])
  with self.assertRaises(ValueError):masked_mse(p,t,~torch.ones(3,dtype=torch.bool))
if __name__=='__main__':unittest.main()
