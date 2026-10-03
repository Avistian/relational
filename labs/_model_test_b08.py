import torch, unittest
from relkit.limix_b08 import CourseModel
class Visibility(unittest.TestCase):
 def test_hidden_truth_and_query_batch(self):
  torch.manual_seed(8);m=CourseModel().eval();x=torch.randn(1,6,4);y=torch.randn(1,6);h=torch.zeros_like(x,dtype=torch.bool);h[:,3:,1]=True
  a,b=m(x,y,h,3)
  xx=x.clone();xx[h]=9999;yy=y.clone();yy[:,3:]=-9999
  aa,bb=m(xx,yy,h,3)
  torch.testing.assert_close(a,aa,rtol=0,atol=0);torch.testing.assert_close(b,bb,rtol=0,atol=0)
  c,d=m(x[:,:4],y[:,:4],h[:,:4],3)
  torch.testing.assert_close(a[:,:4],c,rtol=1e-5,atol=1e-6);torch.testing.assert_close(b[:,:4],d,rtol=1e-5,atol=1e-6)
  xx=x.clone();xx[:,4:]=1000;cc,dd=m(xx,y,h,3)
  torch.testing.assert_close(a[:,:4],cc[:,:4],rtol=0,atol=0)
  yy=y.clone();yy[:,:3]+=5;ee,_=m(x,yy,h,3)
  self.assertGreater((a[:,3:]-ee[:,3:]).abs().max().item(),1e-6)
if __name__=='__main__':unittest.main()
