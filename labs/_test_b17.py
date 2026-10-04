"""Behavioral oracles, including single-key degeneracy and cache dependencies."""
import unittest
import torch
from relkit.representations_b17 import attention, aggregate_layers, cache_identity

class Boundaries(unittest.TestCase):
    def test_attention(self):
        q=torch.tensor([[0.,0.],[1.,-1.]],dtype=torch.float64,requires_grad=True)
        k=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True)
        v=torch.tensor([[2.,4.],[6.,8.]],dtype=torch.float64,requires_grad=True)
        actual=attention(q,k,v)
        self.assertTrue(torch.equal(actual[0],torch.tensor([4.,6.],dtype=torch.float64)))
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        for a,b in zip(torch.autograd.grad(actual.sum(),(q,k,v),retain_graph=True),torch.autograd.grad(expected.sum(),(q,k,v))):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(attention(q,k[:1],v[:1]),v[:1].expand(2,-1))
        with self.assertRaises(ValueError):attention(q,k[:0],v[:0])
    def test_aggregation(self):
        a=torch.tensor([[1.,2.]],dtype=torch.float64)
        b=torch.tensor([[3.,4.]],dtype=torch.float64)
        p=[torch.nn.Linear(2,2,bias=False).double() for _ in range(2)]
        f=torch.nn.Linear(2,2,bias=False).double()
        with torch.no_grad():
            p[0].weight.copy_(torch.eye(2));p[1].weight.copy_(2*torch.eye(2));f.weight.copy_(3*torch.eye(2))
        torch.testing.assert_close(aggregate_layers([a,b],p,f),torch.tensor([[21.,30.]],dtype=torch.float64))
        with self.assertRaises(ValueError):aggregate_layers([a],p,f)
    def test_cache(self):
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64);state={'w':torch.ones(2)}
        key=cache_identity(x,['s','q'],1,state,'numeric-v1')
        self.assertEqual(key,cache_identity(x.clone(),['s','q'],1,state,'numeric-v1'))
        for args in [(x+1,['s','q'],1,state,'numeric-v1'),(x,['q','s'],1,state,'numeric-v1'),(x,['s','q'],2,state,'numeric-v1'),(x,['s','q'],1,{'w':torch.zeros(2)},'numeric-v1'),(x,['s','q'],1,state,'numeric-v2')]:
            self.assertNotEqual(key,cache_identity(*args))
        with self.assertRaises(ValueError):cache_identity(x,['q','q'],1,state,'numeric-v1')
        with self.assertRaises(ValueError):cache_identity(x,['q'],1,state,'numeric-v1')

if __name__=='__main__':unittest.main()
