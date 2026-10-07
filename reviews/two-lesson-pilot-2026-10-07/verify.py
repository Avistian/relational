"""Independent loop oracles and deliberately broken implementations, no training."""
import importlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'labs'))

def verify81():
    from relkit import mpnn_l081 as m
    from _build_l081 import CHECKS
    torch.set_num_threads(1)
    rng=np.random.default_rng(81)
    for _ in range(40):
        n,d,e=7,3,int(rng.integers(0,25))
        h=rng.normal(size=(n,d)).astype('float32');edges=rng.integers(n,size=(2,e))
        incoming=np.zeros_like(h);counts=np.zeros(n)
        for src,dst in edges.T:incoming[dst]+=h[src];counts[dst]+=1
        want=(h+incoming/np.maximum(counts,1)[:,None])/2
        np.testing.assert_allclose(m.mean_step(torch.tensor(h),torch.tensor(edges)),want,atol=2e-7,rtol=2e-6)
    def run(name,fn):
        ns={'torch':torch,'aggregate':m.aggregate,'mean_step':m.mean_step,'graph_sum':m.graph_sum}
        ns[name]=fn;exec(CHECKS[name],ns)
    for name in CHECKS:run(name,getattr(m,name))
    def inplace(h,e):
        for v in range(len(h)):
            src=e[0,e[1]==v]
            h[v]=.5*h[v]+(.5*h[src].mean(0) if len(src) else 0)
        return h
    mutants=[('mean_step','reverse edges',lambda h,e:m.mean_step(h,e.flip(0))),
             ('mean_step','in-place updates',inplace),
             ('aggregate','sum used for mean',lambda x,d,n,reduction='sum':m.aggregate(x,d,n,'sum')),
             ('aggregate','all messages to node 1',lambda x,d,n,reduction='sum':m.aggregate(x,torch.ones_like(d),n,reduction)),
             ('graph_sum','pool all graphs',lambda h,b:h.sum(0,keepdim=True)),
             ('graph_sum','pool by contiguous positions',lambda h,b:m.graph_sum(h,b.sort().values))]
    rejected=[]
    for name,label,fn in mutants:
        try:run(name,fn)
        except (AssertionError,RuntimeError):rejected.append(label)
        else:raise AssertionError('Surviving mutant: '+label)
    return {'lesson':81,'random_directed_multifeature_oracles':40,'rejected_mutants':rejected,'status':'PASS'}

def verify82():
    import scipy.sparse as sp
    from relkit import gcn_l082 as m
    from _build_l082 import CHECKS
    torch.set_num_threads(1)
    rng=np.random.default_rng(82)
    for _ in range(40):
        adjacency=np.triu(rng.integers(0,2,size=(7,7)),1);adjacency=adjacency+adjacency.T
        augmented=adjacency+np.eye(7);degree=augmented.sum(1)
        h=rng.normal(size=(7,2)).astype('float32');w=rng.normal(size=(2,3)).astype('float32')
        expected=np.zeros((7,3))
        # Explicit sum over source nodes AND feature coordinates, no S@H@W oracle.
        for v in range(7):
            for u in range(7):
                for c in range(3):
                    for f in range(2):
                        expected[v,c]+=augmented[v,u]/np.sqrt(degree[v]*degree[u])*h[u,f]*w[f,c]
        got=m.propagate(m.normalized_support(sp.csr_matrix(adjacency)),torch.tensor(h),torch.tensor(w))
        np.testing.assert_allclose(got,expected,atol=3e-7,rtol=3e-6)
    def run(name,fn):
        ns={'torch':torch,'np':np,'sp':sp,**{key:getattr(m,key) for key in CHECKS}}
        ns[name]=fn
        for key in CHECKS:exec(CHECKS[key],ns)
    for key in CHECKS:run(key,getattr(m,key))
    def rowmean(a):
        x=torch.tensor((a.toarray()+np.eye(a.shape[0])),dtype=torch.float32)
        return (x/x.sum(1)[:,None]).to_sparse()
    mutants=[('propagate','extra ReLU',lambda s,h,w:m.propagate(s,h,w).relu()),
             ('propagate','ignore channel weights',lambda s,h,w:torch.sparse.mm(s,h.to_dense())),
             ('normalized_support','row mean instead of symmetric normalization',rowmean),
             ('masked_objective','include held-out labels',lambda z,y,i,w:m.masked_objective(z,y,torch.arange(len(y)),w)),
             ('masked_objective','double regularization',lambda z,y,i,w:m.masked_objective(z,y,i,w)+.00025*w.square().sum())]
    rejected=[]
    for name,label,fn in mutants:
        try:run(name,fn)
        except (AssertionError,RuntimeError):rejected.append(label)
        else:raise AssertionError('Surviving mutant: '+label)
    # Analytic gradient from a one-node cross entropy, independently derived.
    a=sp.csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]])
    h=torch.tensor([[.2],[.4],[.8],[1.]],requires_grad=True)
    z=m.propagate(m.normalized_support(a),h,torch.tensor([[1.,-1.]]))
    m.masked_objective(z,torch.tensor([0,1,0,1]),torch.tensor([1]),torch.zeros(1,2)).backward()
    b=(.2+.8)/np.sqrt(6)+.4/3
    probability=1/(1+np.exp(-2*b))
    expected=2*probability*np.array([1/np.sqrt(6),1/3,1/np.sqrt(6),0])
    np.testing.assert_allclose(h.grad[:,0],expected,atol=1e-7)
    return {'lesson':82,'random_graph_multichannel_oracles':40,'analytic_input_gradient':expected.tolist(),'rejected_mutants':rejected,'status':'PASS'}

if __name__=='__main__':
    result=globals()['verify'+sys.argv[1]]()
    Path(__file__).with_name('l'+sys.argv[1]+'-independent.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
