"""Independent GCN arithmetic, gradient masking and released-data checks."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from relkit.gcn_l082 import normalized_support, propagate, masked_objective, load_cora, GCN

def check():
    a=sp.csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]],dtype=float)
    s=normalized_support(a)
    expected=torch.tensor([[.5,6**-.5,0,0],[6**-.5,1/3,6**-.5,0],[0,6**-.5,.5,0],[0,0,0,1]],dtype=torch.float32)
    torch.testing.assert_close(s.to_dense(),expected)
    x=torch.tensor([[2.],[4.],[8.],[10.]])
    torch.testing.assert_close(propagate(s,x,torch.ones(1,1)),expected@x)
    assert not torch.allclose(expected.sum(1),torch.ones(4))
    p=np.array([2,0,3,1]);ss=normalized_support(a[p][:,p])
    torch.testing.assert_close(propagate(ss,x[p],torch.ones(1,1)),(expected@x)[p])
    z=torch.randn(4,2,requires_grad=True);w=torch.ones(3,2,requires_grad=True);y=torch.tensor([0,1,1,0])
    loss=masked_objective(z,y,torch.tensor([0,1]),w);loss.backward()
    assert torch.count_nonzero(z.grad[2:])==0
    torch.testing.assert_close(w.grad,.0005*w)
    y[2:]=1-y[2:]
    torch.testing.assert_close(loss,masked_objective(z,y,torch.tensor([0,1]),w))
    data=load_cora(Path(__file__).resolve().parent);x,s,y,tr,va,te=data
    assert x.shape==(2708,1433) and [len(tr),len(va),len(te)]==[140,500,1000]
    assert torch.bincount(y[tr]).tolist()==[20]*7
    assert len(set(tr.tolist()+va.tolist()+te.tolist()))==1640
    model=GCN(1433,7).eval()
    with torch.no_grad():
        actual=model(x,s);dense=s.to_dense()@torch.relu(s.to_dense()@x.to_dense()@model.w0)@model.w1
    error=float((actual-dense).abs().max());assert error<2e-6
    return {'status':'PASS','dense_sparse_max_error':error,'checks':['augmented degrees','isolated node self-loop','non-row-stochastic weights','node permutation','masked label intervention','exact L2 gradient','fixed split','dense forward oracle']}
if __name__=='__main__':
    r=check();Path(__file__).with_name('_verify_l082_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

# Independently execute the archived preprocessing code, then compare our loader.
x_t,s_t,y,tr,va,te=load_cora(Path(__file__).resolve().parent)
result=json.loads(Path(__file__).with_name("_verify_l082_results.json").read_text())
# Compare actual preprocessing to the pinned original functions, not a copied rewrite.
import ast,os,tempfile,scipy.sparse as sp,pickle as pkl,networkx as nx,sys,hashlib
lab=Path(__file__).resolve().parent
manifest=json.loads((lab/'_sources_l078.json').read_text())
for record in manifest['files']:
    assert hashlib.sha256((lab/record['path']).read_bytes()).hexdigest()==record['sha256']
source=(lab/'sources/l078/utils.py').read_text();tree=ast.parse(source)
ns={'np':np,'pkl':pkl,'sp':sp,'nx':nx,'sys':sys}
for node in tree.body:
    if isinstance(node,ast.FunctionDef):exec(ast.get_source_segment(source,node),ns)
with tempfile.TemporaryDirectory() as temp:
    (Path(temp)/'data').symlink_to(lab/'data/l078',target_is_directory=True)
    before=Path.cwd()
    try:
        os.chdir(temp);adj,features,yt,yv,ye,mt,mv,me=ns['load_data']('cora')
    finally:os.chdir(before)
def dense_tuple(t):
    coords,values,shape=t
    return sp.coo_matrix((values,(coords[:,0],coords[:,1])),shape=shape).toarray()
np.testing.assert_allclose(x_t.to_dense().numpy(),dense_tuple(ns['preprocess_features'](features)),atol=1e-7)
np.testing.assert_allclose(s_t.to_dense().numpy(),dense_tuple(ns['preprocess_adj'](adj)),atol=1e-7)
for idx,mask in [(tr,mt),(va,mv),(te,me)]:np.testing.assert_array_equal(idx.numpy(),np.flatnonzero(mask))
for labels,idx in [(yt,tr),(yv,va),(ye,te)]:np.testing.assert_array_equal(y[idx].numpy(),labels[idx].argmax(1))
result['checks'].extend(['all pinned source/data hashes','original release preprocessing executed','original release masks/labels'])
result['original_preprocessing_parity']='PASS';result['raw_self_edges']=int(adj.diagonal().sum())
Path(__file__).with_name('_verify_l082_results.json').write_text(json.dumps(result,indent=2)+'\n');print('Original release preprocessing and hashes PASS')
