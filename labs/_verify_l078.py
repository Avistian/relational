"""Independent numerical and protocol checks for L078."""
import json
from pathlib import Path
import numpy as np
import torch
try:
    from relkit.message_passing import mean_neighbors, mean_step, gcn_support, load_cora, GCN
except ImportError:
    raise AssertionError('L078 message passing implementation is missing')
x=np.array([[2.],[4.],[8.],[10.]])
e=np.array([[0,1],[1,0],[1,2],[2,1]])
np.testing.assert_allclose(mean_neighbors(x,e),[[4],[5],[4],[0]])
np.testing.assert_allclose(mean_step(x,e),[[3],[4.5],[6],[5]])
p=np.array([2,0,3,1]);inv=np.argsort(p)
np.testing.assert_allclose(mean_step(x[p],inv[e]),mean_step(x,e)[p])
np.testing.assert_allclose(mean_neighbors(x,e[::-1]),mean_neighbors(x,e))
a=np.zeros((4,4));a[e[:,1],e[:,0]]=1
s=gcn_support(a)
expected=np.array([[.5,1/np.sqrt(6),0,0],[1/np.sqrt(6),1/3,1/np.sqrt(6),0],[0,1/np.sqrt(6),.5,0],[0,0,0,1]])
np.testing.assert_allclose(s,expected)
assert not np.allclose(s.sum(1),1),'GCN is not row mean'
np.testing.assert_allclose(s@x,[[1+4/np.sqrt(6)],[10/np.sqrt(6)+4/3],[4+4/np.sqrt(6)],[10]])
x_t,s_t,y,tr,va,te=load_cora()
assert x_t.shape==(2708,1433) and len(tr)==140 and len(va)==500 and len(te)==1000
assert len(set(tr.tolist())&set(va.tolist()))==0
assert len(set(tr.tolist())&set(te.tolist()))==0
assert len(set(va.tolist())&set(te.tolist()))==0
assert torch.bincount(y[tr]).tolist()==[20]*7
model=GCN(1433,7);model.eval()
with torch.no_grad():
    actual=model(x_t,s_t)
    dense_s=s_t.to_dense();dense_x=x_t.to_dense()
    oracle=dense_s@torch.relu(dense_s@dense_x@model.w0)@model.w1
err=(actual-oracle).abs().max().item();assert err<2e-6
result={'status':'PASS','mean_neighbors':[4,5,4,0],'mean_update':[3,4.5,6,5],'gcn_dense_sparse_max_error':err,'checks':['hand arithmetic','empty neighbors','edge ordering','node permutation equivariance','symmetric normalization','Cora fixed split','dense forward oracle']}
Path(__file__).with_name('_verify_l078_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
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
Path(__file__).with_name('_verify_l078_results.json').write_text(json.dumps(result,indent=2)+'\n');print('Original release preprocessing and hashes PASS')
