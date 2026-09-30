"""Independent source comparisons and reproducible release-defect probes."""
import contextlib,sys,types,copy,json
from pathlib import Path
import torch

@contextlib.contextmanager
def original_modules(root):
    names=['codebook','encoders','local_module','model'];old={n:sys.modules.get(n) for n in names}
    try:
        for n in names:
            m=types.ModuleType(n);sys.modules[n]=m;exec(compile((Path(root)/(n+'.py')).read_text(),n,'exec'),m.__dict__)
        yield sys.modules['model'].RelGT
    finally:
        for n,m in old.items():
            if m is None:sys.modules.pop(n,None)
            else:sys.modules[n]=m

def mechanism_check(source_root,visible_namespace=None):
    import numpy as np,pandas as pd,torch_frame
    from torch_geometric.data import HeteroData
    if visible_namespace is None:
        import relkit.relgt_l145 as visible
        visible_namespace=vars(visible)
    Visible=visible_namespace['RelGT'];torch.manual_seed(145);torch.set_num_threads(1)
    frame=torch_frame.data.Dataset(pd.DataFrame({'x':[1.,2.,3.,4.,5.,6.]}),col_to_stype={'x':torch_frame.numerical});frame.materialize()
    kw=dict(num_nodes=6,max_neighbor_hop=3,node_type_map={'row':0},col_names_dict={'row':frame.tensor_frame.col_names_dict},col_stats_dict={'row':frame.col_stats},local_num_layers=1,channels=16,out_channels=1,global_dim=8,heads=4,ff_dropout=.3,attn_dropout=.3,conv_type='full',num_centroids=8,sample_node_len=3)
    grouped={'grouped_tfs':{0:frame.tensor_frame},'grouped_indices':{0:list(range(6))},'flat_batch_idx':[0,0,0,1,1,1],'flat_nbr_idx':[0,1,2,0,1,2]}
    args=(torch.zeros(2,3,dtype=torch.long),torch.tensor([0,3]),torch.tensor([[0,1,2],[0,1,2]]),torch.tensor([[0.,2.,3.],[0.,1.,4.]]),grouped)
    extra={'edge_index':torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]]),'batch':torch.tensor([0,0,0,1,1,1])}
    with original_modules(source_root) as Original:
        a=Visible(**kw);b=Original(**kw);b.load_state_dict(a.state_dict())
        torch.manual_seed(77);pa=a(*copy.deepcopy(args),**extra);pa.square().sum().backward()
        torch.manual_seed(77);pb=b(*copy.deepcopy(args),**extra);pb.square().sum().backward()
        torch.testing.assert_close(pa,pb,rtol=1e-6,atol=1e-6);grads=0
        for (na,va),(nb,vb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if va.grad is not None:torch.testing.assert_close(va.grad,vb.grad,atol=1e-6,rtol=1e-6,equal_nan=True);grads+=1
        layer=sys.modules['local_module'].EncoderLayer(16,32,.3,.3,4).eval();x=torch.randn(2,3,16)
        p1=layer(x);p2=layer(x);dropout_difference=float((p1-p2).abs().max().detach());assert dropout_difference>0
        # Eval must not update global centroid state even though dropout and PE are random.
        a.eval();state={k:v.clone() for k,v in a.state_dict().items()};a(*copy.deepcopy(args),**extra)
        assert all(torch.equal(v,a.state_dict()[k]) for k,v in state.items())
    # Execute original local_nodes_hetero, replacing multiprocessing only.
    m=types.ModuleType('l145_original_utils');exec(compile((Path(source_root)/'utils.py').read_text().replace('from sentence_transformers import SentenceTransformer',''),'utils.py','exec'),m.__dict__)
    class SerialPool:
        def __init__(self,processes,initializer,initargs):initializer(*initargs)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,items):return list(map(fn,items))
    m.Pool=SerialPool
    graph=HeteroData();graph['driver'].num_nodes=1;graph['event'].num_nodes=2;graph['event'].time=torch.tensor([4.,9.])
    graph['driver','has','event'].edge_index=torch.tensor([[0,0],[0,1]])
    sampled=m.local_nodes_hetero(graph,3,('driver',torch.tensor([0,0])),torch.tensor([5.,10.]),num_workers=1)
    tokens=sampled['driver'][0][0];assert len(sampled['driver'])==1 and any(t[0]=='event' and t[1]==1 for t in tokens)
    # Fallback samples global nodes without cutoff filtering.
    isolated=HeteroData();isolated['driver'].num_nodes=1;isolated['event'].num_nodes=1;isolated['event'].time=torch.tensor([9.])
    m.init_worker_globals({'driver':[set()],'event':[set()]},[('event',0)])
    _,_,fallback,_=m._process_one_seed((isolated,2,'driver',0,5.,0));assert fallback[1][3]<0
    result=dict(status='PASS',full_model_output_max_error=float((pa-pb).abs().max().detach()),parameter_gradient_tensors=grads,eval_attention_dropout_max_difference=dropout_difference,centroids_frozen_during_eval=True,original_cache_overwrite_confirmed=True,original_unfiltered_fallback_confirmed=True,fixture_scope='synthetic mechanism and original-code failure probes, not benchmark performance')
    return result

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=mechanism_check(p/'sources/l145');(p/'evidence/l145/mechanism.json').write_text(json.dumps(r,indent=2));print(r)
