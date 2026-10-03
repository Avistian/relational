"""Three-seed CPU block experiment; no dataset fitting or benchmark predictions."""
import importlib.util,json,sys
from pathlib import Path
import torch
from relkit.baselines_b11 import eligible
P=Path(__file__).resolve().parent;S=P/'sources/b11';E=P/'evidence/b11'
def module(name):
    spec=importlib.util.spec_from_file_location(name,S/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def compare(a,b,args):
    b.load_state_dict(a.state_dict());a.eval();b.eval()
    aa=[v.detach().clone().requires_grad_(v.is_floating_point()) if isinstance(v,torch.Tensor) else v for v in args]
    bb=[v.detach().clone().requires_grad_(v.is_floating_point()) if isinstance(v,torch.Tensor) else v for v in args]
    # GNN tuple inputs need independent leaves too.
    aa=[tuple(t.detach().clone().requires_grad_() for t in v) if isinstance(v,tuple) else v for v in aa]
    bb=[tuple(t.detach().clone().requires_grad_() for t in v) if isinstance(v,tuple) else v for v in bb]
    # Edge tuples are nested in a list to avoid treating integer edges as features.
    aa=[tuple(v) if isinstance(v,list) else v for v in aa];bb=[tuple(v) if isinstance(v,list) else v for v in bb]
    x=a(*aa);y=b(*bb);xs=x if isinstance(x,tuple) else (x,);ys=y if isinstance(y,tuple) else (y,)
    out=max(float((u-v).abs().max().detach()) for u,v in zip(xs,ys))
    for u,v in zip(xs,ys):torch.testing.assert_close(u,v,atol=1e-9,rtol=0)
    sum(v.square().sum() for v in xs).backward();sum(v.square().sum() for v in ys).backward()
    grads=[]
    for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==m and (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=0);grads.append(float((p.grad-q.grad).abs().max()))
    for x,y in zip(aa,bb):
        for u,v in zip(x if isinstance(x,tuple) else (x,),y if isinstance(y,tuple) else (y,)):
            if isinstance(u,torch.Tensor) and u.is_floating_point() and u.grad is not None:
                torch.testing.assert_close(u.grad,v.grad,atol=1e-9,rtol=0);grads.append(float((u.grad-v.grad).abs().max()))
    return dict(output_error=out,gradient_error=max(grads),active_gradient_tensors=len(grads))
def run():
    torch.set_num_threads(1);G=module('relgnn_visible');GO=module('relgnn_original');T=module('relgt_visible');TO=module('relgt_original');rows=[]
    for seed in range(3):
        torch.manual_seed(seed)
        # Two products, three purchases, two customers. Each purchase has one product and customer.
        x=tuple(torch.randn(n,4,dtype=torch.float64) for n in [2,3,2])
        ea=torch.tensor([[0,1,2],[0,0,1]]);eg=torch.tensor([[0,1,0],[0,1,2]])
        for empty in [False,True]:
            a=G.RelGNNConv('dim-fact-dim',(4,4),4,2,'sum').double();b=GO.RelGNNConv('dim-fact-dim',(4,4),4,2,'sum').double()
            edges=[ea if not empty else ea[:,:0],eg]
            r=compare(a,b,[x,edges]);r.update(seed=seed,model='RelGNN composite',empty_attention=empty);rows.append(r)
        a=T.RelGTLayer(8,8,1,4,6,heads=2,conv_type='full',num_centroids=3,sample_node_len=7).double()
        b=TO.RelGTLayer(8,8,1,4,6,heads=2,conv_type='full',num_centroids=3,sample_node_len=7).double()
        # Populated buffers make the frozen global branch observable. One zero-count centroid is excluded.
        a.c_idx.copy_(torch.tensor([0,0,0,1,1,1]));a.vq._embedding_output.copy_(torch.randn(3,4,dtype=torch.float64))
        allrows=torch.cat(x);order=torch.tensor([[5,0,1,2,3,4,6],[6,0,1,2,3,4,5]])
        # Same seven input rows as the GNN. Duplicate four coordinates to fit two heads.
        # These are supplied encoded vectors, not the paper's five-component row encoder.
        encoded=torch.cat([allrows,allrows],dim=-1);tokens=encoded[order];root=encoded[5:];ids=torch.tensor([0,1])
        r=compare(a,b,[tokens,root,ids]);r.update(seed=seed,model='RelGT local+global',empty_attention=False);rows.append(r)
        with torch.no_grad():
            y=a(tokens,root,ids);perm=torch.tensor([0,3,1,2,6,5,4]);yp=a(tokens[:,perm],root,ids)
            torch.testing.assert_close(y,yp,atol=1e-9,rtol=0)
            r['neighbor_permutation_error']=float((y-yp).abs().max())
            # Same eligible raw rows feed both models; exclude both future and late-arriving records.
            events=torch.tensor([7.]*7+[12.,8.]);arrivals=torch.tensor([7.]*7+[9.,11.]);ok=eligible(events,arrivals,10.)
            raw=torch.cat([allrows,torch.randn(2,4,dtype=torch.float64)]);changed=raw.clone();changed[~ok]=1e6
            kept=changed[ok];encoded2=torch.cat([kept,kept],dim=-1)
            changed_y=a(encoded2[order],encoded2[5:],ids)
            torch.testing.assert_close(y,changed_y,atol=0,rtol=0)
            gn=G.RelGNNConv('dim-fact-dim',(4,4),4,2,'sum').double().eval()
            gy=gn(x,(ea,eg))[0];changed_gy=gn(tuple(kept.split([2,3,2])),(ea,eg))[0]
            torch.testing.assert_close(gy,changed_gy,atol=0,rtol=0)
            r['excluded_input_intervention_error']=float((y-changed_y).abs().max())
            # Reindex product rows and update FK source indices together.
            gp=gn((x[0].flip(0),x[1],x[2]),(ea,torch.stack([1-eg[0],eg[1]])))[0]
            torch.testing.assert_close(gy,gp,atol=1e-9,rtol=0)
            r['relgnn_row_permutation_error']=float((gy-gp).abs().max())
            # Frozen global summaries can carry information not present in a query's local sample.
            a.vq._embedding_output[0].add_(torch.tensor([2.,-3.,5.,1.],dtype=torch.float64))
            r['centroid_intervention_change']=float((a(tokens,root,ids)-y).abs().max())
            assert r['centroid_intervention_change']>1e-8
    result=dict(status='PASS',seeds=[0,1,2],cases=rows,tolerance=1e-9,dtype='float64',scope='Reduced block outputs and active input/parameter gradients against pinned sources. RelGT centroid updates and full row encoders are outside parity scope.',fresh_training='NOT_RUN')
    (E/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':run()
