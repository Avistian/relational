"""Independent oracles, actual checkpoint parity and source artifact checks."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relkit.griffin_l164 import cell_attention,relation_pool,eligible_edges,GriffinMod,classification_logits
from _check_l164 import check_all,check_attention,check_relations,check_cutoffs
from _parity_l164 import parity
P=Path(__file__).resolve().parent;E=P/'evidence/l164';torch.set_num_threads(1)

def verify():
    check_all(cell_attention,relation_pool,eligible_edges);rng=np.random.default_rng(164)
    # Scalar-loop attention oracle (no matrix multiply or torch softmax).
    for _ in range(96):
        c=int(rng.integers(2,8));d=int(rng.integers(1,9));q=rng.normal(size=d);k=rng.normal(size=(c,d));v=rng.normal(size=(c,3));blocked=rng.random(c)<.3;blocked[0]=False
        scores=[sum(q[j]*k[i,j] for j in range(d))/math.sqrt(d) for i in range(c)]
        ex=[0 if blocked[i] else math.exp(scores[i]-max(s for i,s in enumerate(scores) if not blocked[i])) for i in range(c)]
        want=[sum(ex[i]*v[i,j] for i in range(c))/sum(ex) for j in range(3)]
        got=cell_attention(torch.tensor(q[None,None]),torch.tensor(k[None]),torch.tensor(v[None]),torch.tensor(blocked[None,None]))
        np.testing.assert_allclose(got.numpy()[0,0],want,atol=1e-12,rtol=1e-12)
    for _ in range(96):
        x=rng.normal(size=(8,4));edges=rng.integers(0,8,size=(2,20));rel=rng.integers(0,3,size=20);emb=rng.normal(size=(3,4));wanted=np.zeros_like(x)
        for receiver in range(8):
            messages=[]
            for relation in range(3):
                neighbours=[int(edges[1,i]) for i in range(20) if edges[0,i]==receiver and rel[i]==relation]
                if neighbours:messages.append([sum(x[n,j] for n in neighbours)/len(neighbours)*emb[relation,j] for j in range(4)])
            if messages:wanted[receiver]=[max(m[j] for m in messages) for j in range(4)]
        got=relation_pool(torch.tensor(x),torch.tensor(edges),torch.tensor(rel),torch.tensor(emb)).numpy();np.testing.assert_allclose(got,wanted,atol=1e-12)
    temporal_cases=0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                times=torch.tensor([[a,b,c,-1],[a,b,c,-1]])
                for cutoff in [torch.tensor([1,3]),torch.tensor([3,1])]:
                    want=torch.tensor([[t!=-1 and t<int(co) for t in row] for row,co in zip(times,cutoff)])
                    assert torch.equal(eligible_edges(times,cutoff),want);temporal_cases+=1
    assert eligible_edges(torch.tensor([[-10,-1,0]]),torch.tensor([-2])).tolist()==[[True,False,False]]
    # Reject three plausible wrong learner solutions.
    wrong=[(check_attention,lambda q,k,v,blocked=None,**kw:torch.softmax(q@k.transpose(-1,-2),-1)@v),
           (check_relations,lambda x,ei,r,e:torch.zeros_like(x)),
           (check_cutoffs,lambda t,c:(t>=0)&(t<c.max()))]
    for check,fn in wrong:
        try:check(fn)
        except (AssertionError,RuntimeError):pass
        else:raise AssertionError('Incorrect learner implementation passed')
    tiny=parity()
    # Check actual released512-wide checkpoint on a real two-query sampled batch.
    sys.path.insert(0,str(P/'sources/l164/upstream'))
    from hmodel import GriffinMod as Source
    from hdataset import Graph,Task
    from hFloatEmb import SimpleRepeater
    from hloaderwrapper import LoaderWrapperTask
    from safetensors.torch import load_file
    root=Path('/tmp/l164-release');os.chdir(root);torch.manual_seed(164)
    graph=Graph(root/'data');tasks=Task(root/'data')
    ds=LoaderWrapperTask(graph,256,False,dict(floatemb=SimpleRepeater(512),fanout=20,hop=2),tasks,['rel-f1-driver-dnf'],'valid',3)
    ds.ind=torch.tensor([[0,0,1]]);batch=ds[0]
    original=Source(hiddim=512,num_mp=4,use_rev=True,use_gate=False).eval();our=GriffinMod().eval()
    weights=load_file(str(root/'checkpoint/model.safetensors'));original.load_state_dict(weights,strict=True);our.load_state_dict(weights,strict=True)
    captured={}
    def trace(module,args):
        if 'q' in captured:return
        q,k,v=args;wq,wk,wv=module.in_proj_weight.detach().chunk(3)
        def project(x,w):return F.linear(x[:1].detach(),w).reshape(1,-1,8,64).transpose(1,2)
        captured.update(q=project(q,wq),k=project(k,wk),v=project(v,wv))
    hook=original.nodefeataggr[1].crossattention.register_forward_pre_hook(trace)
    with torch.no_grad():
        expected=original(*copy.deepcopy(batch[:-3]))[batch[-1]]@batch[-2].T
        got=classification_logits(our,copy.deepcopy(batch))
    hook.remove();torch.testing.assert_close(got,expected,atol=2e-4,rtol=1e-6)
    real_error=(got-expected).abs().max().item()
    real={k:v.numpy().tolist() for k,v in captured.items()};real['expected']=F.scaled_dot_product_attention(**{'query':captured['q'],'key':captured['k'],'value':captured['v']}).numpy().tolist()
    real['note']='First validation root, second layer, eight heads, actual released Others-2 checkpoint; projected Q/K/V before output projection/gating. Replaying this is not fresh model evaluation.'
    (E/'attention-trace.json').write_text(json.dumps(real)+'\n')
    # Confirm default notebook's full model forward/backward calls both live tasks.
    torch.manual_seed(164);model=GriffinMod(hiddim=16,num_mp=4).double()
    node=[(torch.randn(3,16,dtype=torch.float64),torch.randn(4,3,16,dtype=torch.float64))]
    data=[node,[None],[torch.randn(16,dtype=torch.float64)],torch.tensor([[0,0,1,2],[1,2,2,3]]),torch.tensor([0,1,0,1]),torch.randn(2,16,dtype=torch.float64)]
    out=model(*data);out.square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for item in json.loads((P/'sources/l164/source-ledger.json').read_text())['files']:
        assert hashlib.sha256((P/'sources/l164'/item['file']).read_bytes()).hexdigest()==item['sha256']
    r=dict(status='PASS',attention_oracle_cases=96,relation_oracle_cases=96,temporal_oracle_cases=temporal_cases+1,incorrect_implementations_rejected=3,source_parity=tiny,real_checkpoint_logit_max_abs=real_error,real_checkpoint_queries=2,real_checkpoint_metric='NOT_COMPUTED: tiny parity fixture only',gradient_finiteness='PASS',cloud_additional_usd=0,learner='PENDING_WRITTEN_DEFENSE')
    (P/'_verify_l164_results.json').write_text(json.dumps(r,indent=2)+'\n');return r

if __name__=='__main__':print(verify())
