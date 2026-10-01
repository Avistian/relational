"""Small deterministic mechanism lane; source/score claims remain separate."""
import math,json,torch

def worked_report(attention,pool,cutoffs,real_trace):
    q=torch.tensor([[[math.sqrt(2),0.]]],dtype=torch.float64)
    k=torch.eye(2,dtype=torch.float64)[None];v=torch.tensor([[[2.,0.],[0.,4.]]],dtype=torch.float64)
    a=attention(q,k,v);b=attention(q.flip(-1),k,v)
    torch.testing.assert_close(a,attention(q,k.flip(1),v.flip(1)))
    rows=torch.tensor([[0.,0.],[2.,4.],[4.,2.],[8.,1.],[20.,10.]],dtype=torch.float64)
    edges=torch.tensor([[0,0,0,0],[1,2,3,4]]);rel=torch.tensor([0,0,1,1]);emb=torch.tensor([[1.,1.],[.5,2.]],dtype=torch.float64)
    times=torch.tensor([[4,4,4,6]])
    out={}
    for time in [5,7]:
        keep=cutoffs(times,torch.tensor([time]))[0]
        out[str(time)]=pool(rows,edges[:,keep],rel[keep],emb)[0].tolist()
    real={n:torch.tensor(real_trace[n],dtype=torch.float32) for n in ['q','k','v','expected']}
    actual=attention(real['q'],real['k'],real['v'])
    torch.testing.assert_close(actual,real['expected'],atol=1e-5,rtol=1e-5)
    return dict(lane='SYNTHETIC_MECHANISM_AND_CACHED_REAL_ATTENTION',task_A=[round(x,9) for x in a.flatten().tolist()],task_B=[round(x,9) for x in b.flatten().tolist()],relation_messages=out,real_attention_shape=list(actual.shape),real_attention_replay='PASS',full_reproduction='INCOMPLETE_BUDGET_GATE',test_metric='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
