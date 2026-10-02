"""Controlled interventions; no benchmark training or synthetic-prior pretraining."""
import numpy as np
from relkit.composite_l182 import legal_fusion,attend,factorial_interaction

def mechanism182(fuse,attention,interaction):
    source=np.array([[1.,0.],[0.,1.]]);bridge=np.array([[0.,1.],[2.,0.],[99.,99.]])
    si=np.array([0,1,0]);di=np.array([0,0,0]);t=np.array([1.,2.,11.]);cut=np.array([10.]);q=np.array([[1.,0.]])
    args=(source,bridge,si,di,t,cut,np.eye(2),np.eye(2))
    z,d=fuse(*args);out,w=attention(q,z,d)
    states={}
    def record(name,a):
        m,owners=fuse(*a);v,weights=attention(q,m,owners)
        states[name]=dict(messages=m.tolist(),weights=weights.tolist(),output=v[0].tolist(),legal_messages=len(m))
    record('baseline',args)
    order=[1,0,2];a=list(args)
    for i in [1,2,3,4]:a[i]=a[i][order]
    record('permuted',a);np.testing.assert_allclose(states['permuted']['output'],out[0])
    a=list(args);a[1]=bridge.copy();a[1][2]=[-999,888];record('future_value_changed',a)
    np.testing.assert_allclose(states['future_value_changed']['output'],out[0])
    a=list(args)
    for i in [1,2,3,4]:a[i]=np.concatenate([a[i],a[i][:1]])
    record('duplicate_first_message',a)
    assert not np.allclose(states['duplicate_first_message']['output'],out[0])
    # Unrelated routes are excluded by route construction; merging is an explicit intervention.
    merged=np.vstack([z,[8.,0.]])
    mixed,mw=attention(q,merged,np.array([0,0,0]))
    states['merged_other_route']=dict(messages=merged.tolist(),weights=mw.tolist(),output=mixed[0].tolist(),legal_messages=3)
    # Fabricated metric rows illustrate a contrast only; never reported as trained results.
    illustrative=np.array([[.60,.65,.64,.66],[.61,.66,.65,.67]])
    return dict(scope='DETERMINISTIC_MECHANISM_ONLY',states=states,illustrative_scores=illustrative.tolist(),illustrative_interaction=interaction(illustrative).tolist(),hybrid_training='NOT_RUN')

if __name__=='__main__':
 import json
 from pathlib import Path
 p=Path(__file__).resolve().parent
 r=mechanism182(legal_fusion,attend,factorial_interaction)
 (p/'evidence/l182/mechanism.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
