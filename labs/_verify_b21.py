"""Independent Cartesian enumeration and NumPy message-passing audit."""
import copy,itertools,json
from pathlib import Path
import numpy as np


def numpy_forward(data,weights,owners):
    p=np.array(data['parent_x'])@np.array(weights['embed_parent'])
    c=np.array(data['child_x'])@np.array(weights['embed_child'])
    for layer in range(2):
        pp=[];cc=[]
        for j in range(len(p)):
            neighbors=[c[i] for i,owner in enumerate(owners) if owner==j]
            mean=np.mean(neighbors,axis=0) if neighbors else np.zeros(c.shape[1])
            pp.append(np.maximum(0,p[j]@weights[f'{layer}_parent_own']+mean@weights[f'{layer}_parent_neighbor']+weights[f'{layer}_parent_bias']))
        for i,owner in enumerate(owners):
            cc.append(np.maximum(0,c[i]@weights[f'{layer}_child_own']+p[owner]@weights[f'{layer}_child_neighbor']+weights[f'{layer}_child_bias']))
        p,c=np.array(pp),np.array(cc)
    return (p@np.array(weights['head'])+weights['head_bias'])[data['query_ids']]


def audit(report):
    d=report['data'];original=d['original']
    assert d==dict(parent_time=[0,1,2,12],child_time=[3,4,5,6,7,8],session=[0,0,1,2,3,4],original=[0,0,1,2,1,2],cutoff=10,parent_x=[[1.,0.],[0.,1.],[1.,1.],[9.,9.]],child_x=[[1.,0.],[3.,1.],[2.,0.],[5.,1.],[4.,0.],[6.,1.]],target=[2.,4.,6.],query_ids=[0,1,2])
    assert report['seeds']==[0,1,2] and report['budgets']==[0,1,2]
    assert report['methods']==['random','gradient','exhaustive']
    universe=[]
    # Independent enumeration of six FK cells, not five group choices.
    for a in itertools.product(range(4),repeat=6):
        if a[0]!=a[1]:continue
        if any(d['parent_time'][v]>d['child_time'][i] for i,v in enumerate(a)):continue
        if sum(v!=o for v,o in zip(a,original))<=2:universe.append(a)
    assert len(universe)==35
    assert len(report['states'])==105 and len(report['conditions'])==27
    errors=[];scores=[]
    for seed in range(3):
        rows=[x for x in report['states'] if x['seed']==seed]
        assert len(rows)==35 and set(tuple(x['assignment']) for x in rows)==set(universe)
        w={k:np.asarray(v) for k,v in report['weights'][str(seed)].items()}
        clean=numpy_forward(d,w,original);clean_loss=float(np.mean((clean-d['target'])**2))
        gf=np.array(report['gradients'][str(seed)]['forward']);gr=np.array(report['gradients'][str(seed)]['reverse'])
        for x in rows:
            a=x['assignment'];prediction=numpy_forward(d,w,a)
            np.testing.assert_allclose(prediction,x['prediction'],rtol=0,atol=1e-12)
            loss=float(np.mean((prediction-d['target'])**2))
            assert abs(loss-x['loss'])<1e-12
            assert x['cost']==sum(u!=v for u,v in zip(a,original))
            gain=sum(gf[i,new]-gf[i,old]+gr[new,i]-gr[old,i] for i,(old,new) in enumerate(zip(original,a)))
            assert abs(gain-x['linear_gain'])<1e-12
            errors.append(float(np.max(np.abs(prediction-x['prediction']))))
        for budget in [0,1,2]:
            feasible=[x for x in rows if x['cost']<=budget];exact=[x for x in feasible if x['cost']==budget]
            optimal=min(feasible,key=lambda x:(-x['loss'],x['cost'],x['assignment']))
            gradient=min(feasible,key=lambda x:(-x['linear_gain'],x['cost'],x['assignment']))
            rng=np.random.default_rng(1000+10*seed+budget);random=exact[int(rng.integers(len(exact)))]
            conditions=[x for x in report['conditions'] if x['seed']==seed and x['budget']==budget]
            assert len(conditions)==3 and {x['method'] for x in conditions}==set(report['methods'])
            for x in conditions:
                chosen={'random':random,'gradient':gradient,'exhaustive':optimal}[x['method']]
                for k in ['assignment','prediction','loss','cost','linear_gain']:assert x[k]==chosen[k],k
                assert abs(x['clean_loss']-clean_loss)<1e-12
                assert abs(x['actual_gain']-(chosen['loss']-clean_loss))<1e-12
                assert abs(x['regret']-(optimal['loss']-chosen['loss']))<1e-12
                assert x['regret']>=-1e-12
                scores.append({k:x[k] for k in ['seed','budget','method','loss','regret','cost']})
    return dict(status='PASS',independent_states=105,prediction_coordinates=315,conditions=27,
                max_prediction_error=max(errors),scores=scores)


if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=json.loads((p/'evidence/b21/diagnostic.json').read_text());result=audit(r)
    corruptions=[lambda x:x['data']['target'].__setitem__(0,99),lambda x:x['states'].pop(),
      lambda x:x['conditions'].pop(),lambda x:x['states'][0]['prediction'].__setitem__(0,999),
      lambda x:x['conditions'][2].__setitem__('regret',1),lambda x:x['conditions'][1].__setitem__('method','random'),
      lambda x:x['states'][1].__setitem__('cost',99),lambda x:x['conditions'][1].__setitem__('clean_loss',0)]
    for corrupt in corruptions:
        bad=copy.deepcopy(r);corrupt(bad)
        try:audit(bad)
        except (AssertionError,KeyError,ValueError):pass
        else:raise AssertionError('Corrupt evidence accepted')
    result['corruptions_rejected']=len(corruptions)
    (p/'_verify_b21_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='scores'})
