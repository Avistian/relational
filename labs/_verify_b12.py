"""Scalar math oracle, complete keys, forbidden interventions and mutation tests."""
import copy,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
from relkit.adaptation_b12 import make_fixture,predict,label_attention
from _test_b12 import check_eligibility,check_attention,check_intervention
P=Path(__file__).resolve().parent

def oracle(f,reach,mode,channel):
    # Deliberately scalar: no implementation attention, eligibility or intervention calls.
    z=[[sum((f['x'][i,k]+.25*f['parent'][i,k])*f['weights'][k,j] for k in range(4)) for j in range(4)] for i in range(20)]
    labels=list(f['labels'][:8])
    if mode=='shuffled':labels=[labels[i] for i in f['permutation']]
    def read(q,batch):
        ids=[s for s in range(8) if mode!='hidden' and (batch or f[reach][q-8,s]) and math.isfinite(f['event'][s]) and math.isfinite(f['available'][s]) and math.isfinite(f['horizon'][s]) and f['event'][s]<=10 and f['available'][s]<=10 and f['event'][s]+f['horizon'][s]<=10]
        if not ids:return .5
        scores=[sum(a*b for a,b in zip(z[q],z[s]))/2 for s in ids];mx=max(scores);ws=[math.exp(s-mx) for s in scores]
        return sum(w*labels[s] for w,s in zip(ws,ids))/sum(ws)
    return np.array([read(q,False) if channel=='relational' else .5*(read(q,False)+read(q,True)) for q in range(8,20)])

def verify(report):
    expected={(seed,r,m,c,entity,10) for seed,r,m,c,entity in itertools.product(range(3),['high','low'],['intact','shuffled','hidden'],['relational','dual'],range(8,20))}
    keys=[(r['seed'],r['reachability'],r['mode'],r['channel'],r['entity'],r['cutoff']) for r in report['predictions']]
    assert len(keys)==len(set(keys))==432 and set(keys)==expected
    assert len(report['conditions'])==36
    errors=[];interventions=0
    for seed in range(3):
        f=make_fixture(seed)
        for reach,mode,channel in itertools.product(['high','low'],['intact','shuffled','hidden'],['relational','dual']):
            rows=sorted([r for r in report['predictions'] if (r['seed'],r['reachability'],r['mode'],r['channel'])==(seed,reach,mode,channel)],key=lambda r:r['entity'])
            actual=np.array([r['prediction'] for r in rows]);target=oracle(f,reach,mode,channel);errors.append(float(max(abs(actual-target))))
            assert all(r['label']==f['labels'][r['entity']] for r in rows)
            condition=next(r for r in report['conditions'] if (r['seed'],r['reachability'],r['mode'],r['channel'])==(seed,reach,mode,channel))
            assert abs(condition['brier']-sum((r['prediction']-r['label'])**2 for r in rows)/12)<1e-12
            clean=oracle(f,reach,'intact',channel)
            assert abs(condition['mean_abs_change']-float(np.mean(abs(target-clean))))<1e-12
            changed=copy.deepcopy(f);changed['labels'][8:]=1-changed['labels'][8:]
            np.testing.assert_allclose(predict(changed,reach,mode,channel),actual,atol=1e-12);interventions+=1
        # Future / late / incomplete-window supports never affect either channel.
        for field,value in [('event',11.),('available',11.),('horizon',20.)]:
            a=copy.deepcopy(f);a[field][0]=value;b=copy.deepcopy(a);b['labels'][0]=1-b['labels'][0]
            for channel in ['relational','dual']:
                np.testing.assert_allclose(predict(a,'high','intact',channel),predict(b,'high','intact',channel),atol=1e-12);interventions+=1
        np.testing.assert_array_equal(predict(f,'low','intact','relational'),np.full(12,.5))
        np.testing.assert_array_equal(predict(f,'low','shuffled','relational'),np.full(12,.5))
        # Support permutation and independent query batches preserve attention.
        z=(f['x']+.25*f['parent'])@f['weights'];mask=np.ones((12,8),bool);perm=f['permutation']
        full=label_attention(z[8:],z[:8],f['labels'][:8],mask)
        np.testing.assert_allclose(full,label_attention(z[8:],z[:8][perm],f['labels'][:8][perm],mask[:,perm]),atol=1e-12)
        np.testing.assert_allclose(full,np.concatenate([label_attention(z[8+i:9+i],z[:8],f['labels'][:8],mask[i:i+1]) for i in range(12)]),atol=1e-12)
    assert max(errors)<1e-12
    rejected=0
    bad=[(check_eligibility,lambda e,a,h,t:np.asarray(e)<=t),(check_attention,lambda q,k,v,m:np.full(len(q),.5)),(check_intervention,lambda y,m,p:y.copy())]
    for check,fn in bad:
        try:check(fn)
        except AssertionError:rejected+=1
    assert rejected==3
    return dict(status='PASS',keyed_predictions=432,conditions=36,max_scalar_oracle_error=max(errors),forbidden_interventions=interventions,wrong_functions_rejected=rejected,support_permutation=True,query_batch_invariance=True,shuffled_label_changes=[int(np.sum(make_fixture(s)['labels'][:8]!=make_fixture(s)['labels'][:8][make_fixture(s)['permutation']])) for s in range(3)])
if __name__=='__main__':
    r=verify(json.loads((P/'evidence/b12/diagnostic.json').read_text()));(P/'_verify_b12_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
