"""Reconstruct all scores from immutable raw predictions; no author score helpers."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
P=Path(__file__).resolve().parent;E=P/'evidence/b18a'

def verify():
    parts=[json.loads((E/(phase+'.json')).read_text()) for phase in ['pilot','matrix','updates']]
    assert all(p['status']=='COMPLETE' for p in parts)
    d=parts[0]['data'];assert all(p['data']==d for p in parts)
    x=np.asarray(d['x']);y=np.asarray(d['y']);tr=np.asarray(d['train']);te=np.asarray(d['test']);sel=np.asarray(d['selected'])
    original_x,original_y=load_breast_cancer(return_X_y=True)
    np.testing.assert_array_equal(x,original_x);np.testing.assert_array_equal(y,original_y)
    t,q=train_test_split(np.arange(len(y)),test_size=.5,random_state=42,stratify=y)
    np.testing.assert_array_equal(tr,t);np.testing.assert_array_equal(te,q)
    s,_=train_test_split(t,train_size=.25,random_state=42,stratify=y[t]);np.testing.assert_array_equal(sel,s)
    assert not set(tr)&set(te) and len(sel)==71
    records=parts[0]['records']+parts[1]['records'];assert len(records)==18
    expected={(s,a,m) for s in [0,1,2] for a in ['POT-full','POT-selected','TACO4'] for m in ['fit_preprocessors','fit_with_cache']}
    assert {(r['seed'],r['arm'],r['mode']) for r in records}==expected
    rows=[];paired=[]
    for r in records:
        p=np.asarray(r['predictions']);assert p.shape==(4,285) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        expected_ids=sel if r['arm']=='POT-selected' else tr
        assert r['support_ids']==expected_ids.tolist()
        auc=float(roc_auc_score(y[te],p[0]));loss=float(log_loss(y[te],p[0]))
        assert abs(auc-r['auc'])<1e-12 and abs(loss-r['log_loss'])<1e-12
        times=np.asarray(r['batch_seconds']);assert times.shape==(4,6) and (times>0).all()
        rows.append(dict(seed=r['seed'],arm=r['arm'],mode=r['mode'],auc=auc,log_loss=loss,fit_seconds=r['fit_seconds'],first_pass_seconds=float(times[0].sum()),repeat_pass_seconds=float(times[1:].sum(axis=1).mean()),peak_mib=r['peak_allocated_bytes']/2**20,repeat_max_difference=float(abs(p-p[0]).max()),K_values=r['K_values']))
    for seed in [0,1,2]:
        for arm in ['POT-full','POT-selected','TACO4']:
            pair=[r for r in records if r['seed']==seed and r['arm']==arm]
            delta=float(np.max(np.abs(np.asarray(pair[0]['predictions'])-np.asarray(pair[1]['predictions']))))
            paired.append(dict(seed=seed,arm=arm,max_probability_difference=delta,tolerance=1e-5,status='PASS' if delta<=1e-5 else 'FAIL'))
    updates=parts[2]['updates'];assert len(updates)==12
    update_rows=[]
    for arm in ['POT-full','TACO4']:
        subset=[r for r in updates if r['arm']==arm];base=next(r for r in subset if r['intervention']=='baseline');bp=np.asarray(base['predictions'])
        for r in subset:
            p=np.asarray(r['predictions']);assert p.shape==(285,) and np.isfinite(p).all()
            assert not set(r['support_ids'])&set(te)
            if r['intervention'] in ['baseline','query_missing']:assert r['state_key']==base['state_key']
            else:assert r['state_key']!=base['state_key']
            if r['intervention']=='add':assert set(r['support_ids'])==set(tr)
            if r['intervention']=='delete':assert r['support_ids']==tr[2:].tolist()
            update_rows.append(dict(arm=arm,intervention=r['intervention'],auc=float(roc_auc_score(y[te],p)),max_probability_difference=float(abs(p-bp).max()),fit_seconds=r.get('fit_seconds',0),query_seconds=float(np.asarray(r['batch_seconds']).sum())))
    explanations=[]
    for r in parts[2]['explanations']:
        orig,replaced=r['calls'];qx=x[r['query_ids']];bg=x[r['background_ids']]
        np.testing.assert_array_equal(orig['x'],qx)
        expected=np.repeat(qx,8,axis=0);expected[:,0]=np.tile(bg[:,0],16)
        np.testing.assert_array_equal(replaced['x'],expected)
        effect=np.asarray(orig['p'])-np.asarray(replaced['p']).reshape(16,8).mean(axis=1)
        np.testing.assert_allclose(effect,r['effects'],rtol=0,atol=1e-12)
        explanations.append(dict(arm=r['arm'],background=r['background'],mean_contrast=float(effect.mean()),prediction_rows=144,forward_batches=sum(len(t) for c in r['calls'] for t in c['seconds']),seconds=sum(sum(t) for c in r['calls'] for t in c['seconds'])))
    result=dict(status='COMPLETE_RELEASE_CHECKPOINT_EXPERIMENT',verification='PASS',fit_count=18,records=rows,cache_agreement=paired,updates=update_rows,explanations=explanations,paper='NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE',pretraining='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',environment=parts[0]['environment'])
    (E/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (P/'_verify_b18a_results.json').write_text(json.dumps(dict(status='PASS',fits=18,query_rows=285,probability_values=18*4*285,explanation_rows=576,cache_agreement_failures=sum(c['status']=='FAIL' for c in paired)),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','updates','explanations']},indent=2))
    return result
if __name__=='__main__':verify()
