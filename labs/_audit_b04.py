"""Independent scalar scoring and identity audit; does not import learner code."""
import hashlib,json,math
from pathlib import Path

def audit_records(raw,protocol,records):
    expected=protocol['configs'];assert len(records)==len(expected)==6
    assert {r['config']['name'] for r in records}=={c['name'] for c in expected}
    by_name={r['config']['name']:r for r in records};rows=[]
    query=raw['query_ids'];support=raw['support_order'];assert len(query)==64 and len(set(query))==64
    assert len(set(support))==len(support) and not set(support)&set(query)
    X,y,mask=raw['X'],raw['y'],raw['missing_mask'];F=len(X[0]);assert len(X)==len(y)==len(mask)==569 and F==30
    for c in expected:
        r=by_name[c['name']];assert r['config']==c
        assert r['inputs_sha256']==protocol['inputs_sha256'] and r['checkpoint_sha256']==protocol['checkpoint']['sha256']
        assert r['query_ids']==query and r['support_ids']==support[:c['support_n']]
        assert r['classes']==[0,1] and len(r['probabilities'])==len(query)
        assert r['transformed_features']==F*(2 if c['indicators'] else 1)
        assert len(r['means'])==F
        for j in range(F):
            values=[X[i][j] for i in r['support_ids'] if not (c['missing'] and mask[i][j])]
            mu=math.fsum(values)/len(values) if values else 0.
            assert math.isclose(mu,r['means'][j],rel_tol=1e-12,abs_tol=1e-12)
        correct=0;loss=0.;brier=0.
        for i,p in zip(query,r['probabilities']):
            assert len(p)==2 and all(math.isfinite(v) and 0<=v<=1 for v in p)
            assert abs(sum(p)-1)<2e-6
            pred=max(range(2),key=lambda k:p[k]);correct+=int(pred==y[i])
            loss-=math.log(max(1e-15,min(1.,p[y[i]])));brier+=(p[1]-y[i])**2
        for field in ['fit_seconds','predict_seconds','peak_process_rss_mib']:assert math.isfinite(r[field]) and r[field]>0
        rows.append(dict(name=c['name'],support_n=c['support_n'],features=r['transformed_features'],correct=correct,total=len(query),accuracy=correct/len(query),log_loss=loss/len(query),brier=brier/len(query),fit_seconds=r['fit_seconds'],predict_seconds=r['predict_seconds'],peak_process_rss_mib=r['peak_process_rss_mib']))
    delta=max(abs(a-b) for p,q in zip(by_name['clean-128']['probabilities'],by_name['batch-128']['probabilities']) for a,b in zip(p,q))
    return dict(status='COMPLETE_COURSE_DIAGNOSTIC',rows=rows,prediction_rows=sum(x['total'] for x in rows),predict_calls=9,batch_max_abs_delta=delta,batch_atol=protocol['query_batch_atol'],batch_within_tolerance=delta<=protocol['query_batch_atol'],paper_parity='NOT_ESTABLISHED',long_context_scaling='NOT_TESTED_BY_THIS_SMALL_SWEEP')
