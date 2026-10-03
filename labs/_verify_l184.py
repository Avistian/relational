"""Independent scalar mechanism checks and adversarial packet checks."""
import json,math,random
from pathlib import Path
import numpy as np
from _audit_l184 import audit184
from _check_l184 import checks
from relkit.gelgt_l184 import temporal_bfs,gaussian_features,keyed_mae,semantic_refine,attention_trace

def independent184(sample,gaussian,score):
    checks(sample,gaussian,score)
    rng=random.Random(184)
    for i in range(100):
        mu=rng.uniform(-5,5);width=rng.uniform(.1,10);lags=[rng.uniform(0,10) for _ in range(5)]
        assert np.allclose(gaussian(lags,mu,width),[math.exp(-(v-mu)**2/width**2) for v in lags],atol=1e-14)
        truth=[(k%7,k,float(rng.random())) for k in range(25)]
        pred=[(e,t,y+.25) for e,t,y in truth];rng.shuffle(pred)
        assert abs(score(truth,pred)-.25)<1e-12
    # Three distinct incorrect learner implementations must be rejected.
    wrong=[(lambda *a:[0,1,2,3],gaussian,score),(sample,lambda x,m,w:np.ones_like(x),score),(sample,gaussian,lambda y,p:0.)]
    for trio in wrong:
        try:checks(*trio)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Wrong learner implementation accepted')
    return {'status':'PASS','random_scalar_cases':100,'random_keyed_cases':100,'wrong_learner_functions_rejected':3}

if __name__=='__main__':
    p=Path(__file__).resolve().parent;e=p/'evidence/l184';m=json.loads((e/'input-manifest.json').read_text())
    r=audit184(e/'packet',m,gaussian_features)
    assert r==json.loads((e/'report.json').read_text())
    v=independent184(temporal_bfs,gaussian_features,keyed_mae)
    assert semantic_refine([[1,0],[0,1],[.1,0],[2,0]],[0,1,2,2],3)==[0,1,3]
    try:semantic_refine([[1],[2],[3]],[0,1,1],2)
    except ValueError:pass
    else:raise AssertionError('impossible budget accepted')
    for name in ['upstream/utils.py','task/train.parquet','db/results.parquet']:
        bad=json.loads(json.dumps(m));bad['files'][name]='0'*64
        try:audit184(e/'packet',bad,gaussian_features)
        except ValueError:pass
        else:raise AssertionError('changed packet accepted')
    v.update(labels=r['labels'],corrupt_inputs_rejected=3,source_report_parity='EXACT',training='NOT_RUN')
    (p/'_verify_l184_results.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
