"""Complete36-condition fixed-kernel diagnostic with immutable prediction output."""
import hashlib,json,itertools
from pathlib import Path
import numpy as np
from relkit.adaptation_b12 import make_fixture,predict
P=Path(__file__).resolve().parent

def run(output=None):
    rows=[];conditions=[]
    for seed in range(3):
        fixture=make_fixture(seed);before=hashlib.sha256(fixture['weights'].tobytes()).hexdigest()
        for reach,mode,channel in itertools.product(['high','low'],['intact','shuffled','hidden'],['relational','dual']):
            values=predict(fixture,reach,mode,channel);truth=fixture['labels'][8:]
            for key,y,p in zip(fixture['keys'],truth,values):rows.append(dict(seed=seed,reachability=reach,mode=mode,channel=channel,entity=key[0],cutoff=key[1],label=float(y),prediction=float(p)))
            clean=predict(fixture,reach,'intact',channel)
            conditions.append(dict(seed=seed,reachability=reach,mode=mode,channel=channel,brier=float(np.mean((values-truth)**2)),mean_abs_change=float(np.mean(abs(values-clean))),reachable_labels=4 if reach=='high' else 0,eligible_support_budget=8,used_support_labels=0 if mode=='hidden' else (8 if channel=='dual' else (4 if reach=='high' else 0))))
        assert hashlib.sha256(fixture['weights'].tobytes()).hexdigest()==before
    report=dict(status='COMPLETE_COURSE_DIAGNOSTIC',conditions=conditions,predictions=rows,condition_count=36,prediction_count=432,fitted_models=0,weights_unchanged=True,whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    if output:
        p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(report,indent=2)+'\n'
        if p.exists() and p.read_text()!=payload:raise RuntimeError('Refusing to replace different predictions')
        p.write_text(payload)
    return report
if __name__=='__main__':
    r=run(P/'evidence/b12/diagnostic.json');print(r['status'],r['condition_count'],r['prediction_count'])
