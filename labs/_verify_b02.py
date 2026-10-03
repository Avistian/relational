"""Independent replay plus meaningful corruption and wrong-learner tests."""
import hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
from _audit_b02 import audit
from _test_b02 import check_functions,check_budget
from relkit.embeddings_b02 import piecewise_linear,member_predictions,greedy_validation
P=Path(__file__).resolve().parent;E=P/'evidence/b02'
def verify():
    result=audit(E/'compact');check_functions(piecewise_linear,member_predictions,greedy_validation);check_budget()
    wrong=[lambda x,e:np.clip((np.asarray(x)[...,None]-e[:-1])/np.diff(e),0,1),lambda x,w,r,s,b:np.broadcast_to(x@w,(len(r),len(x),w.shape[1])),lambda p,y,max_size=32:[int(np.argmin(np.mean((p-y)**2,axis=1)))]]
    rejected=0
    for i,fn in enumerate(wrong):
        functions=[piecewise_linear,member_predictions,greedy_validation];functions[i]=fn
        try:check_functions(*functions)
        except (AssertionError,ValueError):rejected+=1
        else:raise AssertionError('Wrong learner accepted')
    mutations=['bytes','row_overlap','seed_missing','selected_config','member_identity','nan_prediction','test_prediction']
    for case in mutations:
        with tempfile.TemporaryDirectory(prefix='b02-corrupt-') as tmp:
            d=Path(tmp)/'compact';shutil.copytree(E/'compact',d)
            if case=='bytes':
                p=d/'data/y.npy';p.write_bytes(p.read_bytes()+b'corrupt')
            elif case=='row_overlap':
                p=d/'data/test.npy';a=np.load(p);a[0]=np.load(d/'data/train.npy')[0];np.save(p,a)
            elif case=='seed_missing':
                p=d/'runs/evaluation/report.json';a=json.loads(p.read_text());a['experiments'].pop();p.write_text(json.dumps(a))
            elif case=='selected_config':
                p=d/'runs/evaluation/config.json';a=json.loads(p.read_text());a['base_config']['configs'][0]['model']['dropout']=.999;p.write_text(json.dumps(a))
            else:
                p=d/'runs/0/observed_ensemble.npz'
                with np.load(p) as z:a={k:z[k] for k in z.files}
                if case=='member_identity':a['ids'][0]=999
                elif case=='nan_prediction':a['test'][0,0]=np.nan
                else:a['test']+=1
                np.savez(p,**a)
            # Structural tests rehash the altered file: rejection must go beyond integrity.
            if case!='bytes':
                lock=json.loads((d/'compact-lock.json').read_text());lock[str(p.relative_to(d))]=hashlib.sha256(p.read_bytes()).hexdigest();(d/'compact-lock.json').write_text(json.dumps(lock))
            try:audit(d)
            except (AssertionError,ValueError):pass
            else:raise AssertionError('Accepted corruption '+case)
    summary=dict(status='PASS',prediction_audit=result['status'],mutations_rejected=mutations,wrong_learner_functions_rejected=rejected)
    (P/'_verify_b02_results.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':verify()
