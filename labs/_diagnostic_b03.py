"""Separate course diagnostic: all six label permutations, pinned v2, Iris.
Run with PYTHONPATH=labs/data/cache/l064-source/official. Not the blood benchmark.
"""
import hashlib,itertools,json,platform,time
from pathlib import Path
import numpy as np
import sklearn,torch,tabpfn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
P=Path(__file__).resolve().parent
if __name__=='__main__':
    out=P/'evidence/b03/permutation-diagnostic.json'
    if out.exists():raise SystemExit('Refusing to overwrite immutable diagnostic')
    assert tabpfn.__version__=='2.0.9' and sklearn.__version__=='1.6.1'
    torch.set_num_threads(1)
    checkpoint=P/'data/cache/foundation/tabpfn-v2.ckpt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest=='f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'
    x,y=load_iris(return_X_y=True);train,test=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=0)
    records=[];start=time.monotonic()
    for permutation in itertools.permutations(range(3)):
        pi=np.array(permutation)
        model=TabPFNClassifier(n_estimators=4,device='cpu',model_path=checkpoint,inference_precision=torch.float32,random_state=0,n_jobs=1,fit_mode='fit_preprocessors').fit(x[train],pi[y[train]])
        prob=model.predict_proba(x[test]);assert list(model.classes_)==[0,1,2]
        record=dict(permutation=list(permutation),probabilities=prob.tolist())
        if permutation==(0,1,2):
            # Scoring labels live outside the predictor. Changing them is a scoring intervention.
            altered_targets=np.roll(y[test],1)
            assert not np.array_equal(altered_targets,y[test])
            again=model.predict_proba(x[test])
            record['repeat_probabilities']=again.tolist()
        records.append(record)
    result=dict(status='COMPLETE_COURSE_DIAGNOSTIC',scope='Not blood-transfusion or any paper benchmark; not EquiTabPFN reproduction',dataset='sklearn Iris',data=x.tolist(),labels=y.tolist(),train_ids=train.tolist(),test_ids=test.tolist(),split_seed=0,model_seed=0,n_estimators=4,precision='CPU float32',tolerance=dict(atol=1e-6,rtol=0),checkpoint_sha256=digest,records=records,seconds=time.monotonic()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__,tabpfn=tabpfn.__version__),query_label_check='Repeated prediction receives X_query only; scoring targets are not arguments. This does not prove query-feature independence.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['status'],result['seconds'])
