"""Rebuild L066 row identities, source-wrapper predictions and metrics."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sources/l066-v2'))
from tabicl import TabICLClassifier

def data(name):
    if name=='wdbc':b=load_breast_cancer();x=b.data.astype('float32');raw=b.target
    else:
        table=pd.read_parquet(ROOT/f'data/cache/{name}.parquet');raw=table.pop('class' if name=='diabetes' else 'Class');x=table.to_numpy(dtype='float32')
    _,y=np.unique(raw,return_inverse=True);return x,y

def check(path,output):
    evidence=json.loads(path.read_text());torch.set_num_threads(1)
    checkpoint=ROOT/'data/cache/foundation/tabicl-v1-0208.ckpt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==evidence['checkpoint_sha256']
    assert evidence['protocol']['context_fractions']==[.125,.375,1.] and evidence['protocol']['query_fraction']==.2
    expected={(d,s,f) for d in evidence['config']['datasets'] for s in evidence['config']['seeds'] for f in [.125,.375,1.]}
    assert {(r['dataset'],r['seed'],r['fraction']) for r in evidence['records']}==expected
    assert len(evidence['records'])==len(expected)
    checks=[];predictions=0
    for record in evidence['records']:
        x,y=data(record['dataset']);seed=record['seed'];fraction=record['fraction']
        assert hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==record['data_sha256']
        pool,query=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        pool=np.random.default_rng(seed+66000).permutation(pool);context=pool[:max(2,int(len(pool)*fraction))]
        assert np.array_equal(query,record['query_ids']) and np.array_equal(context,record['context_ids'])
        assert np.array_equal(y[query],record['targets']) and np.array_equal(y[context],record['context_labels']) and not set(query)&set(context)
        assert record['rows']==len(y) and record['features']==x.shape[1]
        clf=TabICLClassifier(n_estimators=1,norm_methods='none',model_path=checkpoint,checkpoint_version='tabicl-classifier-v1-0208.ckpt',allow_auto_download=False,device='cpu',n_jobs=1,use_amp=False,inference_config={k:dict(offload=False) for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
        clf.fit(x[context],y[context]);original=clf.predict_proba(x[query]);p=np.asarray(record['probabilities'])
        delta=float(abs(original-p).max());assert delta<4e-5,(record['dataset'],seed,fraction,delta)
        classes=clf.classes_;assert np.array_equal(classes,np.arange(p.shape[1]))
        keep=np.flatnonzero(clf.ensemble_generator_.unique_filter_.features_to_keep_);assert np.array_equal(keep,record['kept_columns'])
        from_logits=softmax(np.asarray(record['logits'])/.9,axis=-1);assert abs(p-from_logits).max()<3e-7
        assert abs(p.sum(1)-1).max()<3e-7 and np.all((p>0)&(p<=1))
        accuracy=float((p.argmax(1)==y[query]).mean());nll=float(-np.log((p/p.sum(1,keepdims=True))[np.arange(len(query)),y[query]]).mean())
        assert abs(accuracy-record['accuracy'])<1e-12 and abs(nll-record['log_loss'])<2e-6
        assert all(np.isfinite(record[k]) and record[k]>=0 for k in ['preprocessing_seconds','column_seconds','row_seconds','icl_seconds'])
        checks.append(dict(dataset=record['dataset'],seed=seed,fraction=fraction,context=len(context),queries=len(query),source_probability_max_delta=delta,accuracy=accuracy,log_loss=nll));predictions+=len(query)
        print(record['dataset'],seed,fraction,'PASS',flush=True)
    r=dict(status='PASS',evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=checks,predictions=predictions,scope='Independent raw-cache labels, split membership and label-blind nested support prefixes; every probability regenerated using original 0.1.4 sklearn wrapper with original v1 checkpoint, plus direct logit/metric reconstruction. No teaching model, preprocessing or loader imported. Timings checked for validity, not re-created.')
    output.write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path,nargs='?',default=ROOT/'_verify_l066_v2_results.json');p.add_argument('--report',type=Path,default=ROOT.parent/'reviews/lesson-quality-audit-047-070/066-evidence.json');a=p.parse_args();r=check(a.evidence,a.report);print(json.dumps(dict(status=r['status'],records=len(r['records']),predictions=r['predictions'])))
