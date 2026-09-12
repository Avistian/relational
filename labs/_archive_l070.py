"""Independent raw-data and prediction audit of the immutable L070 archive.

No teaching benchmark, selection, preprocessing or model implementation is used.
The original package check separately reproduces the declared one-view wrappers.
"""
import argparse, hashlib, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
TARGETS={'diabetes':'class','blood_transfusion':'Class','kc1':'defects','phoneme':'Class'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def task(name):
    if name=='breast_cancer':
        x,y=load_breast_cancer(return_X_y=True,as_frame=True)
        sha=hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()+y.to_numpy().tobytes()).hexdigest()
        labels=['malignant','benign']
    else:
        path=ROOT/'data/cache'/f'{name}.parquet';sha=digest(path);df=pd.read_parquet(path)
        target=TARGETS[name];x=df.drop(columns=target);raw=df[target]
        if raw.dtype==object or str(raw.dtype)=='category':
            labels=sorted(raw.astype(str).unique());y=(raw.astype(str)==labels[-1]).astype(int)
        else:labels=['0','1'];y=raw.astype(int)
    y=np.asarray(y,dtype=int);ids=np.arange(len(y))
    if len(ids)>600:ids=np.sort(np.random.default_rng(70).choice(ids,600,replace=False))
    tr,te=train_test_split(ids,test_size=.2,random_state=70,stratify=y[ids])
    tr,va=train_test_split(tr,test_size=.25,random_state=71,stratify=y[tr])
    assert all(np.issubdtype(dt,np.number) for dt in x.dtypes),'Declared panel is numeric'
    matrix=x.to_numpy(dtype=float);median=np.nanmedian(matrix[tr],axis=0);median=np.nan_to_num(median)
    matrix=np.where(np.isnan(matrix),median,matrix).astype('float32')
    return matrix,y,dict(train=tr,val=va,test=te),sha,labels

def loss(y,p):
    p=np.clip(np.asarray(p,dtype=float),np.finfo(float).eps,1-np.finfo(float).eps)
    return float(np.mean(-np.asarray(y)*np.log(p)-(1-np.asarray(y))*np.log1p(-p)))

def raw_check(output):
    path=ROOT/'_verify_l070_results.json';r=json.loads(path.read_text());checks=[]
    expected={(d+'/random',s,a) for d in [*TARGETS,'breast_cancer'] for s in [0,1,2] for a in ['XGBoost','TabM-mini','TabPFN-v2','TabICL-v1.1','TabPFN-2.5-synthetic','TabPFN-3','TabICLv2']}
    actual=[(v['dataset'],v['seed'],v['arm']) for v in r['records']];assert len(actual)==len(set(actual)) and set(actual)==expected
    for name in [*TARGETS,'breast_cancer']:
        x,y,parts,sha,labels=task(name);meta=r['datasets'][name+'/random'];assert meta['data_sha256']==sha
        for key,ids in parts.items():assert ids.tolist()==meta['ids'][key]
        for row in [v for v in r['records'] if v['dataset']==name+'/random']:
            assert y[parts['test']].tolist()==row['targets'];p=np.asarray(row['predictions']);assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
            delta=abs(loss(row['targets'],p)-row['error']);assert delta<1e-7
            assert int(np.argmin(row['validation_errors']))==row['selected']
            checks.append(dict(dataset=row['dataset'],seed=row['seed'],arm=row['arm'],rows=len(p),loss_delta=delta,positive_label=labels[-1]))
    report=dict(status='PASS',archive_sha256=digest(path),checker_sha256=digest(__file__),records=len(checks),predictions=sum(v['rows'] for v in checks),scope='Independent original parquet/sklearn labels, sampling, three disjoint partitions, semantic positive class, all 105 binary losses and saved-scalar argmins; archived validation predictions are absent, so argmin replay alone does not establish their losses.',checks=checks)
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}));return report

def pretrained(mode,output):
    import torch
    from threadpoolctl import threadpool_limits
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    expected=('2.0.9','0.1.4') if mode=='historical' else ('8.5.0','2.2.0')
    assert tuple(importlib.metadata.version(k) for k in ['tabpfn','tabicl'])==expected
    torch.set_num_threads(1);start=time.perf_counter();output=Path(output);assert not output.exists();journal=output.with_suffix('.partial.jsonl');assert not journal.exists()
    archive=json.loads((ROOT/'_verify_l070_results.json').read_text());manifest=json.loads((ROOT/'_sources_foundation.json').read_text());records=[]
    roster={'TabPFN-v2':('tabpfn-v2.ckpt',manifest['checkpoints']['v2']['sha256']),'TabICL-v1.1':('tabicl-v1.1.ckpt',manifest['checkpoints']['tabicl']['sha256'])} if mode=='historical' else {arm:(manifest['current_checkpoints'][key]['filename'],manifest['current_checkpoints'][key]['sha256']) for arm,key in [('TabPFN-2.5-synthetic','v25'),('TabPFN-3','v3'),('TabICLv2','tabiclv2')]}
    with threadpool_limits(limits=1):
        for name in [*TARGETS,'breast_cancer']:
            x,y,ids,sha,labels=task(name)
            for seed in [0,1,2]:
                for arm,(filename,checkpoint_sha) in roster.items():
                    path=ROOT/'data/cache/foundation'/filename;assert digest(path)==checkpoint_sha
                    if arm.startswith('TabICL'):
                        m=TabICLClassifier(n_estimators=1,device='cpu',random_state=seed,n_jobs=1,use_amp=False,model_path=str(path),checkpoint_version='tabicl-classifier-v1.1-0506.ckpt' if mode=='historical' else filename,allow_auto_download=False)
                    else:
                        kw=dict(n_jobs=1) if mode=='historical' else dict(n_preprocessing_jobs=1)
                        m=TabPFNClassifier(n_estimators=1,device='cpu',random_state=seed,model_path=path,**kw)
                    m.fit(x[ids['train']],y[ids['train']]);column=list(m.classes_).index(1)
                    vp=m.predict_proba(x[ids['val']])[:,column];p=m.predict_proba(x[ids['test']])[:,column]
                    row=next(v for v in archive['records'] if v['dataset']==name+'/random' and v['seed']==seed and v['arm']==arm)
                    delta=float(np.max(abs(p-np.asarray(row['predictions']))));vd=abs(loss(y[ids['val']],vp)-row['validation_errors'][0]);assert np.allclose(p,row['predictions'],atol=2e-6,rtol=2e-6),(name,seed,arm,delta);assert vd<2e-6,(name,seed,arm,vd)
                    record=dict(dataset=name+'/random',seed=seed,arm=arm,data_sha256=sha,checkpoint_sha256=checkpoint_sha,positive_label=labels[-1],prediction_delta=delta,validation_loss_delta=vd,validation_probabilities=vp.tolist(),test_probabilities=p.tolist(),validation_ids=ids['val'].tolist(),test_ids=ids['test'].tolist())
                    records.append(record)
                    with journal.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                    print('ORIGINAL',name,seed,arm,'delta',delta,'validation',vd,flush=True)
    result=dict(status='PASS',mode=mode,archive_sha256=digest(ROOT/'_verify_l070_results.json'),checker_sha256=digest(__file__),versions={k:importlib.metadata.version(k) for k in ['tabpfn','tabicl','torch','numpy','scikit-learn','pandas']},records=records,seconds=time.perf_counter()-start,scope='Fresh predictions from pinned original packages/checkpoints on independently reconstructed raw data and context/query partitions; every archived test probability and pretrained validation scalar checked. New validation vectors are retained; this does not certify original pretraining or paper benchmark protocols.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('WROTE',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','historical','current']);p.add_argument('output');a=p.parse_args()
    raw_check(a.output) if a.mode=='raw' else pretrained(a.mode,a.output)
