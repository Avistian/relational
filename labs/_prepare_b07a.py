"""Freeze full Tier-A tables and paired80/20 splits before model execution."""
import hashlib,json,requests,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
P=Path(__file__).resolve().parent;E=P/'evidence/b07a';D=P/'data/b07a'
meta={};splits=[]
for name,oid in [('banknote',1462),('phoneme',1489),('diabetes',37)]:
    if name=='banknote':
        url='https://archive.ics.uci.edu/static/public/267/data.csv'
        r=requests.get(url,timeout=30)
        if r.status_code!=200:
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
            r=requests.get(url,timeout=30);r.raise_for_status()
            from io import StringIO
            df=pd.read_csv(StringIO(r.text),header=None)
        else:
            from io import StringIO
            df=pd.read_csv(StringIO(r.text))
        X=df.iloc[:,:-1].to_numpy(dtype=np.float64);y=df.iloc[:,-1].to_numpy(dtype=np.int64)
        origin=url
    else:
        df=pd.read_parquet(P/f'data/cache/{name}.parquet');X=df.iloc[:,:-1].to_numpy(dtype=np.float64)
        raw=df.iloc[:,-1];classes=sorted(raw.astype(str).unique());y=(raw.astype(str)==classes[-1]).to_numpy(dtype=np.int64)
        origin=f'https://www.openml.org/d/{oid}'
    assert X.shape[0]=={'banknote':1372,'phoneme':5404,'diabetes':768}[name]
    assert np.isfinite(X).all() and set(y)=={0,1}
    path=D/(name+'.npz');np.savez_compressed(path,X=X,y=y,row_id=np.arange(len(y)))
    meta[name]=dict(source=origin,openml_id=oid,rows=len(y),features=X.shape[1],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for seed in [0,1,2]:
        tr,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        splits.append(dict(dataset=name,seed=seed,train=tr.tolist(),test=te.tolist()))
protocol=dict(name='B07a-HYPERFAST-AMORTIZATION',datasets=meta,splits=splits,seeds=[0,1,2],arms=['weights_only','retrieval'],n_ensemble=1,optimization=None,batch_size=512,sampling='torch.randperm(train)[:512], repeat_interleave twice to1024 for784 PCA components',preprocessing='StandardScaler fit on full train; complete finite numeric data; no categorical columns',model='Full released dimensions: RF32768, PCA784, hyper hidden1024,3 main layers,46 label slots',timing=dict(query_rows=[1,32,128],repetitions=3,warmups=1,order='off/on alternating by repetition',support_refresh='banknote seed0 only: append first held-out row with its known label; time regeneration only; no refreshed quality claim'),metrics=['balanced_accuracy','log_loss'],versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','scikit-learn','scipy']},source_pin=json.loads((P/'sources/b07a/source-pin.json').read_text()),deviations=['Fresh stratified80/20 splits; original mini-test IDs unverified','512 generation rows, repeated to1024; remaining training rows affect scaler only','Current source release; publication-era identity audited separately','No tuning, fine-tuning, meta-training, comparative ICL/MLP execution or broad speed claim','Banknote from UCI full numeric table; correspondence to original row order not authenticated'])
path=E/'course-protocol.json'
if path.exists():raise SystemExit('Protocol already frozen')
path.write_text(json.dumps(protocol,indent=2)+'\n');print('Frozen',[(k,v['rows']) for k,v in meta.items()])
