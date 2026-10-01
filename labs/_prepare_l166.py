"""Fetch immutable selected inputs, preserve full keys, and audit released features."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
P=Path(__file__).resolve().parent;E=P/'evidence/l166';S=P/'sources/l166';OUT=Path('/tmp/l166-input');OUT.mkdir(exist_ok=True)
UP=Path('/tmp/l166-upstream');CODE='a95378225478daa262b85f180d482da7516b0af6'
DATA='d6a88c0a8cce79607cfc0fca0dcba78ba262ffad'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(S/'upstream/model_pretrain'))
from src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
from src.eval_utils import load_task_split,downsample_split,_stable_random_state,fill_nans
db=DBBRDBDataset('/tmp/l166-data');task=next(t for t in db.tasks if t.metadata.name=='driver-dnf')
features=sorted(c.name for c in task.metadata.columns if c.dtype in ['float','category'] and c.name!='did_not_finish')
train,y=load_task_split(task,'train');test,yt=load_task_split(task,'test')
assert len(train)==11411 and len(test)==702
assert train.shape[1]==len(features)
assert np.isfinite(train[~np.isnan(train)]).all() and np.isfinite(test[~np.isnan(test)]).all()
support=[]
for seed in range(10):
    key=f'{db.dataset_name}:{task.metadata.name}:{seed}'
    idx=np.random.default_rng(_stable_random_state(key)).choice(len(y),size=512,replace=False)
    a,b=downsample_split(train,y,512,_stable_random_state(key))
    np.testing.assert_array_equal(a,train[idx]);np.testing.assert_array_equal(b,y[idx]);support.append(idx)
keys={};split_report=[]
raw=pd.read_parquet(Path.home()/'.cache/relbench/rel-f1/db/results.parquet')
raw['ns']=raw.date.astype('datetime64[ns]').astype('int64')
group={int(k):v for k,v in raw.groupby('driverId')};horizon=30*86400*10**9
for split,source in [('train',task.train_set),('validation',task.validation_set),('test',task.test_set)]:
    k=np.column_stack([source['driverId'],source['date']]);assert len(set(map(tuple,k)))==len(k);keys[split]=k
    # Independently enumerate the actual 30-day outcome window; no labels enter feature checks.
    observed=[];recomputed=[];counts=[]
    for (driver,cutoff),label in zip(k,source['did_not_finish']):
        rows=group[int(driver)];future=rows[(rows.ns>cutoff)&(rows.ns<=cutoff+horizon)]
        assert len(future)>0
        recomputed.append(int((future.statusId!=1).any()));observed.append(int(label));counts.append(len(future))
    label_match=int(np.sum(np.array(recomputed)==observed))
    # A directly represented MAX timestamp must precede each owner cutoff.
    maxcols=[f for f in features if '.MAX(' in f and 'TIMESTAMP_date' in f]
    comparisons={}
    for c in maxcols:
        value=np.asarray(source[c]);valid=np.isfinite(value)
        # This release stores BOTH in nanoseconds (feature values rounded to float32).
        comparisons[c]=dict(checked=int(valid.sum()),future=int(np.sum(value[valid].astype('float64')>=k[valid,1])))
    split_report.append(dict(split=split,rows=len(k),positive_count=int(sum(observed)),raw_sql_labels_equal=label_match,
                             raw_sql_labels_opposite=len(k)-label_match,max_timestamp_audit=comparisons))
assert set(map(tuple,keys['train'])).isdisjoint(set(map(tuple,keys['test'])))
np.savez_compressed(OUT/'prepared.npz',X_train=train,y_train=y,X_test=test,y_test=yt,support=np.array(support),train_keys=keys['train'],test_keys=keys['test'])
shutil.copyfile(OUT/'prepared.npz',E/'prepared.npz')
for arm,file in [('RDBPFN','model_eval00528.pt'),('RDBPFN_single','model_eval00360.pt')]:
    shutil.copyfile(UP/'model_pretrain/checkpoints'/arm/file,OUT/(arm+'.pt'))
url='https://huggingface.co/api/models/jingang/TabICL-clf';info=json.load(urllib.request.urlopen(url));rev=info['sha']
name='tabicl-classifier-v1.1-0506.ckpt';dest=OUT/name
if not dest.exists():
    with urllib.request.urlopen('https://huggingface.co/jingang/TabICL-clf/resolve/'+rev+'/'+name) as response,dest.open('wb') as f:shutil.copyfileobj(response,f)
(S/'tabicl-model-api.json').write_text(json.dumps(dict(revision=rev,filename=name,sha256=digest(dest)),indent=2)+'\n')
manifest=dict(code_revision=CODE,data_revision=DATA,tabicl_revision=rev,features=features,seed_key='rel-f1-dfs-2:driver-dnf:{seed}',support=512,seeds=list(range(10)),test_rows=len(test),
              files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name!='input-manifest.json'},split_audit=split_report,
              raw_label_source=dict(path=str(Path.home()/'.cache/relbench/rel-f1/db/results.parquet'),sha256=digest(Path.home()/'.cache/relbench/rel-f1/db/results.parquet')),
              original_dfs_regeneration='NOT_RUN',historical_availability='NOT_ESTABLISHED',historical_identity='NOT_ESTABLISHED')
(OUT/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['features','files']},indent=2))
