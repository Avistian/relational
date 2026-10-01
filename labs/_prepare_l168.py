"""Prepare the complete released trial task, with source/key/label/time audits."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,shutil,signal,sys,urllib.request,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
signal.alarm(600)
P=Path(__file__).resolve().parent;E=P/'evidence/l168';S=P/'sources/l168';D=Path('/tmp/l168-data');OUT=Path('/tmp/l168-input')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((P/'sources/l166/source-ledger.json').read_text())
source_files={}
for p in sorted((P/'sources/l166/upstream/model_pretrain').rglob('*')):
 if p.is_file() and '__pycache__' not in str(p):
  key=str(p.relative_to(P/'sources/l166'));assert sha(p)==original['files'][key];source_files[str(p.relative_to(P))]=sha(p)
meta=yaml.safe_load((D/'metadata.yaml').read_text());tm=next(t for t in meta['tasks'] if t['name']=='study-outcome')
# Use the original loader, with the selected metadata task; do not load unrelated tasks.
sys.path.insert(0,str(P/'sources/l166/upstream/model_pretrain'))
from src.eval_utils import load_task_split,downsample_split,_stable_random_state
columns=[SimpleNamespace(**x) for x in tm['columns']]
task=SimpleNamespace(metadata=SimpleNamespace(name=tm['name'],target_column=tm['target_column'],columns=columns))
splits={s:dict(np.load(D/'study-outcome'/(s+'.npz'),allow_pickle=False)) for s in ['train','validation','test']}
for s,a in splits.items():setattr(task,s+'_set',a)
features=sorted(c.name for c in columns if c.dtype in ['float','category'] and c.name!='outcome')
assert len(features)==176 and not set(features)&{'nct_id','timestamp','outcome'}
X,y=load_task_split(task,'train');Q,yt=load_task_split(task,'test')
assert X.shape==(11994,176) and Q.shape==(825,176)
assert set(y)==set(yt)=={0,1} and not np.isinf(X).any() and not np.isinf(Q).any()
keys={};audits=[];reference=S/'study-outcome.zip'
assert sha(reference)=='20eb922c1a8f894563f4b4c900c912e396688d2bd71eeb6b13f19429aa74a649'
with zipfile.ZipFile(reference) as z:
 for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
 z.extractall('/tmp/l168-task')
for s,a in splits.items():
 k=np.column_stack([a['nct_id'],a['timestamp']]);assert len(set(map(tuple,k)))==len(k);keys[s]=k
 f='val' if s=='validation' else s
 ref=pd.read_parquet(Path('/tmp/l168-task/study-outcome')/(f+'.parquet'))
 expected={(int(r.nct_id),int(r.timestamp.value)):int(r.outcome) for r in ref.itertuples()}
 assert set(map(tuple,k))==set(expected),'Query population differs from raw task archive'
 opposite=sum(int(v)==1-expected[tuple(key)] for key,v in zip(k,a['outcome']));assert opposite==len(k)
 timestamps={}
 for c in features:
  if 'MAX(' in c and 'TIMESTAMP' in c:
   v=a[c];finite=np.isfinite(v);future=int(np.sum(v[finite].astype('float64')>=k[finite,1]));assert future==0
   timestamps[c]=dict(checked=int(finite.sum()),at_or_after_cutoff=future)
 audits.append(dict(split=s,rows=len(k),positive_count=int(a['outcome'].sum()),labels_complement_raw_task=opposite,max_timestamp_audit=timestamps))
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert set(map(tuple,keys[a])).isdisjoint(map(tuple,keys[b]))
support=[]
for seed in range(10):
 token=f'{meta["dataset_name"]}:study-outcome:{seed}';integer=_stable_random_state(token)
 idx=np.random.default_rng(integer).choice(len(y),512,replace=False)
 actual,labels=downsample_split(X,y,512,integer);np.testing.assert_array_equal(actual,X[idx]);np.testing.assert_array_equal(labels,y[idx]);support.append(idx)
# The 365-day target horizon ends before the earliest test query for ALL train candidates.
horizon=365*86400*10**9
assert (keys['train'][:,1]+horizon<keys['test'][:,1].min()).all()
np.savez_compressed(OUT/'prepared.npz',X_train=X,y_train=y,X_test=Q,y_test=yt,support=np.array(support),train_keys=keys['train'],test_keys=keys['test'])
shutil.copyfile(OUT/'prepared.npz',E/'prepared.npz');shutil.copyfile(D/'metadata.yaml',S/'metadata.yaml')
old=json.loads((P/'evidence/l166/input-manifest.json').read_text())
for name in ['RDBPFN.pt','RDBPFN_single.pt','tabicl-classifier-v1.1-0506.ckpt']:
 origin=Path('/tmp/l166-input')/name;assert sha(origin)==old['files'][name]['sha256'];shutil.copyfile(origin,OUT/name)
manifest=dict(code_revision=old['code_revision'],data_revision=old['data_revision'],tabicl_revision=old['tabicl_revision'],dataset='rel-trial-dfs-2',task='study-outcome',
 features=features,seed_key='rel-trial-dfs-2:study-outcome:{seed}',support=512,seeds=list(range(10)),test_rows=len(Q),
 files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.name!='input-manifest.json'},
 split_audit=audits,label_archive=dict(path='sources/l168/study-outcome.zip',sha256=sha(reference)),
 label_orientation='ONE_MINUS_RAW_RELBench_PRIMARY_OUTCOME',support_label_horizon='ALL_TRAIN_HORIZONS_BEFORE_ALL_TEST_QUERIES',
 original_dfs_regeneration='NOT_RUN',historical_availability='NOT_ESTABLISHED',historical_identity='NOT_ESTABLISHED',
 pretraining=dict(reported_predictor_data='SYNTHETIC',real_schema_generator_training='SPIDER_AND_BIRD_REPORTED',independent_checkpoint_training_lineage='NOT_ESTABLISHED',exact_target_schema_exclusion='NOT_ESTABLISHED'),
 source_files=source_files,source_manifest_sha256=sha(P/'sources/l166/source-ledger.json'),
 paper_targets={'RDBPFN':.5986,'RDBPFN_single':.5961,'TabICLv1.1':.5926},descriptive_tolerance=.02)
for p in [OUT/'input-manifest.json',E/'input-manifest.json']:p.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['features','files','source_files','split_audit']},indent=2))
