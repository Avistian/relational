"""Independent numerical, protocol and evidence-corruption checks; no inference."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,io,json,signal,shutil,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from _audit_l168 import audit168
from _check_l168 import check168
from relkit.transfer_l167 import keyed_auc
from relkit.generalization_l168 import transfer_regime,paired_gains,database_macro
signal.alarm(600);P=Path(__file__).resolve().parent;E=P/'evidence/l168'
assert check168(transfer_regime,paired_gains,database_macro)=='PASS'
rng=np.random.default_rng(168);max_error=0
for n in range(2,202):
 y=np.r_[0,1,rng.integers(0,2,n-2)];scores=rng.integers(0,9,n)/8;keys=np.column_stack([np.arange(n)//2,np.arange(n)%2]);idx=rng.permutation(n)
 value=keyed_auc(keys,y,keys[idx],scores[idx]);oracle=roc_auc_score(y,scores)
 pos=scores[y==1,None];neg=scores[None,y==0];pairwise=float(np.mean((pos>neg)+.5*(pos==neg)))
 max_error=max(max_error,abs(value-oracle),abs(value-pairwise));assert max_error<1e-12
 assert abs(keyed_auc(keys,1-y,keys,1-scores)-value)<1e-12
regime_cases=0
for known in [False,True]:
 for source in [[],['a'],['b'],['a','b']]:
  for labels in [0,1,512]:
   for updates in [0,5]:
    if updates and not labels:
     try:transfer_regime(source,'b',known,labels,updates)
     except ValueError:pass
     else:raise AssertionError('Unlabeled supervised adaptation accepted')
    else:
     got=transfer_regime(source,'b',known,labels,updates)
     assert got['database_holdout']==('SEEN' if 'b' in source else ('HELD_OUT' if known else 'NOT_ESTABLISHED'))
    regime_cases+=1
wrong=[(lambda *a:dict(database_holdout='HELD_OUT',adaptation='FEW_SHOT_ICL'),paired_gains,database_macro),
 (transfer_regime,lambda *a:{'small':{'per_seed':[0,0]},'large':{'per_seed':[0,0]}},database_macro),
 (transfer_regime,paired_gains,lambda *a:dict(databases=4,macro_gain=-.2,positive_databases=1))]
for f in wrong:
 try:check168(*f)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('Bad learner function passed')
pins=json.loads((E/'audit-manifest.json').read_text());report=audit168(P,pins,keyed_auc,paired_gains,database_macro)
assert report==json.loads((E/'report.json').read_text())
# Score raw predictions with a separate library; independence from the rank implementation.
raw_count=0
for spec in pins['experiments']:
 for phase in spec['phases']:
  receipt=json.loads((P/spec['folder']/phase/'receipt.json').read_text())
  for row in receipt['records']:
   a=np.load(P/spec['folder']/phase/f"{row['arm']}-{row['seed']}.npz")
   assert abs(roc_auc_score(a['label'],a['probability'])-row['auc'])<1e-12;raw_count+=len(a['label'])
# Match the independently prepared task archive, including train and validation labels.
label_matches=0
with zipfile.ZipFile(P/'sources/l168/study-outcome.zip') as z:
 for split in ['train','validation','test']:
  raw_name='val' if split=='validation' else split
  df=pd.read_parquet(io.BytesIO(z.read('study-outcome/'+raw_name+'.parquet')))
  data=np.load(P/'sources/l168/validation.npz') if split=='validation' else None
  # Training/test identities can always be independently checked using shipped prepared arrays.
  if split!='validation':
   prepared=np.load(E/'prepared.npz');keys=prepared['train_keys' if split=='train' else 'test_keys'];labels=prepared['y_train' if split=='train' else 'y_test']
  elif data is not None:keys=np.column_stack([data['nct_id'],data['timestamp']]);labels=data['outcome']
  else:continue
  lookup={(int(r.nct_id),int(r.timestamp.value)):int(r.outcome) for r in df.itertuples()}
  assert set(map(tuple,keys))==set(lookup)
  assert all(label==1-lookup[tuple(k)] for k,label in zip(keys,labels));label_matches+=len(keys)
# Verify every reused source byte against its ORIGINAL L166 source ledger.
original=json.loads((P/'sources/l166/source-ledger.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text())
for path,digest in manifest['source_files'].items():assert digest==original['files'][path.removeprefix('sources/l166/')]==hashlib.sha256((P/path).read_bytes()).hexdigest()
# Record pairing is independent of serialization order, and macro is independent of row counts.
records=[]
for d,v in report['datasets'].items():
 for arm in ['RDBPFN','TabICLv1.1']:
  for seed,auc in enumerate(v['models'][arm]['per_seed']):records.append(dict(database=d,arm=arm,seed=seed,auc=auc,test_rows=v['test_rows'],metric='AUROC'))
for _ in range(100):
 rows=copy.deepcopy(records);rng.shuffle(rows)
 assert paired_gains(rows,['rel-f1','rel-trial'],list(range(10)))==report['paired']
 for row in rows:row['test_rows']=1 if row['database']=='rel-f1' else 1000000
 assert database_macro(paired_gains(rows,['rel-f1','rel-trial'],list(range(10))))==report['macro']
# Re-pin corrupt packets to force semantic checks past the outer hash layer.
corruptions=0
with tempfile.TemporaryDirectory(prefix='l168-corrupt-') as tmp:
 root=Path(tmp)
 for name in pins['files']:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,p)
 receipt_name='evidence/l168/full-1/receipt.json';rp=root/receipt_name;original_receipt=rp.read_bytes()
 for mode in ['missing','duplicate','wrong_labels','wrong_support','wrong_keys','bad_hash']:
  changed=copy.deepcopy(pins);receipt=json.loads(original_receipt);pred_name='evidence/l168/full-1/RDBPFN-1.npz';pp=root/pred_name;original_pred=pp.read_bytes()
  if mode=='missing':receipt['records'].pop()
  elif mode=='duplicate':receipt['records'].append(receipt['records'][0])
  elif mode=='bad_hash':pp.write_bytes(original_pred+b'bad')
  else:
   with np.load(pp) as a:arrays={k:a[k] for k in a.files}
   if mode=='wrong_labels':arrays['label']=1-arrays['label']
   elif mode=='wrong_support':arrays['support_keys'][0]=arrays['support_keys'][1]
   elif mode=='wrong_keys':arrays['keys'][0]=arrays['keys'][1]
   np.savez_compressed(pp,**arrays);digest=hashlib.sha256(pp.read_bytes()).hexdigest();changed['files'][pred_name]=digest
   for row in receipt['records']:
    if row['arm']=='RDBPFN' and row['seed']==1:row['sha256']=digest
  rp.write_text(json.dumps(receipt));changed['files'][receipt_name]=hashlib.sha256(rp.read_bytes()).hexdigest()
  try:audit168(root,changed,keyed_auc,paired_gains,database_macro)
  except ValueError:corruptions+=1
  else:raise AssertionError('Corruption accepted: '+mode)
  pp.write_bytes(original_pred);rp.write_bytes(original_receipt)
result=dict(status='PASS',raw_predictions_independently_rescored=raw_count,rank_sklearn_pairwise_cases=200,max_auc_error=max_error,
 label_complement_keys_checked=label_matches,regime_cases=regime_cases,permutation_and_weighting_cases=100,incorrect_learner_functions_rejected=3,
 evidence_corruptions_rejected=corruptions,original_source_hashes_checked=len(manifest['source_files']),fresh_inference_runs=30,reused_inference_runs=30)
(P/'_verify_l168_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
