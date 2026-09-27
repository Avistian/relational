"""Fetch evidence, then require every complete seed before aggregate scoring."""
import argparse,hashlib,json
from pathlib import Path
import modal,numpy as np
from sklearn.metrics import roc_auc_score
from relkit.trial_l139 import keyed_auc
P=Path(__file__).resolve().parent;E=P/'evidence/l139';v=modal.Volume.from_name('l139-trial-evidence')
def fetch(name,target=None):
 raw=b''.join(v.read_file(name));path=E/(target or name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return path

def collect(mode):
 if mode=='prepare':
  for name in ['prepared.json','data_identity.json','task_audit.json','queries.npz','samples.json','stypes.json']:fetch(name)
  return
 if mode=='pilot':
  for name in ['result.json','predictions.npz','data_identity.json','progress.json','sampled_trace.npz','source_parity.json']:fetch('pilot2/seed-99/'+name)
  return
 results=[];hashes={};checkpoints={};queries=np.load(E/'queries.npz');count=0
 for seed in range(5):
  for name in ['result.json','predictions.npz','data_identity.json','progress.json','sampled_trace.npz','source_parity.json']:
   path=fetch(f'full/seed-{seed}/{name}');hashes[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest()
  path=P/'results/l139/checkpoints'/f'seed-{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
  with path.open('wb') as f:
   for block in v.read_file(f'full/seed-{seed}/selected.pt'):f.write(block);h.update(block);n+=len(block)
  checkpoints[str(seed)]=dict(sha256=h.hexdigest(),bytes=n,volume='l139-trial-evidence',path=f'full/seed-{seed}/selected.pt',published=False)
  result=json.loads((E/f'full/seed-{seed}/result.json').read_text());assert result['seed']==seed and result['epochs']==20 and result['status']=='COMPLETE' and len(result['history'])==20
  assert result['original_model_parity']['status']=='NUMERIC_CLOSE'
  assert result['best_epoch']==1+int(np.argmax([r['val']['roc_auc'] for r in result['history']]))
  assert all(r['queries']==len(queries['train_target']) and r['steps']==int(np.ceil(len(queries['train_target'])/512)) for r in result['history'])
  z=np.load(E/f'full/seed-{seed}/predictions.npz')
  for split in ['val','test']:
   for field in ['study','time','target']:assert np.array_equal(z[split+'_'+field],queries[split+'_'+field])
   keys=list(zip(queries[split+'_study'],queries[split+'_time']));pkeys=list(zip(z[split+'_study'],z[split+'_time']));pred=z[split+'_pred'];assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
   auc=keyed_auc(keys,queries[split+'_target'],pkeys,pred)
   assert abs(auc-roc_auc_score(queries[split+'_target'],pred))<1e-12
   assert abs(auc-result['scores'][split]['roc_auc'])<1e-12
   count+=len(pred)
  results.append(result)
 summary=dict(status='COMPLETE',seeds=list(range(5)),epochs=20,verified_predictions=count,audited_query_occurrences=sum(x['temporal_audit']['query_occurrences'] for x in results),metrics={},checkpoints=checkpoints,artifact_hashes=hashes,original_model_parity='NUMERIC_CLOSE on a real512-query batch per selected checkpoint; rtol1e-5 atol1e-6',historical_identity='NOT_ESTABLISHED')
 for split,target in [('val',.6818),('test',.6860)]:
  values=[r['scores'][split]['roc_auc'] for r in results];mean=float(np.mean(values));summary['metrics'][split]=dict(values=values,mean=mean,sample_sd=float(np.std(values,ddof=1)),paper_target=target,tolerance=.01,verdict='CLOSE' if abs(mean-target)<=.01 else 'OUTSIDE_TOLERANCE')
 (E/'training.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','full']);collect(p.parse_args().mode)
