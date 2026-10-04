"""Authenticate all fresh predictions before using any learner comparison function."""
import hashlib,json
from pathlib import Path
import numpy as np
from relkit.exit_l200 import keyed_auc
from relkit.comparison_b23 import admit_grid,paired_comparison,claim_status

def audit(root,admit=admit_grid,paired=paired_comparison,claim=claim_status):
 root=Path(root);manifest=json.loads((root/'packet-manifest.json').read_text())
 for name,h in manifest['files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed evidence '+name)
 data=np.load(root/'packet/prepared.npz');records=[]
 for phase in ['pilot-1','full-1','baseline-1']:
  receipt=json.loads((root/phase/'receipt.json').read_text())
  if receipt['input_manifest_sha256']!=manifest['input_manifest_sha256']:raise ValueError('Wrong input manifest')
  for row in receipt['records']:
   if type(row['seed']) is not int or not 0<=row['seed']<10:raise ValueError('Invalid seed')
   path=root/phase/f"{row['arm']}-{row['seed']}.npz"
   if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed predictions')
   x=np.load(path);idx=data['support'][row['seed']]
   np.testing.assert_array_equal(x['keys'],data['test_keys']);np.testing.assert_array_equal(x['label'],data['y_test']);np.testing.assert_array_equal(x['support_keys'],data['train_keys'][idx])
   if len(set(map(tuple,x['support_keys'])))!=512 or set(map(tuple,x['support_keys']))&set(map(tuple,x['keys'])):raise ValueError('Invalid support')
   value=keyed_auc(data['test_keys'],data['y_test'],x['keys'],x['probability'])
   if abs(value-row['auc'])>1e-12:raise ValueError('Wrong metric receipt')
   records.append(dict(arm=row['arm'],seed=row['seed'],auc=value,rows=row['rows'],support=row['support']))
 counts=admit(records);targets={'RDBPFN':.7219,'RDBPFN_single':.6640,'TabICLv1.1':.7176};models={}
 for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1','Logistic']:
  values=[r['auc'] for r in sorted(records,key=lambda r:r['seed']) if r['arm']==arm]
  mean=float(np.mean(values));models[arm]=dict(mean=mean,sample_sd=float(np.std(values,ddof=1)),per_seed=values,lane='PUBLISHED_SELECTED' if arm in targets else 'COURSE_BASELINE',target=targets.get(arm),close=bool(abs(mean-targets[arm])<=.02) if arm in targets else None)
 comparisons=[paired(records,a,'RDBPFN') for a in ['RDBPFN_single','TabICLv1.1','Logistic']]
 return dict(experiment='B23-RDBPFN-F1-COMPARISON',execution='COMPLETE_SELECTED_RELEASE_EVALUATION',**counts,models=models,comparisons=comparisons,claims=claim(True,all(models[a]['close'] for a in targets),False,False),fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',full_dfs_regeneration='NOT_RUN',tolerance=.02)
if __name__=='__main__':
 e=Path(__file__).resolve().parent/'evidence/b23';r=audit(e);(e/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
