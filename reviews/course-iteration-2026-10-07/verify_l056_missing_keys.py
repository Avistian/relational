"""A missing evaluation identity must not silently disappear during aggregation."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'labs'))
import pandas as pd
from relkit.leaderboard import aligned_errors
base=pd.DataFrame([dict(dataset='d',fold=0,arm=a,metric='rmse',metric_error=e,imputed=False) for a,e in [('A',2.),('B',1.)]])
accepted=[]
for key in ['dataset','fold','metric']:
 bad=base.copy();bad.loc[0,key]=None
 try:aligned_errors(bad,['A','B'])
 except ValueError:pass
 else:accepted.append(key)
assert not accepted, 'Missing keys accepted: '+str(accepted)
print('PASS: missing dataset, fold and metric rejected')
Path(__file__).with_name('l056-missing-keys.json').write_text(json.dumps({'status':'PASS','rejected_null_fields':['dataset','fold','metric']},indent=2)+'\n')
