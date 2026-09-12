"""All15 source Iris feature subsets and genuine added-column prediction control."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from relkit import openenv_l069_v2 as c
ROOT=Path(__file__).resolve().parent

def run():
 c.torch.set_num_threads(1);start=time.perf_counter();x,y,ids,meta=c.load_dataset(ROOT,'iris');context,query=train_test_split(np.arange(len(y)),test_size=.2,random_state=42);model,_=c.load_pretrained(c.ensure_checkpoint(ROOT));tree=XGBClassifier(**c.PROTOCOL['xgboost'],random_state=42).fit(x[context],y[context]);rows=[];base={};names=meta['features']
 for size in range(5):
  for cols in itertools.combinations(range(4),size):
   q=c.impute_features(x[context],x[query],cols)
   for arm in ['v2','xgboost']:
    p=c.predict_numeric(model,x[context],y[context],q)[0].numpy() if arm=='v2' else tree.predict_proba(q)
    if size==0:base[arm]=p
    rows.append(dict(arm=arm,columns=list(cols),removed_fraction=size/4,probabilities=p.tolist(),accuracy=float(accuracy_score(y[query],p.argmax(1)))))
 controls=[]
 augmented=np.column_stack([x[query],np.linalg.norm(x[query],axis=1)]);aligned=c.align_schema(augmented,names+['new_sensor'],names)
 for arm in ['v2','xgboost']:
  p=c.predict_numeric(model,x[context],y[context],aligned)[0].numpy() if arm=='v2' else tree.predict_proba(aligned);delta=float(abs(p-base[arm]).max());assert delta<1e-7;controls.append(dict(arm=arm,max_probability_delta=delta,aligned_input_max_delta=float(abs(aligned-x[query]).max())))
 summary=[dict(arm=arm,removed_fraction=size/4,subsets=sum(r['arm']==arm and len(r['columns'])==size for r in rows),accuracy=float(np.mean([r['accuracy'] for r in rows if r['arm']==arm and len(r['columns'])==size]))) for arm in ['v2','xgboost'] for size in range(5)]
 result=dict(status='PASS',dataset='iris',seed=42,context_ids=ids[context].tolist(),query_ids=ids[query].tolist(),targets=y[query].tolist(),source_sha256=meta['sha256'],kernel_identity=c.kernel_identity(vars(c),ROOT),operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,summary=summary,incremental_prediction_controls=controls,seconds=time.perf_counter()-start,scope='All15 nonempty Iris feature subsets, plus clean, on complete v2 and fixed XGBoost; no claim for larger all-subset suite')
 return result
if __name__=='__main__':
 p=ROOT/'_feature_control_l069_v2_results.json'
 if p.exists():raise FileExistsError('Preserve old measured control')
 r=run();p.write_text(json.dumps(r,indent=2)+'\n');print(r['summary'],r['incremental_prediction_controls'])
