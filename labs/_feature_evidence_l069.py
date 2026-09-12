"""Independently reconstruct all Iris subset-control probabilities."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from _evidence_l069 import original_model,source_predict
ROOT=Path(__file__).resolve().parent

def check():
 path=ROOT/'_feature_control_l069_v2_results.json';r=json.loads(path.read_text());df=pd.read_csv(ROOT/'data/cache/l069-source/iris/iris.csv');x=df.iloc[:,:-1].to_numpy(dtype='float32');y=pd.Categorical(df.iloc[:,-1]).codes;tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=42)
 assert np.array_equal(tr,r['context_ids']) and np.array_equal(te,r['query_ids']) and np.array_equal(y[te],r['targets'])
 expected={(a,cols) for n in range(5) for cols in itertools.combinations(range(4),n) for a in ['v2','xgboost']};actual=[(v['arm'],tuple(v['columns'])) for v in r['rows']];assert len(actual)==32 and set(actual)==expected
 model,adapter=original_model();tree=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.3,subsample=1.,colsample_bytree=1.,n_jobs=1,tree_method='hist',random_state=42).fit(x[tr],y[tr]);checks=[]
 for row in r['rows']:
  q=x[te].astype(float);columns=row['columns'];q[:,columns]=x[tr].astype(float)[:,columns].mean(0)
  p=source_predict(model,x[tr],y[tr],q)[0] if row['arm']=='v2' else tree.predict_proba(q);saved=np.array(row['probabilities']);delta=float(abs(saved-p).max());assert np.allclose(p,saved,atol=8e-5,rtol=8e-5);assert row['accuracy']==float((saved.argmax(1)==y[te]).mean());checks.append(dict(arm=row['arm'],columns=columns,delta=delta))
 for summary in r['summary']:
  rows=[v for v in r['rows'] if v['arm']==summary['arm'] and v['removed_fraction']==summary['removed_fraction']];assert summary['subsets']==len(rows) and abs(summary['accuracy']-np.mean([v['accuracy'] for v in rows]))<1e-12
 for control in r['incremental_prediction_controls']:assert control['max_probability_delta']==0 and control['aligned_input_max_delta']==0
 report=dict(status='PASS',evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=32,predictions=960,attention_adapter=adapter,cases=checks,scope='Independent original source full-model and fresh XGBoost reconstruction for every Iris subset and clean baseline; exact IDs, matrix replacement, mean accuracy and added-column control ledger.')
 (ROOT.parent/'reviews/lesson-quality-audit-047-070/069-feature-evidence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
if __name__=='__main__':check()
