"""Fresh synthetic intervention through the unmodified released full preprocessor.

Requires pandas, sklearn, autogluon.features1.5.0, pydantic2 and FastDFS0.2.1.
No model checkpoints, benchmark inference, or task effect estimates.
"""
import hashlib,importlib.metadata,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l193';Q=P/'evidence/l193/packet'
sys.path[:0]=[str(S/'rdblearn'),str(S)]
from rdblearn.preprocessing import TabularPreprocessor
import rdblearn.preprocessing as module
import numpy as np,pandas as pd
assert Path(module.__file__).resolve()==S/'rdblearn/rdblearn/preprocessing.py'
source_ledger=json.loads((S/'source-ledger.json').read_text())
for name,h in source_ledger['files'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==h,name
train=pd.DataFrame({'category':['b','c','d']*4,'number':list(range(12))})
known=pd.DataFrame({'category':['b','c','d'],'number':[1,2,3]});observations=[]
for unknown in ['a','z','0','e']:
 pipeline=TabularPreprocessor().fit(train);fitted=pipeline.transform(train);before=pipeline.transform(known)
 probe=pd.DataFrame({'category':[unknown],'number':[1]});pipeline.transform(probe);after=pipeline.transform(known)
 fresh=TabularPreprocessor().fit(train);alone=fresh.transform(known.iloc[[0]]);fresh2=TabularPreprocessor().fit(train)
 together=fresh2.transform(pd.concat([probe,known.iloc[[0]]],ignore_index=True))
 observations.append(dict(unseen=unknown,before=before.category.tolist(),after=after.category.tolist(),numeric_before=before.number.tolist(),numeric_after=after.number.tolist(),query_alone=alone.category.tolist(),query_with_other=together.category.tolist(),classes_after=pipeline.pipeline.named_steps['label_encoder'].label_encoders_['category'].classes_.tolist(),changed_codes=int((before.category!=after.category).sum())))
result=dict(status='FAIL' if any(x['changed_codes'] for x in observations) else 'PASS',scope='Fresh synthetic full released preprocessing; no benchmark data or backend',train_categories=['b','c','d'],known_categories=['b','c','d'],observations=observations,source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),model_evaluations=0,task_impact='NOT_ESTABLISHED',historical_environment='NOT_ESTABLISHED',packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','autogluon.features','autogluon.common','pydantic','fastdfs']})
(Q/'preprocessing.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
