"""Run original released preprocessing and audit complete official task identity.

Synthetic categorical interventions test a required pipeline invariant. They do
not measure study-outcome feature frequency or AUROC impact.
"""
import hashlib,importlib.metadata,json,platform,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from rdblearn.preprocessing import TabularPreprocessor,SafeLabelEncoderTransformer
P=Path(__file__).resolve().parent;E=P/'evidence/l192';Q=E/'packet';Q.mkdir(exist_ok=True)
start=time.monotonic()
# Three categories avoid a two-value boolean conversion by AutoGluon.
train=pd.DataFrame({'category':['b','c','d']*4,'number':range(12)})
known=pd.DataFrame({'category':['b','c','d'],'number':[1,2,3]})
unseen=pd.DataFrame({'category':['a'],'number':[1]})
pipeline=TabularPreprocessor().fit(train)
fit_matrix=pipeline.transform(train)
before=pipeline.transform(known)
intervention=pipeline.transform(unseen)
after=pipeline.transform(known)
# Fresh two-batch comparison; same queried row, different other rows in its batch.
fresh=TabularPreprocessor().fit(train)
alone=fresh.transform(known.iloc[[0]])
together=fresh.transform(pd.concat([unseen,known.iloc[[0]]],ignore_index=True))
cols=[c for c in before if not before[c].equals(after[c])]
result=dict(status='FAIL' if cols else 'PASS',invariant='Known category codes must remain consistent with fitted support after unseen-query transformation',scope='Synthetic inputs; original complete TabularPreprocessor including AutoGluon1.5.0; no backend model',train_categories=train.category.tolist(),known_categories=known.category.tolist(),unseen_categories=unseen.category.tolist(),before=before.to_dict('list'),after=after.to_dict('list'),unseen=intervention.to_dict('list'),changed_columns=cols,query_alone=alone.to_dict('list'),query_with_unseen=together.to_dict('list'),fitted_classes=['b','c','d'],classes_after=pipeline.pipeline.named_steps['label_encoder'].label_encoders_['category'].classes_.tolist(),seconds=time.monotonic()-start)
assert cols==['category'],result
assert result['before']['category']==[0,1,2] and result['after']['category']==[1,2,3]
for name,df in [('train',train),('known',known),('unseen-input',unseen),('fit-matrix',fit_matrix),('before',before),('after',after)]:df.to_csv(Q/(name+'.csv'),index=False)
(Q/'preprocessing.json').write_text(json.dumps(result,indent=2)+'\n')
# Archive authentication is independent of label reconstruction from raw DB.
rows=[];stats={}
for split in ['train','val','test']:
 df=pd.read_parquet(E/'task'/(split+'.parquet'))
 assert not df.duplicated(['nct_id','timestamp']).any()
 stats[split]=dict(rows=len(df),unique_entities=int(df.nct_id.nunique()),entity_only_collisions=len(df)-int(df.nct_id.nunique()),positives=int(df.outcome.sum()),cutoffs=sorted(str(x) for x in df.timestamp.unique()))
 for r in df.itertuples():
  day=int(pd.Timestamp(r.timestamp).value//86400000000000)
  rows.append(dict(split=split,entity=int(r.nct_id),cutoff=day,label=int(r.outcome),available_at=day+365))
(Q/'queries.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
(Q/'task-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
for name in ['protocol.json']:(Q/name).write_bytes((E/name).read_bytes())
environment=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','autogluon.features','autogluon.common','pydantic','fastdfs','featuretools']},historical_environment='NOT_ESTABLISHED',cloud_usd=0,model_evaluations=0)
(E/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
print(json.dumps(result,indent=2));print(stats)
