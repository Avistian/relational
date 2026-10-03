"""Independent original-primitive execution, all-query identity, and packet rejection."""
import ast,hashlib,json,random,tempfile,shutil,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder
from _audit_l192 import audit
from _test_l192 import checks
from relkit.setup_l192 import keyed_rows,available_history,select_candidate
P=Path(__file__).resolve().parent;E=P/'evidence/l192';Q=E/'packet';S=P/'sources/l192'
manifest=json.loads((E/'input-manifest.json').read_text());report=audit(Q,manifest,keyed_rows,available_history,select_candidate)
assert report==json.loads((E/'report.json').read_text())
checks(keyed_rows,available_history,select_candidate)
wrong_functions=[(lambda rows:{r['entity']:r for r in rows},available_history,select_candidate),(keyed_rows,lambda rows,t:[r for r in rows if r['cutoff']<t],select_candidate),(keyed_rows,available_history,lambda records,expected:expected[0])]
for trio in wrong_functions:
 try:checks(*trio)
 except AssertionError:pass
 else:raise AssertionError('Incorrect learner function accepted')
ledger=json.loads((S/'source-ledger.json').read_text())
for name,h in ledger['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
with zipfile.ZipFile(S/'task.zip') as z:
 for name in z.namelist():
  if name.endswith('.parquet'):assert z.read(name)==(E/'task'/Path(name).name).read_bytes()
# Independent direct parquet -> full keys comparison, no reliance on task-stats.json.
allrows=json.loads((Q/'queries.json').read_text());keys=[]
for split in ['train','val','test']:
 df=pd.read_parquet(E/'task'/(split+'.parquet'))
 direct={(int(x.nct_id),int(pd.Timestamp(x.timestamp).value//86400000000000),int(x.outcome)) for x in df.itertuples()}
 saved={(x['entity'],x['cutoff'],x['label']) for x in allrows if x['split']==split}
 assert direct==saved and len(direct)==len(df)
 keys.extend((x.nct_id,x.timestamp) for x in df.itertuples())
assert len(keys)==len(set(keys))==13779
# Independent integer interval count for all distinct official cutoffs.
train=[x for x in allrows if x['split']=='train']
for c in report['clock_audit']:
 assert c['unavailable_past_rows']==sum(x['cutoff']<c['cutoff']<x['cutoff']+365 for x in train)==0
source=(S/'rdblearn/rdblearn/preprocessing.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='SafeLabelEncoderTransformer')
primitive=ast.get_source_segment(source,node);exec(primitive,globals())
rng=random.Random(192)
for _ in range(100):
 count=rng.randint(3,12);names=[f'b{i:02d}' for i in range(count)]
 enc=SafeLabelEncoderTransformer().fit(pd.DataFrame({'category':names}))
 before=enc.transform(pd.DataFrame({'category':names})).category.tolist()
 enc.transform(pd.DataFrame({'category':['a']}))
 after=enc.transform(pd.DataFrame({'category':names})).category.tolist()
 assert before==list(range(count)) and after==list(range(1,count+1))
# Fresh original encoder confirms same known query differs with batch composition.
enc=SafeLabelEncoderTransformer().fit(pd.DataFrame({'category':['b','c','d']}))
assert enc.transform(pd.DataFrame({'category':['b']})).iloc[0,0]==0
assert enc.transform(pd.DataFrame({'category':['a','b']})).iloc[1,0]==1
rejections=0
for bad in ['queries.json','before.csv','protocol.json']:
 with tempfile.TemporaryDirectory() as td:
  dest=Path(td)/'packet';shutil.copytree(Q,dest);(dest/bad).write_bytes((dest/bad).read_bytes()+b' ')
  try:audit(dest,manifest,keyed_rows,available_history,select_candidate)
  except ValueError:rejections+=1
  else:raise AssertionError('Corruption accepted')
assert rejections==3
result=dict(status='PASS',source_files=len(ledger['files']),official_queries=13779,randomized_original_encoder_cases=100,corrupt_packets_rejected=rejections,wrong_learner_functions_rejected=3,source_primitive_sha256=hashlib.sha256(primitive.encode()).hexdigest(),fresh_full_preprocessor='Executed separately by _preflight_l192.py',whole_model_inference='NOT_RUN',historical_identity='NOT_ESTABLISHED')
(P/'_verify_l192_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
