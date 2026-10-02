"""Independent scalar oracle over every cell plus adversarial contract checks."""
import collections,hashlib,json,math,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from _audit_l172 import load172,replay172
from _check_l172 import check172
from relkit.tokenization_l172 import fit_column,encode_column,tokenize_table
P=Path(__file__).resolve().parent;manifest=json.loads((P/'evidence/l172/input-manifest.json').read_text())
assert check172(fit_column,encode_column,tokenize_table)=='PASS'
tables,schema=load172(P,manifest);cells=0;references=0
for name,frame in tables.items():
 spec=schema[name];clock=spec['time_col'];admit=[bool(v<pd.Timestamp('2005-01-01')) for v in frame[clock]] if clock else [False]*len(frame)
 tokens=tokenize_table(frame,spec['columns'],admit)
 for col,policy in spec['columns'].items():
  kind=policy['kind'];values=frame[col].tolist();out=tokens['columns'][col]
  train=[v for v,yes in zip(values,admit) if yes and not pd.isna(v)]
  if kind=='number':
   avg=math.fsum(float(v) for v in train)/len(train) if train else 0.
   sd=math.sqrt(math.fsum((float(v)-avg)**2 for v in train)/len(train)) if train else 1.
   sd=sd or 1.
  vocab={v:i+1 for i,v in enumerate(sorted(set(str(v) for v in train)))} if kind=='category' else {}
  for v,state,payload in zip(values,out['state'],out['payload']):
   if pd.isna(v):assert state=='MISSING' and payload==('' if kind in {'key','text'} else 0.)
   else:
    expected_state='UNKNOWN' if kind=='category' and str(v) not in vocab else 'VALUE'
    assert state==expected_state
    if kind=='number':assert math.isclose(payload,(float(v)-avg)/sd,rel_tol=1e-12,abs_tol=1e-12)
    elif kind=='category':assert payload==vocab.get(str(v),0)
    elif kind=='timestamp':assert math.isclose(payload,pd.Timestamp(v).timestamp()/86400,abs_tol=1e-10)
    else:assert payload==str(v)
   cells+=1
  if policy['role']=='foreign_key':
   target=policy['target_table'];pk=next(c for c,x in schema[target]['columns'].items() if x['role']=='primary_key')
   parents=set(str(v) for v in tables[target][pk].dropna())
   assert all(p in parents for p,s in zip(out['payload'],out['state']) if s=='VALUE')
   references+=sum(s=='VALUE' for s in out['state'])
report=json.loads((P/'evidence/l172/report.json').read_text())
assert replay172(P,manifest,fit_column,encode_column,tokenize_table)==report
assert cells==report['cells'] and references==227716
# Live checks must reject plausible wrong learner implementations.
def leaky_fit(v,k,a):return fit_column(v,k,[True]*len(v))
def ignore_mask(v,f,m=None):return encode_column(v,f)
def positional(frame,schema,admitted,masks=None):
 r=tokenize_table(frame,schema,admitted,masks);r['order']=list(frame.columns);return r
for funcs in [(leaky_fit,encode_column,tokenize_table),(fit_column,ignore_mask,tokenize_table),(fit_column,encode_column,positional)]:
 try:check172(*funcs)
 except AssertionError:pass
 else:raise AssertionError('Bad learner implementation passed')
# Reject corrupt evidence before loading or transforming rows.
for name in ['evidence/l172/schema.json','evidence/l171/db/results.parquet','evidence/l171/rel-f1-db.zip']:
 bad=json.loads(json.dumps(manifest));bad['files'][name]='0'*64
 try:load172(P,bad)
 except ValueError:pass
 else:raise AssertionError('Corrupt identity passed')
r=dict(status='PASS',independent_scalar_cell_checks=cells,independent_fk_references=references,complete_replay='EXACT',rejected_wrong_learner_functions=3,rejected_corrupt_manifests=3,numerical_tolerance=dict(rtol=1e-12,atol=1e-12),historical_availability='NOT_ESTABLISHED',whole_paper_reproduction='NOT_RUN')
(P/'_verify_l172_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
