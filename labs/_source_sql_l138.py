"""Execute publication-era SQL on real archived query histories."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('amazon_source',P/'sources/l138/relbench__tasks__amazon.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);task=mod.UserChurnTask.__new__(mod.UserChurnTask)
samples=json.loads((P/'evidence/l138/samples.json').read_text());count=0
for rows in samples.values():
 for row in rows:
  t=pd.Timestamp(row['cutoff']);review=pd.DataFrame({'customer_id':[0]*len(row['events']),'review_time':t+pd.to_timedelta(row['events'],unit='D')})
  db=SimpleNamespace(table_dict={k:SimpleNamespace(df=v) for k,v in {'review':review,'customer':pd.DataFrame({'customer_id':[0]}),'product':pd.DataFrame()}.items()})
  actual=task.make_table(db,pd.Series([t])).df
  assert len(actual)==1 and int(actual.churn.iloc[0])==row['target'];count+=1
r={'status':'PASS','real_query_examples':count,'check':'Execute pinned upstream task SQL on individual real-customer event histories; compare stored labels'}
(P/'_sql_source_l138_results.json').write_text(json.dumps(r,indent=2));print(r)
