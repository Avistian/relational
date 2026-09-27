from pathlib import Path
import modal
ROOT=Path('/home/avist/Projects/relational')
app=modal.App('l139-audit-diagnostic');image=modal.Image.debian_slim(python_version='3.11').pip_install('relbench==1.1.0','duckdb==1.1.3','pandas==2.2.3','numpy==1.26.4','pyarrow==18.1.0')
v=modal.Volume.from_name('l139-trial-evidence')
@app.function(image=image,cpu=2,memory=16384,timeout=300,retries=0,volumes={'/evidence':v})
def diag():
 import pandas as pd,json,time
 from relbench.datasets import get_dataset
 from relbench.tasks import get_task
 from relbench.base import Table
 start=time.perf_counter();root=Path('/evidence');d=get_dataset('rel-trial');d.cache_dir=str(root/'unpacked');db=d.get_db(upto_test_timestamp=False);task=get_task('rel-trial','study-outcome');studies=db.table_dict['studies'].df;outcomes=db.table_dict['outcomes'].df;oa=db.table_dict['outcome_analyses'].df
 result={}
 for split in ['train','val','test']:
  expected=Table.load(root/'unpacked/study-outcome'/f'{split}.parquet').df
  actual=task.make_table(db,pd.Series(sorted(expected.timestamp.unique()))).df
  e=expected.set_index(['timestamp','nct_id']);a=actual.set_index(['timestamp','nct_id']);extra=a.index.difference(e.index);missing=e.index.difference(a.index)
  result[split]=dict(expected=len(e),actual=len(a),extra=len(extra),missing=len(missing),extra_examples=[str(x) for x in extra[:5]],missing_examples=[str(x) for x in missing[:5]],expected_columns=str(expected.dtypes),actual_columns=str(actual.dtypes),timestamps=[str(x) for x in expected.timestamp.unique()])
 result['seconds']=time.perf_counter()-start;(root/'diagnostic.json').write_text(json.dumps(result,indent=2));v.commit();return result
@app.local_entrypoint()
def main():
 import json,fcntl
 p=ROOT/'labs/_budget_l139.json'
 with p.open('r+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);b=json.load(f);upper=300*(2*.0000131+16*.00000222);assert sum(r['upper_usd'] for r in b['reservations'])+upper+3<=10;b['reservations'].append(dict(phase='audit-diagnostic',upper_usd=upper));f.seek(0);json.dump(b,f,indent=2);f.truncate()
 print(diag.remote())
