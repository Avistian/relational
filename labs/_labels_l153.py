"""Independent interval/join reconstruction; no call to task.make_table."""
import json
from pathlib import Path
import pandas as pd

def independent_labels(output):
 from relbench.datasets import get_dataset
 from relbench.tasks import get_task
 ds=get_dataset('rel-trial');task=get_task('rel-trial','site-sponsor-run');db=ds.get_db(upto_test_timestamp=False)
 fs=db.table_dict['facilities_studies'].df;ss=db.table_dict['sponsors_studies'].df
 # Many facilities can belong to a study; group sponsor sets before traversing links.
 sponsor_sets=ss.groupby('nct_id')['sponsor_id'].agg(lambda x:set(int(v) for v in x.dropna() if int(v)<task.num_dst_nodes)).to_dict()
 result={}
 for split in ['train','val','test']:
  table=task.get_table(split,mask_input_cols=False).df;count=0
  for cutoff,group in table.groupby(task.time_col):
   active=fs[(fs.date>cutoff)&(fs.date<=cutoff+pd.Timedelta(days=365))];expected={}
   for site,study in active[['facility_id','nct_id']].itertuples(index=False,name=None):
    if pd.isna(site) or int(site)>=task.num_src_nodes:continue
    values=sponsor_sets.get(study,set())
    if values:expected.setdefault(int(site),set()).update(values)
   observed={int(site):set(map(int,targets)) for site,targets in group[[task.src_entity_col,task.dst_entity_col]].itertuples(index=False,name=None)}
   assert expected==observed,(split,str(cutoff),'independent labels mismatch');count+=len(group)
  result[split]=count
 report={'status':'PASS','independently_rebuilt_queries':result,'interval':'(cutoff,cutoff+365days]','join':'facility-study dates; sponsor-study nct_id; no sponsor date filter'}
 Path(output).write_text(json.dumps(report,indent=2));return report
