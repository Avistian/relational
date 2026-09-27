"""Test the entire reconstruction operator on original-source synthetic tables."""
import contextlib,io,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/l126/beta'))
import pandas as pd
from relbench.data import Dataset,Table
from relbench.tasks.stackex import EngageTask
from _source_check_l126 import as_db
from _check_l126 import fixture
from _recover_l126 import reconstruct
users,events,t=fixture();users.loc[users.Id==1,'CreationDate']=t-pd.Timedelta(days=3000)
events=pd.concat([events,pd.DataFrame({'user':[1,1],'time':[t-pd.Timedelta(days=2000),t+pd.Timedelta(days=900)]})],ignore_index=True)
db=as_db(users,events);dataset=Dataset(db,pd.Timestamp('2019-01-01'),pd.Timestamp('2021-01-01'),[EngageTask]);task=EngageTask(dataset,process=True)
with contextlib.redirect_stderr(io.StringIO()):
    archived={'train':task.train_table,'val':task.val_table,'test':task.test_table,'full_test':task._full_test_table}
r=reconstruct(db,archived);assert r['status']=='PASS'
archived['val'].df.loc[archived['val'].df.index[0],'contribution']^=1
try:reconstruct(db,archived)
except ValueError:pass
else:raise AssertionError('Corrupt archived label accepted')
report={'status':'PASS','scope':'SYNTHETIC_ONLY','complete_reconstruction_path':'PASS','corrupt_archive_label':'REJECTED','historical_archives':'NOT_RUN','fixture_rows':r['splits']}
(P/'_recovery_check_l126_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
