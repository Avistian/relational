"""Differential task check against the pinned original SQL, including absent queries."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
from relkit.trial_l139 import trial_target
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_trial',P/'sources/l139/relbench__tasks__trial.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
task=object.__new__(module.StudyOutcomeTask);rng=np.random.default_rng(139);base=pd.Timestamp('2020-01-01');studies=[];outcomes=[];analyses=[];records={}
for study in range(100):
 start=int(rng.choice([-30,0,1]));studies.append(dict(nct_id=study,start_date=base+pd.Timedelta(days=start)));records[study]=[start,[]]
 for index in range(6):
  key=6*study+index;day=int(rng.choice([-1,0,1,364,365,366]));p=rng.choice([np.nan,-.1,0,.05,.051,.5,1,1.1]);modifier=rng.choice([None,'>','<']);kind=rng.choice(['Primary','Secondary'])
  outcomes.append(dict(id=key,nct_id=study,outcome_type=kind));analyses.append(dict(id=key,nct_id=study,outcome_id=key,date=base+pd.Timedelta(days=day),p_value=p,p_value_modifier=modifier));records[study][1].append([day,None if np.isnan(p) else float(p),modifier,kind])
db=SimpleNamespace(table_dict={name:SimpleNamespace(df=pd.DataFrame(rows)) for name,rows in [('studies',studies),('outcomes',outcomes),('outcome_analyses',analyses)]})
actual=task.make_table(db,pd.Series([base])).df.set_index('nct_id').outcome.to_dict();expected={}
for study,(start,rows) in records.items():
 eligible,label=trial_target(start,rows,0)
 if eligible:expected[study]=label
assert actual==expected
r=dict(status='PASS',candidate_studies=100,analysis_rows=600,included=len(expected),excluded=100-len(expected),source='Pinned StudyOutcomeTask.make_table',scope='Synthetic differential SQL boundary and missingness test; full real archive audit separate')
(P/'_sql_source_l139_results.json').write_text(json.dumps(r,indent=2));print(r)
