"""Original release data checks and full F1 target-history window audit; no model."""
import sys,json,time,traceback,importlib.metadata,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b14/relarena';E=P/'evidence/b14'
sys.path.insert(0,str(S/'src'));start=time.monotonic()
result=dict(status='INCOMPLETE',model_evaluations=0)
try:
 from relarena.dataset import RelBenchDatasetTask
 from relarena.checksums.checksum import _db_checksums,_label_checksums
 source=RelBenchDatasetTask('rel-f1','driver-dnf',download=True)
 observed={**_db_checksums(source),**_label_checksums(source)}
 expected=json.loads((S/'src/relarena/checksums/relbench_v1_checksums.json').read_text())['rel-f1/driver-dnf']
 result.update(checksums_observed=observed,checksums_expected=expected,checksum_match=observed==expected)
 task=source.task;tr=task.get_table('train').df;va=task.get_table('val').df;te=task.get_table('test',mask_input_cols=False).df
 result.update(rows=dict(train=len(tr),validation=len(va),test=len(te)),val_timestamp=str(task.dataset.val_timestamp),test_timestamp=str(task.dataset.test_timestamp),timedelta=str(task.timedelta))
 # Does the original history-at-anchor representation admit unfinished labels?
 # This is an availability policy audit; historical publication dates are unknown.
 import pandas as pd
 history=pd.concat([tr,va]);bad=[]
 for entity,g in history.groupby(task.entity_col):
  times=sorted(g[task.time_col].tolist())
  for anchor in times:
   for earlier in times:
    if earlier<anchor<earlier+task.timedelta:bad.append(dict(entity=int(entity),anchor=str(anchor),earlier=str(earlier),available_at=str(earlier+task.timedelta)))
 result.update(unfinished_history_pairs=len(bad),history_examples=bad[:8],status='PASS' if observed==expected else 'INCOMPLETE_DATA_IDENTITY')
except Exception as e:result.update(status='INCOMPLETE_ENVIRONMENT',error=repr(e),traceback=traceback.format_exc())
result['seconds']=time.monotonic()-start;result['packages']={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','relbench','fastdfs','tabpfn']}
(E/'paper-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
