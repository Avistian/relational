"""Save complete official query keys/labels independently of cloud predictions."""
import sys,json,hashlib,importlib.metadata,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b14/relarena';E=P/'evidence/b14';sys.path.insert(0,str(S/'src'))
from relarena.dataset import RelBenchDatasetTask
s=RelBenchDatasetTask('rel-f1','driver-dnf',download=False);t=s.task;rows={}
for split in ['train','val','test']:
 df=t.get_table(split,mask_input_cols=False).df
 rows[split]=[dict(entity=int(e),date=int(d),label=int(y)) for e,d,y in zip(df[t.entity_col],df[t.time_col].astype('int64'),df[t.target_col])]
 assert len({(r['entity'],r['date']) for r in rows[split]})==len(rows[split])
(E/'official-keys.json').write_text(json.dumps(rows,indent=2)+'\n')
(E/'paper-cpu-environment.json').write_text(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},indent=2)+'\n')
print({k:len(v) for k,v in rows.items()})
