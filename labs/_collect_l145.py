"""Collect complete artifacts, independently check raw owner cutoffs and scores."""
from pathlib import Path
import json,hashlib,sys
import modal,numpy as np
from sklearn.metrics import mean_absolute_error
P=Path(__file__).resolve().parent;E=P/'evidence/l145';v=modal.Volume.from_name('l145-relgt-evidence')
def fetch(name):
    p=E/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b''.join(v.read_file(name)));return p
phase=sys.argv[1]
for suffix in ['-started.json','-cost.json']:fetch(phase+suffix)
if phase.startswith('prepare'):
    a=json.loads(fetch('prepared/audit.json').read_text());fetch('mechanism-pinned.json')
    for split in ['train','val','test']:
        p=fetch('prepared/'+split+'-audit.npz');z=np.load(p);mask=(z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None]);r=a['split_audits'][split]
        assert mask.sum()==r['future_token_occurrences'];assert mask.any(1).sum()==r['affected_queries']
    print({s:{k:r[k] for k in ['queries','overwritten_queries','future_token_occurrences','affected_queries']} for s,r in a['split_audits'].items()})
else:
    try:r=json.loads(fetch(phase+'/result.json').read_text())
    except Exception:
        fetch(phase+'-failure.txt');raise
    z=np.load(fetch(phase+'/predictions.npz'));assert abs(mean_absolute_error(z['val_target'],z['val_pred'])-r['scores']['val'])<1e-9
    a=np.load(E/'prepared/val-audit.npz');lookup={(int(i),int(t)):float(y) for i,t,y in zip(a['entity'],a['cutoff'],a['target'])}
    assert len(set(zip(z['val_entity'],z['val_cutoff'])))==len(z['val_entity'])
    assert all(lookup[(int(i),int(t))]==y for i,t,y in zip(z['val_entity'],z['val_cutoff'],z['val_target']))
    print(r)
