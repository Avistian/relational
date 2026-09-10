"""Audit-only regression checks; historical training operators remain untouched."""
import importlib.util
from pathlib import Path
path=Path(__file__).with_name('_audit_l050.py')
assert path.exists(), 'Missing stable live identity and ensemble teaching functions.'
import json, subprocess, sys, tempfile
import numpy as np
from _audit_l050 import mean_predictions, implementation_identity
from relkit import checkpoint as c
from sklearn.metrics import roc_auc_score
p=np.array([[.9,.8,.7,.1],[.1,.2,.8,.9]])
y=np.array([0,0,1,1])
assert np.allclose(mean_predictions(p),[.5,.5,.75,.5])
assert roc_auc_score(y,mean_predictions(p))==.75
assert np.mean([roc_auc_score(y,row) for row in p])==.5
for bad in [[], [.2,.3], [[.2,np.nan]], [[-.1,.3]], [[.2,1.1]]]:
    try: mean_predictions(bad)
    except ValueError: pass
    else: raise AssertionError(('invalid probability matrix accepted',bad))
objects=vars(c).copy()
before=implementation_identity(objects)
c.reference_parity()
assert implementation_identity(objects)==before, 'Execution changed identity.'
original=c.CheckpointMLP.__init__
def changed(self,features,d=64):
    original(self,features,d+1)
c.CheckpointMLP.__init__=changed
assert implementation_identity(objects)!=before, 'Constructor changes must change identity.'
c.CheckpointMLP.__init__=original
objects['predict']=lambda model,x,device='cpu': np.zeros(len(x))
assert implementation_identity(objects)!=before, 'Prediction changes must change identity.'
command=[sys.executable,'-c','from relkit import checkpoint as c; from _audit_l050 import implementation_identity; print(implementation_identity(vars(c)))']
assert subprocess.check_output(command,text=True,cwd=path.parent).strip()==before, 'Fresh process identity drift.'
print('PASS: ensemble arithmetic, invalid inputs, execution-stable identity, fresh-process identity, constructor/predict changes.')
# Real bounded scale-up and unchanged resume; changed live prediction must refuse reuse.
from _paper_repro_l050 import run
objects=vars(c).copy()
def live_experiment(**kwargs):
    return c.run_comparison(**kwargs)
with tempfile.TemporaryDirectory(prefix='l050-resume-') as folder:
    identity=implementation_identity(objects)
    first=run('smoke',folder,experiment=live_experiment,implementation_id=identity)
    seed=Path(folder)/'seed-0.json';stamp=seed.stat().st_mtime_ns
    second=run('smoke',folder,experiment=live_experiment,implementation_id=implementation_identity(objects))
    assert second==first and seed.stat().st_mtime_ns==stamp
    objects['predict']=lambda model,x,device='cpu': np.zeros(len(x))
    try: run('smoke',folder,experiment=live_experiment,implementation_id=implementation_identity(objects))
    except ValueError as e: assert 'identity changed' in str(e)
    else: raise AssertionError('Changed helper silently reused completed seed.')
print('PASS: real smoke, unchanged completed-seed resume, changed-helper cache rejection.')
