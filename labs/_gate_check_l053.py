"""Exercise live notebook code, completed-seed resume, identity rejection and holdout boundaries."""
import contextlib,copy,io,json,os,tempfile
from pathlib import Path
import numpy as np
import nbformat
from _paper_repro_l053 import reproduce,identity

ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
nb=nbformat.read(ROOT/'solutions/0053-realmlp-strong-defaults.ipynb',as_version=4)
env={'__name__':'__main__'}
with contextlib.redirect_stdout(io.StringIO()):
    for cell in nb.cells:
        if cell.cell_type!='code' or '@colab-bootstrap' in cell.source:continue
        if 'live_results=run_suite' in cell.source:break
        exec(compile(cell.source,'<live notebook>','exec'),env)
before=identity(env['RealMLPS'],env['run_suite'],env['RobustSmooth'])
assert {'smooth_clip','robust_parameters','ntp_linear','coslog4'}<=set(before)
folder=Path(tempfile.mkdtemp(prefix='l053-live-'))
a=reproduce('smoke',folder,'cpu',env['RealMLPS'],env['run_suite'],env['RobustSmooth'])
buf=io.StringIO()
with contextlib.redirect_stdout(buf):b=reproduce('smoke',folder,'cpu',env['RealMLPS'],env['run_suite'],env['RobustSmooth'])
assert 'RESUME 0' in buf.getvalue() and a['runs']==b['runs']
original=env['smooth_clip'];exec('def smooth_clip(z):\n    return z',env)
try:reproduce('smoke',folder,'cpu',env['RealMLPS'],env['run_suite'],env['RobustSmooth'])
except ValueError:pass
else:raise AssertionError('Changed live student helper was silently accepted')
env['smooth_clip']=original

# No test-label dependence in validation selection. Same train/validation and RNG.
data=env['load_task']('california',160,64);changed=copy.deepcopy(data);changed['y']['test']+=100
with contextlib.redirect_stdout(io.StringIO()):
    first=env['fit_neural'](env['RealMLPS'],data,0,width=16,epochs=2,prep_class=env['RobustSmooth'])
    altered=env['fit_neural'](env['RealMLPS'],changed,0,width=16,epochs=2,prep_class=env['RobustSmooth'])
    trees=env['fit_trees'](data,0,trees=8,candidates=6)
    trees_changed=env['fit_trees'](changed,0,trees=8,candidates=6)
assert first['history']==altered['history'] and first['prediction']==altered['prediction']
assert first['error']!=altered['error']
for arm in trees:
    assert trees[arm]['selected']==trees_changed[arm]['selected']
    assert trees[arm]['search']==trees_changed[arm]['search']
    assert trees[arm]['prediction']==trees_changed[arm]['prediction']
result=dict(status='PASS',live_functions=sorted(before),resume='PASS',changed_helper_rejected='PASS',test_label_intervention='PASS',scope='Actual live solution definitions, tiny fits; completed-seed resume, changed smooth_clip rejection, held-out labels cannot change selected models')
(ROOT/'_gate_l053_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
