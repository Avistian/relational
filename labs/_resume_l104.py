"""Exercise complete-result reuse and rejection of changed code or predictions."""
import json,shutil,tempfile,time
from pathlib import Path
import torch
import _run_l104 as runner
P=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1)
 with tempfile.TemporaryDirectory(prefix='l104-resume-') as tmp:
  root=Path(tmp);(root/'relkit').mkdir()
  for name in ['_inputs_l104.json','_run_l104.py','relkit/tgat_l103.py','relkit/leakage_l104.py']:
   shutil.copy2(P/name,root/name)
  runner.P=root
  kwargs={'seed':0,'checkpoint_root':P/'results/l103/gpu','data_dir':P/'l103-cache','output':root/'results','pilot':True,'max_seconds':120}
  first=runner.run(**kwargs);path=root/'results/predictions.npz';original=path.read_bytes()
  before=path.stat().st_mtime_ns;again=runner.run(**kwargs)
  assert first==again and path.stat().st_mtime_ns==before,'Completed result was recomputed'
  helper=root/'relkit/leakage_l104.py';code=helper.read_text();helper.write_text(code.replace('lookahead=86400','lookahead=86401'))
  try:
   try:runner.run(**kwargs)
   except AssertionError as e:assert 'input/code/runtime changed' in str(e)
   else:raise AssertionError('Changed helper accepted')
  finally:helper.write_text(code)
  path.write_bytes(original+b'corruption')
  try:runner.run(**kwargs)
  except AssertionError as e:assert 'prediction artifact changed' in str(e)
  else:raise AssertionError('Changed predictions accepted')
 report={'status':'PASS','same_identity_resume':'EXACT_NO_RECOMPUTE','changed_helper_default':'REJECTED','changed_prediction_bytes':'REJECTED','scope':'CPU pilot resume behavior; full GPU inference is separate'}
 (P/'_resume_l104_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
