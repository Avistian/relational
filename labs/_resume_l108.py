"""Exercise real completed-result reuse and rejection under a changed helper."""
import json,shutil,tempfile
from pathlib import Path
import _run_l108 as runner
P=Path(__file__).resolve().parent
source=P/'results/l108/seed-0';original=json.loads((source/'result.json').read_text());assert original['status']=='PILOT'
with tempfile.TemporaryDirectory(prefix='l108-resume-') as tmp:
 root=Path(tmp)
 for name in runner.FILES:
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,target)
 out=root/'cached';shutil.copytree(source,out)
 before=(out/'result.json').stat().st_mtime_ns
 old=runner.P;runner.P=root
 try:
  same=runner.run(0,P/'results/l103/gpu',P/'l103-cache',out,'cpu',True,240)
  assert same==original and (out/'result.json').stat().st_mtime_ns==before
  helper=root/'relkit/sampling_l108.py';helper.write_text(helper.read_text().replace("side='left'","side='right'"))
  try:runner.run(0,P/'results/l103/gpu',P/'l103-cache',out,'cpu',True,240)
  except AssertionError as e:assert 'Resume identity mismatch' in str(e)
  else:raise AssertionError('Changed boundary helper reused completed predictions')
 finally:runner.P=old
r={'status':'PASS','unchanged_result_reuse':'EXACT_NO_REWRITE','changed_boundary_helper':'REJECTED','scope':'Completed local pilot cache; full run uses identical reuse guard'}
(P/'_resume_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
