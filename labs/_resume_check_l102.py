"""Verify real completed-run reuse, then reject a changed helper before reuse."""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;PY=P.parent/'.venv/bin/python';out=P/'results/l102/smoke'
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('seed-*.json')}
cmd=[str(PY),str(P/'_verify_l102.py'),'--preset','smoke','--seeds','0,1,2','--output',str(out)]
a=subprocess.run(cmd,capture_output=True,text=True,check=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
assert '"epoch"' not in a.stdout,'Completed runs were unexpectedly retrained'
assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('seed-*.json')}
with tempfile.TemporaryDirectory(prefix='l102-mismatch-') as tmp:
 root=Path(tmp);(root/'relkit').mkdir();(root/'relkit/__init__.py').write_text('')
 source=(P/'relkit/tgn_l102.py').read_text();assert 'return torch.cat([own, other, edge,' in source
 (root/'relkit/tgn_l102.py').write_text(source.replace('return torch.cat([own, other, edge,','return torch.cat([other, own, edge,'))
 (root/'_verify_l102.py').write_text((P/'_verify_l102.py').read_text());(root/'data').symlink_to(P/'data')
 b=subprocess.run([str(PY),str(root/'_verify_l102.py'),'--preset','smoke','--seeds','0','--output',str(out)],capture_output=True,text=True)
 assert b.returncode!=0 and 'Run identity changed' in b.stderr,b.stderr
 assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('seed-*.json')}
r={'status':'PASS','unchanged_completed_runs':'Reused with identical artifacts and no epochs executed','changed_raw_message_helper':'Rejected before reuse by implementation hash','partial_epoch_resume':'NOT_SUPPORTED'}
(P/'_resume_check_l102_results.json').write_text(json.dumps(r,indent=2));print(r)
