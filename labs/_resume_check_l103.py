"""Completed-run reuse must accept unchanged code and reject changed helpers."""
import json,os,shutil,subprocess,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;python=P.parent/'.venv/bin/python'
with tempfile.TemporaryDirectory(prefix='l103-identity-') as tmp:
 root=Path(tmp);(root/'relkit').mkdir();shutil.copy(P/'_verify_l103.py',root/'_verify_l103.py');shutil.copy(P/'relkit/tgat_l103.py',root/'relkit/tgat_l103.py');shutil.copytree(P/'results/l103/smoke',root/'results/smoke')
 args=[str(python),str(root/'_verify_l103.py'),'--preset','smoke','--data',str(P/'data/l102'),'--output',str(root/'results')]
 env={**os.environ,'OMP_NUM_THREADS':'1'}
 passed=subprocess.run(args,capture_output=True,text=True,env=env);assert passed.returncode==0,passed.stderr
 source=root/'relkit/tgat_l103.py';source.write_text(source.read_text().replace("side='left'","side='right'"))
 rejected=subprocess.run(args,capture_output=True,text=True,env=env);assert rejected.returncode!=0 and 'Run identity changed' in rejected.stderr,rejected.stderr
 report={'status':'PASS','unchanged_completed_seed':'REUSED','changed_strict_boundary_helper':'REJECTED','mid_run_resume':'NOT_IMPLEMENTED; interrupted seeds restart'}
 (P/'_resume_check_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
