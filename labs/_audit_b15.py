"""Adversarial independent checks on copied artifacts; originals stay immutable."""
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='b15-audit-') as td:
 root=Path(td);shutil.copyfile(P/'_verify_b15.py',root/'_verify_b15.py');(root/'evidence/b15').mkdir(parents=True)
 good=json.loads((P/'evidence/b15/diagnostic.json').read_text());cases={}
 for kind in ['wrong_truth','missing_world','changed_hidden_prediction']:
  j=json.loads(json.dumps(good))
  if kind=='wrong_truth':j['rules'][0]['truth']=1-j['rules'][0]['truth']
  elif kind=='missing_world':j['columns'].pop()
  else:j['interventions'][0]['safe']=1-j['interventions'][0]['safe']
  (root/'evidence/b15/diagnostic.json').write_text(json.dumps(j))
  r=subprocess.run([sys.executable,str(root/'_verify_b15.py')],capture_output=True)
  assert r.returncode!=0,kind;cases[kind]='REJECTED'
 shutil.copyfile(P/'_reproduce_b15.py',root/'_reproduce_b15.py');shutil.copytree(P/'sources',root/'sources',ignore=lambda d,ns:[x for x in ns if Path(d)==P/'sources' and x!='b15'])
 shutil.copyfile(P/'evidence/b15/paper-status.json',root/'evidence/b15/paper-status.json')
 r=subprocess.run([sys.executable,str(root/'_reproduce_b15.py'),'--phase','paper'],capture_output=True,text=True)
 assert r.returncode!=0 and 'NOT_RUN' in r.stderr;cases['unknown_paper_protocol']='REJECTED'
 f=root/'sources/b15/rdblearn/rdblearn/config.py';f.write_text(f.read_text()+'\n# changed\n')
 r=subprocess.run([sys.executable,str(root/'_reproduce_b15.py')],capture_output=True,text=True)
 assert r.returncode!=0 and 'SOURCE_HASH_MISMATCH' in r.stderr;cases['source_corruption']='REJECTED'
(P/'_audit_b15_results.json').write_text(json.dumps(dict(status='PASS',cases=cases),indent=2)+'\n');print(cases)
