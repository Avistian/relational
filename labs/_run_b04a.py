"""Complete frozen matrix or explicit incomplete result; refuses changed code on resume."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b04a';protocol=json.loads((E/'protocol.json').read_text())
files=['_worker_b04a.py','relkit/scaling_b04a.py','evidence/b04a/protocol.json','evidence/b04a/inputs.json']
contract={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files};lock=E/'execution-contract.json'
if lock.exists():
 if json.loads(lock.read_text())!=contract:raise SystemExit('Changed execution inputs/code: refuse resume')
else:lock.write_text(json.dumps(contract,indent=2)+'\n')
start=time.monotonic();completed=0
for config in protocol['configs']:
 path=E/'runs'/(config['name']+'.json')
 if path.exists():completed+=1;continue
 remaining=900-(time.monotonic()-start)
 if remaining<=0:raise SystemExit('INCOMPLETE_LOCAL_BUDGET_GATE')
 r=subprocess.run([sys.executable,str(P/'_worker_b04a.py'),config['name']],timeout=remaining,capture_output=True,text=True)
 if r.returncode:raise SystemExit(r.stderr)
 completed+=1
print('COMPLETE',completed,'configurations in',round(time.monotonic()-start,3),'seconds')
