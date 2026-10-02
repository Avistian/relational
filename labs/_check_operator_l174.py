"""Fail closed on existing output and corrupt packets before launching any fit."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from relkit.finetune_l174 import load_adaptation_packet
P=Path(__file__).resolve().parent;E=P/'evidence/l174/runs'
paths=[E/'report.json',E/'freeze-0/epoch-1.pt'];before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
p=subprocess.run([sys.executable,str(P/'_run_l174.py'),'--output',str(E)],capture_output=True,text=True)
assert p.returncode!=0 and 'Refusing to overwrite' in p.stderr
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp);(d/'population.npz').write_bytes(b'corrupt');(d/'manifest.json').write_text(json.dumps(dict(files={'population.npz':'0'*64})))
 try:load_adaptation_packet(d)
 except ValueError as e:assert 'Input hash mismatch' in str(e)
 else:raise AssertionError('Corrupt population accepted')
for script in ['_run_l174.py','_verify_l174.py']:
 p=subprocess.run([sys.executable,str(P/script),'--help'],capture_output=True,text=True);assert p.returncode==0
r=dict(status='PASS',overwrite_refusal='PASS',prior_evidence='UNCHANGED',corrupt_population='REJECTED',cli_help='PASS')
(P/'_operator_l174_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
