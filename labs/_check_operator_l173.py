"""Validate overwrite refusal and fail-closed data authentication without new fits."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from relkit.multitask_l173 import load_packet
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l173'
paths=[E/'report.json',E/'cell-0/epoch-1.pt']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
proc=subprocess.run([sys.executable,str(P/'_run_l173.py')],cwd=R,capture_output=True,text=True)
assert proc.returncode!=0 and 'Existing run evidence' in proc.stderr
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
with tempfile.TemporaryDirectory(prefix='l173-corrupt-') as tmp:
    d=Path(tmp);(d/'population.npz').write_bytes(b'wrong bytes')
    (d/'manifest.json').write_text(json.dumps({'files':{'population.npz':'0'*64}}))
    try:load_packet(d)
    except ValueError as exc:assert 'Input hash mismatch' in str(exc)
    else:raise AssertionError('Corrupt population accepted')
for script in ['_run_l173.py','_verify_l173.py']:
    proc=subprocess.run([sys.executable,str(P/script),'--help'],cwd=R,capture_output=True,text=True)
    assert proc.returncode==0 and 'usage:' in proc.stdout
result=dict(status='PASS',overwrite_refusal='PASS',original_report_and_checkpoint='UNCHANGED',corrupted_packet='REJECTED',fresh_run_and_verification_cli='PASS')
(P/'_operator_l173_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
