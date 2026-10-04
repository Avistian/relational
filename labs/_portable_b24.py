"""Deterministic builds and independent execution of the downloadable packet."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='b24-architecture-thesis-defense'
paths=[R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html',R/'reference/b24-proposal-template.html',P/f'{S}.ipynb',P/'evidence/b24/portable-packet.zip',*sorted((P/'figures/b24').glob('*.png'))]
def hashes():return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
subprocess.run([sys.executable,str(P/'_build_b24.py')],check=True);first=hashes()
subprocess.run([sys.executable,str(P/'_build_b24.py')],check=True);assert hashes()==first
with tempfile.TemporaryDirectory(prefix='b24-packet-') as td:
 with zipfile.ZipFile(P/'evidence/b24/portable-packet.zip') as z:z.extractall(td)
 code="import json;from pathlib import Path;from _audit_b24 import replay;from _test_b24 import checks;assert checks()=='PASS';r=replay('b23-packet.zip',json.loads(Path('source-identity.json').read_text()));assert r==json.loads(Path('report.json').read_text());print('standalone packet PASS')"
 subprocess.run([sys.executable,'-c',code],cwd=td,check=True)
out=dict(status='PASS',deterministic_files=first,standalone_packet=True)
(P/'_portable_b24_results.json').write_text(json.dumps(out,indent=2)+'\n');print('Deterministic build PASS',len(first),'files')
