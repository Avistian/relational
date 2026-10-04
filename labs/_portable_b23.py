"""Standalone archive execution plus reproducible build, preserving executed solution."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='b23-declared-comparison';E=P/'evidence/b23'
with tempfile.TemporaryDirectory(prefix='b23-cli-') as td:
 with zipfile.ZipFile(E/'portable-packet.zip') as z:z.extractall(td)
 for name in ['_audit_b23.py','_verify_b23.py']:
  subprocess.run([sys.executable,name],cwd=td,check=True,capture_output=True)
 assert json.loads((Path(td)/'evidence/b23/report.json').read_text())==json.loads((E/'report.json').read_text())
files=[P/f'{S}.ipynb',R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html',E/'portable-packet.zip',*sorted((P/'figures/b23').glob('*.png'))]
before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};sol=P/'solutions'/f'{S}.ipynb';saved=sol.read_bytes()
try:
 subprocess.run([sys.executable,str(P/'_build_b23.py')],cwd=R,check=True,capture_output=True)
 assert before=={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'Nondeterministic builder'
finally:sol.write_bytes(saved)
r=dict(status='PASS',standalone_archive_audit=True,standalone_independent_verifier=True,deterministic_builder_files=len(files));(P/'_portable_b23_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
