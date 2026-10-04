"""Check repeatable generation without treating notebook execution as deterministic metadata."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='b20-curriculum-order'
files=[R/'lessons'/f'{S}.html',R/'reference'/f'{S}.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb',P/'evidence/b20/portable-packet.zip']+list((P/'figures/b20').glob('*.png'))
def digest():return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
subprocess.run([sys.executable,str(P/'_build_b20.py')],check=True);a=digest()
subprocess.run([sys.executable,str(P/'_build_b20.py')],check=True);b=digest();assert a==b
(P/'_build_b20_results.json').write_text(json.dumps(dict(status='PASS',deterministic_files=len(a),files=a),indent=2)+'\n');print('Deterministic builder PASS:',len(a),'files')
