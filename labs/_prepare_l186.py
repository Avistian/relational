"""Freeze original receipts, source pages and protocol before execution."""
import hashlib,json,shutil,urllib.request,sys
from pathlib import Path
from relkit.serving_l186 import CONFIG
P=Path(__file__).resolve().parent;E=P/'evidence/l186';S=P/'sources/l186';packet=E/'packet'
E.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
seal=json.loads((P/'evidence/l176/artifact-manifest.json').read_text())['files']
names=['_run_l176.py','evidence/l176/artifact-manifest.json']
for phase in ['pilot-1','remaining-1']:
 names += [p.relative_to(P).as_posix() for p in sorted((P/f'evidence/l176/{phase}').glob('*.json'))]
for name in names:
 src=P/name;data=src.read_bytes()
 if 'labs/'+name in seal:assert hashlib.sha256(data).hexdigest()==seal['labs/'+name],name
 dest=packet/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
urls={'huyen-realtime':'https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html','huyen-monitoring':'https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html','google-sre':'https://sre.google/sre-book/monitoring-distributed-systems/'}
sources=[]
for name,url in urls.items():
 dest=S/(name+'.html')
 if not dest.exists():
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 lesson source archive'})
  dest.write_bytes(urllib.request.urlopen(req,timeout=25).read())
 sources.append(dict(url=url,path=dest.relative_to(P).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),access_date='2026-10-02',role='Conceptual source, no published numerical target'))
(S/'source-ledger.json').write_text(json.dumps(dict(sources=sources,receipt_origin='L176 original sealed receipts; archive authentication is not historical clock truth',python=sys.version,protocol=CONFIG),indent=2)+'\n')
(E/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
files={p.relative_to(E).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(packet.rglob('*')) if p.is_file()}
(E/'input-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print('Frozen',len(files),'receipt/source inputs and',len(sources),'primary pages')
