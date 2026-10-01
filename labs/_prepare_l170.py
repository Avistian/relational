"""Freeze original replay inputs and versioned primary reading snapshots."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l170';E=P/'evidence/l170'
sources=[]
for name,url in [('rdblearn','https://arxiv.org/html/2602.18495v1'),('griffin','https://arxiv.org/html/2505.05568v1'),('vision','https://arxiv.org/html/2305.15321v1'),('rdbpfn','https://arxiv.org/html/2603.03805v5')]:
    path=S/(name+'.html')
    if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=45).read())
    sources.append(dict(name=name,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(S/'source-ledger.json').write_text(json.dumps(dict(sources=sources,checked='2026-10-01',role='Primary reading; no RDBLearn benchmark execution'),indent=2)+'\n')
pins=json.loads((P/'evidence/l169/audit-manifest.json').read_text())
files=dict(pins['files'])
extra=['evidence/l169/audit-manifest.json','evidence/l169/report.json','evidence/l164/report.json','sources/l170/source-ledger.json','_audit_l169.py','relkit/transfer_l167.py','relkit/scaling_l169.py']+['sources/l170/'+s['name']+'.html' for s in sources]
for name in extra:files[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
(E/'input-manifest.json').write_text(json.dumps(dict(experiment='L170 complete saved-evidence replay of L169 selected RDB-PFN v5 Tables 6–10',files=files,original_grid=dict(tasks=['rel-f1/driver-dnf','rel-trial/study-outcome'],contexts=[64,128,256,512,1024],models=['RDBPFN','RDBPFN_single','TabICLv1.1'],seeds=list(range(10))),new_inference='NOT_RUN',cloud_usd=0),indent=2)+'\n')
print('Pinned',len(files),'original inputs and source snapshots')
