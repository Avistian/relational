"""Preserve supplemental primary-source examples; never merge into API coverage."""
import hashlib,json,time,urllib.request,urllib.error
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/l188';S.mkdir(parents=True,exist_ok=True)
items=[('2609.25541','A JEPA Recipe for Tabular Foundation Models','2026-09-22','failure_mode','INCLUDE','The abstract reports an unfavorable latent-objective comparison and one run per arm. Read training-budget and dataset-identity details before generalizing.'),('2609.02766','Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit','2026-09-02','failure_mode','INCLUDE','The abstract raises unit and deterministic-limit failures. Track as a limitation to investigate, not evidence of failure on relational databases.'),('2609.03880','Xiaomi-TabLDM: A Tabular Foundation Model Technical Report','2026-09-03','baseline','INCLUDE','A potential tabular comparator with reported benchmark gains. Check artifacts, fitting budgets and evaluation versions before a matched comparison.')]
if (S/'supplemental-log.json').exists():raise SystemExit('Refusing to overwrite a frozen supplemental snapshot')
rows=[]
for ident,title,date,role,decision,reason in items:
    url='https://arxiv.org/abs/'+ident;path=S/(ident+'.html');receipt=dict(url=url,retrieved_at=datetime.now(timezone.utc).isoformat())
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RelationalCourse-L188/1.0'}),timeout=20) as r:raw=r.read();receipt.update(status=r.status,final_url=r.url)
    except urllib.error.HTTPError as e:raw=e.read();receipt.update(status=e.code,error=str(e))
    except Exception as e:raw=b'';receipt.update(status=None,error=repr(e))
    path.write_bytes(raw);receipt.update(file=path.name,sha256=hashlib.sha256(raw).hexdigest())
    matched=receipt['status']==200 and title.lower() in raw.decode(errors='replace').lower()
    rows.append(dict(id=ident,title=title,submitted=date,version='v1 verified on official abstract page',scope='SUPPLEMENTAL_PRIMARY_PAGE; NOT_API_QUERY_COVERAGE',decision=decision if matched else 'DEFER',relevance=role,reason=reason,claim_status='REPORTED_NOT_REPRODUCED',artifact_audit='NOT_CHECKED',next_action='Read methods, locate released code/data, and fill protocol comparison before promotion to core.',receipt=receipt))
    time.sleep(3)
(S/'supplemental-log.json').write_text(json.dumps(rows,indent=2)+'\n')
print([(r['id'],r['receipt']['status'],r['decision']) for r in rows])
