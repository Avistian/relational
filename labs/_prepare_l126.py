"""Pin the pre-publication beta source; bounded HTTP availability probes only."""
import concurrent.futures, hashlib, json, subprocess, urllib.request, urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent
REV='0433616ee94003fb15a4ac4d633e499d0f129077'
REPO=Path('/tmp/l125-relbench-upstream')
if not REPO.exists():
 REPO=Path('/tmp/l126-relbench-upstream')
 if not REPO.exists():subprocess.run(['git','clone','--filter=blob:none','https://github.com/stanford-star/relbench.git',str(REPO)],check=True)
files=subprocess.check_output(['git','ls-tree','-r','--name-only',REV],cwd=REPO,text=True).splitlines()
manifest={'paper':'https://arxiv.org/html/2312.04615v1','revision':REV,'selection':'Last reachable commit before 2023-12-08; not author-attested paper experiment identity','files':{}}
for file in files:
 if not (file.startswith('relbench/') or file in ['LICENSE','README.md','pyproject.toml','examples/train.py','examples/baseline.py','examples/xgboost_baseline.py','examples/text_embedder.py']):continue
 raw=subprocess.check_output(['git','show',REV+':'+file],cwd=REPO)
 dest=P/'sources/l126/beta'/file;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 manifest['files'][str(dest.relative_to(P))]={'sha256':hashlib.sha256(raw).hexdigest(),'url':f'https://raw.githubusercontent.com/stanford-star/relbench/{REV}/{file}'}
(P/'_sources_l126.json').write_text(json.dumps(manifest,indent=2)+'\n')
urls=[f'https://relbench.stanford.edu/{base}/rel-stackex/{tail}' for base in ['staging_data','data','download'] for tail in ['db.zip','tasks/rel-stackex-engage.zip']]+['https://relbench.stanford.edu/data/relbench-forum-raw.zip','https://web.archive.org/cdx/search/cdx?url=relbench.stanford.edu/staging_data/rel-stackex/db.zip&output=json&filter=statuscode:200&collapse=digest']
def probe(url):
 try:
  with urllib.request.urlopen(url,timeout=20) as r:return {'url':url,'status':r.status,'final_url':r.url,'prefix':r.read(256).decode(errors='replace')}
 except urllib.error.HTTPError as e:return {'url':url,'status':e.code,'final_url':e.url}
 except Exception as e:return {'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:probes=list(ex.map(probe,urls))
audit={'date':'2026-09-27','status':'AUDITED','archive_probes':probes,'expected_sha256':{'db.zip':'dfb84faa4918c6c4ecac791a69a30a477a7bee097d7295d48c78ceb8f59c997c','engage.zip':'9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18','raw.zip':'ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f'},'historical_full_contract':'NOT_RUN','historical_identity':'NOT_ESTABLISHED','numerical_paper_target':'NOT_APPLICABLE: beta v1 has no reported predictive-score table','spend_usd':0}
(P/'_paper_audit_l126_results.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
