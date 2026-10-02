"""Freeze exact input bytes once; collection does not certify novelty."""
import concurrent.futures,datetime,hashlib,json,shutil,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l189';Q=E/'packet'
sources={
 'survey':('https://arxiv.org/html/2506.16654v1','Survey: temporal, heterogeneous and foundation-model challenges; a 2025 starting map, not a current novelty certificate.'),
 'relgnn':('https://arxiv.org/html/2502.06784v2','Composite message passing already exists; the proposed contribution must isolate its role in a foundation learner.'),
 'rdbpfn':('https://arxiv.org/html/2603.03805v5','Relational synthetic priors and DFS plus an ICL predictor already exist; generator and predictor changes differ.'),
 'rt':('https://arxiv.org/html/2510.06377v1','Schema-spanning relational pretraining already exists; general cross-database transfer is not a new idea.'),
 'temporal':('https://arxiv.org/html/2609.35219v1','Temporal pretraining with supervised controls already exists. Introduction specifies fine-tuning within the same database.'),
 'relarena':('https://arxiv.org/html/2608.16319v2','Standardized relational evaluation and strong flattened baselines already exist; do not claim benchmarking itself is new.')}
def fetch(item):
 key,(url,note)=item;path=Q/'sources'/(key+'.html');stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if path.exists():raise ValueError('Refusing to refresh frozen source '+key)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'relational-course-research-audit/1.0'}),timeout=25) as res:
   data=res.read();status=res.status;final=res.url
  if status!=200 or b'<html' not in data.lower():raise ValueError('Not an HTML source')
  path.write_bytes(data)
  return dict(id=key,url=url,final_url=final,retrieved_at=stamp,status=status,file='sources/'+path.name,sha256=hashlib.sha256(data).hexdigest(),scope=note,review='PRIMARY_TEXT',novelty='NOT_ESTABLISHED')
 except Exception as exc:return dict(id=key,url=url,retrieved_at=stamp,status='FAILED',error=str(exc),scope=note)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,sources.items()))
 (Q/'sources.json').write_text(json.dumps(records,indent=2)+'\n')
 for l in [169,177,182,183,184]:
  origin=P/f'evidence/l{l}/report.json';shutil.copyfile(origin,Q/'inherited'/f'l{l}-report.json')
 origin=P/'evidence/l188/packet/collection.json';shutil.copyfile(origin,Q/'inherited/l188-collection.json')
 inheritance=[]
 for path in sorted((Q/'inherited').glob('*.json')):
  inheritance.append(dict(file='inherited/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Frozen report/receipt only; underlying predictions, caches and models are not replayed in L189.'))
 (Q/'inherited.json').write_text(json.dumps(inheritance,indent=2)+'\n')
 print([(r['id'],r['status']) for r in records])
 if any(r['status']!=200 for r in records):raise SystemExit('INCOMPLETE_SOURCE_COLLECTION')
