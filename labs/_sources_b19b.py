"""Archive primary sources and historical candidates; do not run models."""
import hashlib,json,urllib.request,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent/'sources/b19b';P.mkdir(parents=True,exist_ok=True)
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'B19b-source-audit'}),timeout=45) as r:return r.read()
prior=json.loads((P/'manifest.json').read_text()) if (P/'manifest.json').exists() else {'files':[]}
for e in prior['files']:
 if hashlib.sha256((P/e['file']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('SOURCE_HASH_MISMATCH '+e['file'])
entries=[]
def save(name,url):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True)
 data=p.read_bytes() if p.exists() else get(url)
 p.write_bytes(data)
 return dict(file=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def one(name,url):
 e=save(name,url);entries.append(e);return json.loads((P/name).read_text())
jobs=[('paper.html','https://arxiv.org/html/2501.02945v4'),('figure4-1.svg','https://arxiv.org/html/2501.02945v4/main_result.svg')];pins={}
for label,repo in [('wrapper','PriorLabs/tabpfn-time-series'),('gift','SalesforceAIResearch/gift-eval')]:
 history=one(label+'-history.json',f'https://api.github.com/repos/{repo}/commits?until=2026-01-26T23:59:59Z&per_page=1');sha=history[0]['sha'];pins[label]=sha
 tree=one(label+'-tree.json',f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1')
 for x in tree['tree']:
  p=x['path']
  models=['Seasonal_Naive','Auto_Arima','Auto_ETS','Auto_Theta','DeepAR','TFT','TiRex','Toto_Open_Base_1.0','timesfm_2_0_500m','chronos_bolt_base','Moirai2','TabPFN-TS','sundial_base_128m','PatchTST','tabpfn_ts']
  keep=x['type']=='blob' and ((label=='wrapper' and (p.endswith('.py') or p.endswith('.toml') or p.endswith('.txt') or p.endswith('.md') or p in ['LICENSE','NOTICE'] or p.startswith('gift_eval/') and p.endswith('.json') or p.startswith('gift_eval/submission/'))) or (label=='gift' and (p in ['README.md','pyproject.toml','requirements.txt','dataset_properties.json'] or p.endswith('.py') and not p.startswith('results/') or any(p.lower().startswith('results/'+m.lower()+'/') for m in models))))
  if keep:jobs.append((label+'/'+p,f'https://raw.githubusercontent.com/{repo}/{sha}/{p}'))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for e in pool.map(lambda args:save(*args),jobs):entries.append(e)
entries={e['file']:e for e in prior['files']+entries}
pins={**prior.get('pins',{}),**pins}
(P/'manifest.json').write_text(json.dumps(dict(pins=pins,selection='Last commit by end of arXiv v4 submission date; historical candidate, not proof of original experiment identity',files=sorted(entries.values(),key=lambda x:x['file'])),indent=2)+'\n')
print(pins,'files',len(entries))
