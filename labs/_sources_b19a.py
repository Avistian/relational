"""Pin primary sources and historical release candidate; no model execution."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent/'sources/b19a';P.mkdir(parents=True,exist_ok=True)
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'B19a-source-audit'}),timeout=45) as r:return r.read()
prior=json.loads((P/'manifest.json').read_text()) if (P/'manifest.json').exists() else {'files':[]}
for entry in prior['files']:
 path=P/entry['file']
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('SOURCE_HASH_MISMATCH '+entry['file'])
manifest=[entry for entry in prior['files'] if entry['file'].startswith('output/')]
def save(name,url):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(get(url))
 manifest.append(dict(file=name,url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 return p.read_bytes()
for repo in ['ScoringBench','ScoringBenchOutput']:
 save(repo+'-history.json',f'https://api.github.com/repos/jonaslandsgesell/{repo}/commits?per_page=100')
hist=json.loads((P/'ScoringBench-history.json').read_text());C=next(x['sha'] for x in hist if x['sha'].startswith('ca1660023ba4'))
tree=json.loads(save('code-tree.json',f'https://api.github.com/repos/jonaslandsgesell/ScoringBench/git/trees/{C}?recursive=1'))
O=next(x['sha'] for x in tree['tree'] if x['path']=='output');print('PIN',C,O)
save('output-tree.json',f'https://api.github.com/repos/jonaslandsgesell/ScoringBenchOutput/git/trees/{O}?recursive=1')
for name,url in [('paper.html','https://arxiv.org/html/2603.29928v3'),('distributional.html','https://arxiv.org/html/2603.08206v1')]:save(name,url)
for x in tree['tree']:
 p=x['path']
 if x['type']=='blob' and (p.endswith('.py') or p in ['README.md','requirements.txt','.gitmodules']):save('code/'+p,f'https://raw.githubusercontent.com/jonaslandsgesell/ScoringBench/{C}/{p}')
(P/'manifest.json').write_text(json.dumps(dict(code_commit=C,output_commit=O,selection='Last source commit on arXiv v3 submission day; candidate historical identity, not proof of paper provenance',files=manifest),indent=2)+'\n')
print('Archived',len(manifest),'files')
