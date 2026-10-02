"""Authenticate inherited full F1 data and freeze local protocol before execution."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l187';S=P/'sources/l187'
packet=E/'packet';(packet/'db').mkdir(parents=True,exist_ok=True);S.mkdir(parents=True,exist_ok=True)
inherited=json.loads((P/'evidence/l181/input-manifest.json').read_text())
for path in sorted((P/'evidence/l181/packet/db').glob('*.parquet')):
    key='db/'+path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==inherited['files'][key],key
    shutil.copy2(path,packet/key)
source=P/'sources/l181/upstream/relbench/datasets/f1.py'
ledger=json.loads((P/'sources/l181/source-ledger.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==ledger['files']['labs/sources/l181/upstream/relbench/datasets/f1.py']
shutil.copy2(source,S/'f1.py');shutil.copy2(P/'sources/l181/LICENSE',S/'LICENSE')
config={'experiment':'L187-F1-ENTITY-PRIVACY','caps':[1,5,20],'epsilons':[.5,1.,2.],'seeds':list(range(30)),
        'domain':'All constructorIds in the pinned public constructors table, fixed across neighbors',
        'adjacency':'Add/remove one driver and all declared directly owned rows; shared metadata fixed',
        'ordering':'Per-driver ascending resultId, never renumber on deletion','dtype':'float64',
        'scope':'Complete static descriptive snapshot; no forecasting split or training',
        'rng':'numpy PCG64 seeded by SeedSequence([187, cap, epsilon*10, seed]); public simulation only'}
(packet/'config.json').write_text(json.dumps(config,indent=2)+'\n')
pins={'source_revision':ledger['revision'],'historical_identity':'NOT_ESTABLISHED',
      'inherited_manifest_sha256':hashlib.sha256((P/'evidence/l181/input-manifest.json').read_bytes()).hexdigest(),
      'files':{str(p.relative_to(packet)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(packet.rglob('*')) if p.is_file()}}
(E/'input-manifest.json').write_text(json.dumps(pins,indent=2)+'\n')
urls={'privacybook.pdf':'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf',
      'node-privacy-critique.html':'https://arxiv.org/html/2311.06888v2',
      'gap-abstract.html':'https://arxiv.org/abs/2203.00949'}
receipts={}
for name,url in urls.items():
    target=S/name
    if not target.exists():
        with urllib.request.urlopen(url,timeout=45) as response:target.write_bytes(response.read())
    receipts[name]={'url':url,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
receipts['f1.py']={'url':f'https://github.com/relbench/relbench/blob/{ledger["revision"]}/relbench/datasets/f1.py','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(S/'source-ledger.json').write_text(json.dumps({'accessed':'2026-10-02','sources':receipts,
 'reading':'Dwork/Roth Definition 2.4, Definition 3.4, Theorem 3.6, Corollary 3.15; Xiang et al. v2 Section VI',
 'deviation':'No GAP/HeterPoisson implementation or published benchmark reproduction; own declared relational query experiment.'},indent=2)+'\n')
print('Pinned',len(pins['files']),'packet files and',len(receipts),'primary sources')
