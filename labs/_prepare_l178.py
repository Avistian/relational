"""Pin complete selected replay, raw database and actual released implementations."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l178';S=P/'sources/l178'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data,str(path)
    else:path.write_bytes(data)
# Complete checked-out RDBLearn tag, peeled commit rather than annotated tag-object ID.
for file in Path('/tmp/l178-rdblearn').rglob('*'):
    if file.is_file() and '.git' not in file.parts:
        freeze(S/'rdblearn'/file.relative_to('/tmp/l178-rdblearn'),file.read_bytes())
for file in Path('/tmp/l178-fastdfs/fastdfs').rglob('*.py'):
    freeze(S/'fastdfs'/file.relative_to('/tmp/l178-fastdfs/fastdfs'),file.read_bytes())
freeze(S/'fastdfs-METADATA.txt',next(Path('/tmp/l178-fastdfs').glob('*.dist-info/METADATA')).read_bytes())
freeze(S/'fastdfs-LICENSE',next(Path('/tmp/l178-fastdfs').glob('*.dist-info/licenses/LICENSE')).read_bytes())
urls={'rdb-pfn-v5.html':'https://arxiv.org/html/2603.03805v5','rdblearn-v1.html':'https://arxiv.org/html/2602.18495v1','relgnn-v2.html':'https://arxiv.org/html/2502.06784v2','pricing.html':'https://modal.com/pricing'}
for name,url in urls.items():
    if not (S/name).exists():
        with urllib.request.urlopen(url,timeout=40) as response:freeze(S/name,response.read())
for split in ['train','validation','test']:
    src=Path('/tmp/l166-data/driver-dnf')/(split+'.npz')
    pins=json.loads((P/'sources/l166/input-downloads.json').read_text())
    expected=next(x['sha256'] for x in pins if x['path'].endswith('/driver-dnf/'+split+'.npz'))
    assert sha(src)==expected;freeze(E/'released-task'/(split+'.npz'),src.read_bytes())
freeze(E/'released-task/metadata.yaml',Path('/tmp/l166-data/metadata.yaml').read_bytes())
# Authenticate the raw data against earlier sealed corpus evidence, not only a mutable cache.
prior=json.loads((P/'evidence/l171/input-manifest.json').read_text())
paths=[P/'evidence/l166/prepared.npz',P/'evidence/l166/input-manifest.json',P/'evidence/l166/report.json',P/'sources/l166/input-downloads.json',P/'_report_l166.py']
for phase in ['pilot-2','full-1']:
    folder=P/'evidence/l166'/phase;receipt=json.loads((folder/'receipt.json').read_text());paths.append(folder/'receipt.json')
    for row in receipt['records']:
        file=folder/(row['arm']+'-'+str(row['seed'])+'.npz');assert sha(file)==row['sha256'];paths.append(file)
# Database table parquet metadata preserves original key and time columns.
paths+=list((P/'evidence/l171/db').glob('*.parquet'))
for p in (P/'evidence/l171/db').glob('*.parquet'):assert sha(p)==prior['files'][p.relative_to(P).as_posix()]
paths += [P/'sources/l175/relbench-f1-task.py',P/'sources/l166/upstream/data_preprocessing/configs/dfs/dfs-2-sql.yaml',P/'relkit/relgnn_l141.py']
paths += list((S).rglob('*'));paths += list((E/'released-task').iterdir())
source=dict(rdblearn_commit='b5b03ebf8091547285a6e06cba53d2d1a40cb171',rdblearn_tag='v0.1.2',fastdfs_version='0.2.1',fastdfs_wheel_sha256='e651a3b5db092a11deab0c9f01fe5797425caacb5e312833106e508994b2b0e3',urls=urls,
            relgnn_commit='cffdb8b54627e92c7dd112c1243dde739c90d35b',rdb_pfn_commit='a95378225478daa262b85f180d482da7516b0af6',date='2026-10-02')
freeze(S/'source-ledger.json',(json.dumps(source,indent=2)+'\n').encode());paths.append(S/'source-ledger.json')
manifest=dict(experiment='L178 Matched-Information F1 Comparison',files={p.relative_to(P).as_posix():sha(p) for p in sorted(set(paths)) if p.is_file()})
freeze(E/'input-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print('Frozen',len(manifest['files']),'inputs')
