"""Pin local code, source archives, tested wrappers, data and named-lane provenance."""
import hashlib,inspect,json,platform
from pathlib import Path
import torch,torch_geometric
from torch_geometric.loader import NeighborLoader
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'date':'2026-09-20','data':json.loads((P/'_sources_l099.json').read_text())['data'],
 'rgcn_source_manifest':'_sources_l091.json','hgt_source_manifest':'_sources_l093.json','native_source_manifest':'_sources_l098.json',
 'papers':{'rgcn':'https://arxiv.org/abs/1703.06103v4','hgt':'https://arxiv.org/abs/2003.01332v1'},
 'source_revisions':{'dgl':'3d16000b4170fa741ed9e9667f22ba84d3493026','rgcn':'4bec1341dd46b72bf482f7ed26c2dca4533577f6','hgt_modern':'85eaccd482bc1d1af56c2de297b6e3a88b96d5cd','hgt_publication':'fd4a244db8efc72410537f3effec3b0c432892f7','pyg_lib':'edc9e2a88d1c5d0953b5f69c98b8365597c6b699'},
 'local_files':{},'archive_validation':{},'environment':{'python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__},
 'full_paper_parity':'NOT_ESTABLISHED','hgt_cs_training':'NOT_RUN'}
for name in ['relkit/checkpoint_l100.py','relkit/compare_l099.py','relkit/rgcn_l091.py','relkit/hgt_l093.py','_run_l100.py','_paper_l100.py','_sources_l091.json','_sources_l093.json','_sources_l098.json','_sources_l099.json']:
 manifest['local_files'][name]=sha(P/name)
# Validate inherited archived source bytes against their established manifests.
for name,folder,key in [('_sources_l091.json','sources/l091','source_files'),('_sources_l093.json','sources/hgt-l093','files')]:
 records=json.loads((P/name).read_text())[key]
 for file,digest in records.items():
  path=P/folder/file
  assert path.exists(),path
  assert sha(path)==digest,path
 manifest['archive_validation'][folder]={'status':'MATCH','files':len(records)}
for row in json.loads((P/'_sources_l098.json').read_text())['sources']:
 assert sha(P/row['file'])==row['sha256']
wrapper=Path(inspect.getfile(NeighborLoader));manifest['installed_loader_sha256']=sha(wrapper)
assert sha(wrapper)==sha(P/'sources/l098/installed-neighbor-loader.py')
assert sha(P/'data/l099/ACM.mat')==manifest['data']['sha256']
# Hash full CS bytes without unpickling; this is identity verification, not a capacity/training run.
cs=P/'data/l093/graph_CS.pk';h=hashlib.sha256()
with cs.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
expected=json.loads((P/'_sources_l093.json').read_text())['cs_data']
assert h.hexdigest()==expected['sha256'] and cs.stat().st_size==expected['bytes']
manifest['cs_bytes_verified']={'bytes':cs.stat().st_size,'sha256':h.hexdigest(),'deserialized':False}
(P/'_sources_l100.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Pinned source archives, code, native wrapper and ACM/CS bytes')
